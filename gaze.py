import math
import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

# one euro filter — see https://hal.inria.fr/hal-00670496
# tried simple ema first but it was either too laggy or too jittery,
# this adapts cutoff based on speed which is exactly what gaze needs
class _OneEuroFilter:

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x: float, dt: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self._x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0


class OneEuroFilter2D:

    def __init__(self, **kwargs):
        self._fx = _OneEuroFilter(**kwargs)
        self._fy = _OneEuroFilter(**kwargs)

    def update(self, x: float, y: float, dt: float) -> tuple[float, float]:
        return self._fx.update(x, dt), self._fy.update(y, dt)

    def reset(self):
        self._fx.reset()
        self._fy.reset()


_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def _get_model_path() -> str:
    base = os.environ.get("APPDATA", os.path.expanduser("~"))
    d = os.path.join(base, "VisioALS", "models")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "face_landmarker.task")
    if not os.path.exists(path):
        print("downloading face landmarker model...")
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, path)
        print("model downloaded ok")
    return path


def get_eye_position(landmarks, face_cx, face_cy, face_w, face_h):
    # 468/473 are mediapipe iris center indices
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    eye_x = (left_iris.x + right_iris.x) / 2
    eye_y = (left_iris.y + right_iris.y) / 2
    norm_x = (eye_x - face_cx) / face_w
    norm_y = (eye_y - face_cy) / face_h
    return norm_x, norm_y


class _LstsqModel:
    def __init__(self):
        self.coeffs = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        A = np.column_stack((Xs, np.ones(len(Xs))))
        self.coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    def predict(self, X):
        Xs = self.scaler.transform(X)
        A = np.column_stack((Xs, np.ones(len(Xs))))
        return A.dot(self.coeffs)


def train_models(calibration_data: list) -> dict:
    X = np.array([d[0] for d in calibration_data])
    y_x = np.array([d[1][0] for d in calibration_data])
    y_y = np.array([d[1][1] for d in calibration_data])

    models = {}

    lr_x, lr_y = _LstsqModel(), _LstsqModel()
    lr_x.fit(X, y_x)
    lr_y.fit(X, y_y)
    models["LR_X"] = lr_x
    models["LR_Y"] = lr_y

    poly_x = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Ridge(alpha=1.0))
    poly_y = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Ridge(alpha=1.0))
    poly_x.fit(X, y_x)
    poly_y.fit(X, y_y)
    models["POLY_X"] = poly_x
    models["POLY_Y"] = poly_y

    # C=100 tuned empirically, lower values underfit with small calibration sets
    svr_x = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale"))
    svr_y = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale"))
    svr_x.fit(X, y_x)
    svr_y.fit(X, y_y)
    models["SVR_X"] = svr_x
    models["SVR_Y"] = svr_y

    knn_x = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))
    knn_y = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))
    knn_x.fit(X, y_x)
    knn_y.fit(X, y_y)
    models["KNN_X"] = knn_x
    models["KNN_Y"] = knn_y

    print("4 model pairs trained")
    return models


class MovingDotCalibration:

    # corners visited twice so models get more data at screen edges
    # where accuracy drops off the most
    _PATH = [
        (0.50, 0.50, 0.00),
        (0.08, 0.08, 0.08),
        (0.50, 0.08, 0.15),
        (0.92, 0.08, 0.23),
        (0.92, 0.50, 0.30),
        (0.92, 0.92, 0.38),
        (0.50, 0.92, 0.45),
        (0.08, 0.92, 0.53),
        (0.08, 0.50, 0.60),
        (0.08, 0.08, 0.68),
        (0.92, 0.92, 0.78),
        (0.92, 0.08, 0.86),
        (0.08, 0.92, 0.94),
        (0.50, 0.50, 1.00),
    ]

    def __init__(self, screen_w: int, screen_h: int, duration: float = 14.0):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.duration = duration
        self.start_time: float | None = None
        self.data_points: list[tuple] = []

    def start(self):
        self.start_time = time.time()
        self.data_points = []

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def is_complete(self) -> bool:
        return self.elapsed() >= self.duration

    def progress(self) -> float:
        return min(1.0, self.elapsed() / self.duration)

    def get_dot_position(self) -> tuple[int, int] | None:
        if self.start_time is None or self.is_complete():
            return None
        frac = self.elapsed() / self.duration
        path = self._PATH
        for i in range(len(path) - 1):
            x0, y0, t0 = path[i]
            x1, y1, t1 = path[i + 1]
            if frac <= t1:
                seg_frac = (frac - t0) / (t1 - t0)
                # cosine ease — dot slows near waypoints so we get denser samples there
                ease = 0.5 * (1.0 - math.cos(math.pi * seg_frac))
                x = x0 + (x1 - x0) * ease
                y = y0 + (y1 - y0) * ease
                return int(x * self.screen_w), int(y * self.screen_h)
        return int(path[-1][0] * self.screen_w), int(path[-1][1] * self.screen_h)

    def record_sample(self, norm_x: float, norm_y: float, dot_x: int, dot_y: int):
        self.data_points.append(((norm_x, norm_y), (dot_x, dot_y)))

    def get_calibration_data(self) -> list:
        return list(self.data_points)


MODEL_NAMES = ("LR", "POLY", "SVR", "KNN")
MODEL_COLORS = {
    "LR":   (0, 0, 255),
    "POLY": (0, 255, 0),
    "SVR":  (255, 0, 0),
    "KNN":  (255, 255, 0),
}

# lr weighted higher bc its the most stable when calibration data is sparse
MODEL_WEIGHTS = {"LR": 0.35, "POLY": 0.25, "SVR": 0.25, "KNN": 0.15}


class GazeTracker:
    def __init__(self):
        model_path = _get_model_path()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.cap: cv2.VideoCapture | None = None

        self.models: dict | None = None
        self.calibration_data: list = []

        # 3-stage smoothing: input filter -> per-model ema -> output filter
        # single stage wasn't enough, cursor was either floaty or shaky.
        # splitting it lets us tune noise reduction vs responsiveness at each step
        self._input_filter = OneEuroFilter2D(min_cutoff=1.5, beta=0.01)
        # output filter more aggressive (lower cutoff) since models already smoothed things
        self._output_filter = OneEuroFilter2D(min_cutoff=0.8, beta=0.005)
        self._last_predict_time: float | None = None
        self.smoothing = 0.15
        self.model_positions: dict[str, tuple[int, int] | None] = {n: None for n in MODEL_NAMES}
        self.ensemble_position: tuple[int, int] | None = None

        self._face_lost_since: float | None = None

    def open_camera(self) -> bool:
        self.cap = cv2.VideoCapture(0)
        ok = self.cap is not None and self.cap.isOpened()
        if ok:
            print("camera opened")
        else:
            print("failed to open camera")
        return ok

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if ok:
            # mirror so it feels natural to the user
            frame = cv2.flip(frame, 1)
        return ok, frame

    def process_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)
        if not results.face_landmarks:
            return None
        lm = results.face_landmarks[0]
        face_cx = (lm[454].x + lm[234].x) / 2
        face_cy = (lm[10].y + lm[152].y) / 2
        face_w = abs(lm[454].x - lm[234].x)
        face_h = abs(lm[10].y - lm[152].y)
        if face_w < 1e-6 or face_h < 1e-6:
            return None
        return get_eye_position(lm, face_cx, face_cy, face_w, face_h)

    def predict_gaze(self, norm_x: float, norm_y: float) -> tuple[int, int] | None:
        if self.models is None:
            return None

        now = time.time()
        # fallback dt assumes ~30fps, only used on first frame
        dt = 0.033
        if self._last_predict_time is not None:
            dt = max(now - self._last_predict_time, 0.001)
        self._last_predict_time = now

        norm_x, norm_y = self._input_filter.update(norm_x, norm_y, dt)

        X_in = np.array([[norm_x, norm_y]])

        for name in MODEL_NAMES:
            px = self.models[f"{name}_X"].predict(X_in)[0]
            py = self.models[f"{name}_Y"].predict(X_in)[0]
            old = self.model_positions[name]
            if old is None:
                self.model_positions[name] = (int(px), int(py))
            else:
                s = self.smoothing
                nx = int(old[0] * (1 - s) + px * s)
                ny = int(old[1] * (1 - s) + py * s)
                self.model_positions[name] = (nx, ny)

        ex, ey = 0.0, 0.0
        for name in MODEL_NAMES:
            pos = self.model_positions[name]
            if pos is None:
                return None
            w = MODEL_WEIGHTS[name]
            ex += pos[0] * w
            ey += pos[1] * w

        fx, fy = self._output_filter.update(ex, ey, dt)
        self.ensemble_position = (int(fx), int(fy))
        return self.ensemble_position

    def get_all_model_positions(self) -> dict[str, tuple[int, int] | None]:
        return dict(self.model_positions)

    def check_face_lost(self, detected: bool) -> bool:
        # 3s threshold — shorter felt too twitchy when user just blinks or looks away briefly
        if detected:
            self._face_lost_since = None
            return False
        if self._face_lost_since is None:
            self._face_lost_since = time.time()
        return (time.time() - self._face_lost_since) > 3.0

    def reset_calibration(self):
        self.calibration_data = []
        self.models = None
        self.model_positions = {n: None for n in MODEL_NAMES}
        self.ensemble_position = None
        self._input_filter.reset()
        self._output_filter.reset()
        self._last_predict_time = None
        self._face_lost_since = None
        print("calibration reset")

    def train(self):
        if len(self.calibration_data) < 10:
            print(f"not enough calibration data ({len(self.calibration_data)} pts)")
            return
        self.models = train_models(self.calibration_data)
