"""
gaze.py — Eye tracking via MediaPipe, moving-dot calibration, ML model
training and gaze prediction.
"""

import math
import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------------------------
# FaceLandmarker model download
# ---------------------------------------------------------------------------

_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def _get_model_path() -> str:
    """Download the face_landmarker.task model to AppData if needed."""
    base = os.environ.get("APPDATA", os.path.expanduser("~"))
    d = os.path.join(base, "VisioALS", "models")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "face_landmarker.task")
    if not os.path.exists(path):
        print("[Gaze] Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, path)
        print("[Gaze] Download complete.")
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_eye_position(landmarks, face_cx, face_cy, face_w, face_h):
    """Return normalised (x, y) of the average iris centre.
    ``landmarks`` is a list of NormalizedLandmark from the tasks API."""
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    eye_x = (left_iris.x + right_iris.x) / 2
    eye_y = (left_iris.y + right_iris.y) / 2
    norm_x = (eye_x - face_cx) / face_w
    norm_y = (eye_y - face_cy) / face_h
    return norm_x, norm_y


# ---------------------------------------------------------------------------
# Least-squares linear model (kept from original code)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_models(calibration_data: list) -> dict:
    """
    Train LR, RF, GBR, KNN regressors for X and Y screen coordinates.
    ``calibration_data`` is a list of ((norm_x, norm_y), (screen_x, screen_y)).
    Returns dict like {'LR_X': model, 'LR_Y': model, ...}.
    """
    X = np.array([d[0] for d in calibration_data])
    y_x = np.array([d[1][0] for d in calibration_data])
    y_y = np.array([d[1][1] for d in calibration_data])

    models = {}

    # Linear regression (least squares)
    lr_x, lr_y = _LstsqModel(), _LstsqModel()
    lr_x.fit(X, y_x)
    lr_y.fit(X, y_y)
    models["LR_X"] = lr_x
    models["LR_Y"] = lr_y

    # Random Forest
    rf_x = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    rf_y = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    rf_x.fit(X, y_x)
    rf_y.fit(X, y_y)
    models["RF_X"] = rf_x
    models["RF_Y"] = rf_y

    # Gradient Boosting
    gbr_x = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    gbr_y = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    gbr_x.fit(X, y_x)
    gbr_y.fit(X, y_y)
    models["GBR_X"] = gbr_x
    models["GBR_Y"] = gbr_y

    # KNN (replaces SVR from old code)
    knn_x = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    knn_y = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    knn_x.fit(X, y_x)
    knn_y.fit(X, y_y)
    models["KNN_X"] = knn_x
    models["KNN_Y"] = knn_y

    print("[Gaze] All 4 model pairs trained.")
    return models


# ---------------------------------------------------------------------------
# Moving-dot calibration (Lissajous figure-8)
# ---------------------------------------------------------------------------

class MovingDotCalibration:
    """Generates a calibration path that emphasises the four screen corners
    and vertical variance for better top/bottom gaze discrimination."""

    def __init__(self, screen_w: int, screen_h: int, duration: float = 14.0):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.duration = duration
        self.cx = screen_w / 2
        self.cy = screen_h / 2
        self.ax = screen_w * 0.42
        self.ay = screen_h * 0.42
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
        """0.0 → 1.0"""
        return min(1.0, self.elapsed() / self.duration)

    def get_dot_position(self) -> tuple[int, int] | None:
        """Current dot screen position, or None if complete.

        The path visits all four corners with extra vertical sweeps so the
        ML models collect more data along the Y-axis.  The trajectory is:
          - x uses sin(2πt/T)          → one full horizontal cycle
          - y uses sin(3πt/T)          → 1.5 vertical cycles (more Y coverage)
        combined with a slight corner-biasing power curve.
        """
        if self.start_time is None or self.is_complete():
            return None
        t = self.elapsed()
        T = self.duration
        # Raw sinusoidal components
        raw_x = math.sin(2 * math.pi * t / T)
        raw_y = math.sin(3 * math.pi * t / T)
        # Cube-root bias pushes the dot toward the edges / corners
        bias_x = math.copysign(abs(raw_x) ** 0.7, raw_x)
        bias_y = math.copysign(abs(raw_y) ** 0.7, raw_y)
        x = self.cx + self.ax * bias_x
        y = self.cy + self.ay * bias_y
        return int(x), int(y)

    def record_sample(self, norm_x: float, norm_y: float, dot_x: int, dot_y: int):
        self.data_points.append(((norm_x, norm_y), (dot_x, dot_y)))

    def get_calibration_data(self) -> list:
        return list(self.data_points)


# ---------------------------------------------------------------------------
# Gaze Tracker — main interface
# ---------------------------------------------------------------------------

MODEL_NAMES = ("LR", "RF", "GBR", "KNN")
MODEL_COLORS = {
    "LR":  (0, 0, 255),     # red  (BGR for debug; UI uses its own colours)
    "RF":  (0, 255, 0),     # green
    "GBR": (255, 0, 0),     # blue
    "KNN": (255, 255, 0),   # cyan
}


class GazeTracker:
    def __init__(self):
        # New mediapipe tasks API: FaceLandmarker replaces FaceMesh
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

        # ML models (set after calibration)
        self.models: dict | None = None
        self.calibration_data: list = []

        # Smoothing
        self.smoothing = 0.2
        self.model_positions: dict[str, tuple[int, int] | None] = {n: None for n in MODEL_NAMES}

        # Face-lost timer
        self._face_lost_since: float | None = None

    # -- Camera --------------------------------------------------------------

    def open_camera(self) -> bool:
        self.cap = cv2.VideoCapture(0)
        ok = self.cap is not None and self.cap.isOpened()
        if ok:
            print("[Gaze] Camera opened.")
        else:
            print("[Gaze] Failed to open camera.")
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
            frame = cv2.flip(frame, 1)
        return ok, frame

    # -- Frame processing ----------------------------------------------------

    def process_frame(self, frame: np.ndarray):
        """
        Returns (norm_x, norm_y) or None if no face detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)
        if not results.face_landmarks:
            return None
        lm = results.face_landmarks[0]  # list of NormalizedLandmark
        face_cx = (lm[454].x + lm[234].x) / 2
        face_cy = (lm[10].y + lm[152].y) / 2
        face_w = abs(lm[454].x - lm[234].x)
        face_h = abs(lm[10].y - lm[152].y)
        if face_w < 1e-6 or face_h < 1e-6:
            return None
        return get_eye_position(lm, face_cx, face_cy, face_w, face_h)

    # -- Gaze prediction -----------------------------------------------------

    def predict_gaze(self, norm_x: float, norm_y: float) -> tuple[int, int] | None:
        """Run all models, apply smoothing, return LR position as primary."""
        if self.models is None:
            return None
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

        return self.model_positions["LR"]

    def get_all_model_positions(self) -> dict[str, tuple[int, int] | None]:
        return dict(self.model_positions)

    # -- Face-lost detection -------------------------------------------------

    def check_face_lost(self, detected: bool) -> bool:
        """Returns True when face has been lost for >3 seconds."""
        if detected:
            self._face_lost_since = None
            return False
        if self._face_lost_since is None:
            self._face_lost_since = time.time()
        return (time.time() - self._face_lost_since) > 3.0

    # -- Calibration reset ---------------------------------------------------

    def reset_calibration(self):
        self.calibration_data = []
        self.models = None
        self.model_positions = {n: None for n in MODEL_NAMES}
        self._face_lost_since = None
        print("[Gaze] Calibration reset.")

    def train(self):
        """Train models from stored calibration_data."""
        if len(self.calibration_data) < 10:
            print(f"[Gaze] Not enough calibration data ({len(self.calibration_data)} pts).")
            return
        self.models = train_models(self.calibration_data)
