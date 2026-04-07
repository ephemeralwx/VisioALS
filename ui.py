import time
from PySide6.QtWidgets import QMainWindow, QWidget
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, QRectF
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush

from gaze import GazeTracker, MovingDotCalibration, MODEL_NAMES
from backend import BackendClient


class _Worker(QObject):
    finished = Signal(object)

    def __init__(self, fn, *args):
        super().__init__()
        self._fn = fn
        self._args = args

    def run(self):
        try:
            result = self._fn(*self._args)
        except Exception as e:
            print(f"worker error: {e}")
            result = None
        self.finished.emit(result)


def _run_in_thread(fn, callback, *args) -> tuple[QThread, _Worker]:
    thread = QThread()
    worker = _Worker(fn, *args)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(callback)
    # prevent leaks - qt wont clean these up on its own
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker


BG = QColor(255, 255, 255)
TEXT_COLOR = QColor(30, 30, 30)
QUAD_COLORS = {
    "top_left":     QColor(200, 235, 200),
    "top_right":    QColor(200, 200, 235),
    "bottom_left":  QColor(235, 200, 200),
    "bottom_right": QColor(235, 235, 200),
}
NONE_COLOR = QColor(220, 200, 240)
PROGRESS_BG = QColor(200, 200, 200)
PROGRESS_FG = QColor(60, 180, 60)
REC_COLOR = QColor(220, 40, 40)

# alpha 100 so overlapping blobs are still distinguishable
BLOB_COLORS = {
    "LR":   QColor(220, 60, 60, 100),
    "POLY": QColor(60, 180, 60, 100),
    "SVR":  QColor(60, 60, 220, 100),
    "KNN":  QColor(60, 200, 200, 100),
}

# tuned by trial and error, 1.0 felt too twitchy for most users
DWELL_THRESHOLD = 1.2


class GazeScreen(QWidget):
    def __init__(self, gaze_tracker: GazeTracker, backend: BackendClient):
        super().__init__()
        self.gaze = gaze_tracker
        self.backend = backend

        self.sw = 800
        self.sh = 600
        self.quads = {}
        self.none_rect = (0, 0, 0, 0)
        self._geometry_ready = False

        self.phase = "waiting"
        self.calibration: MovingDotCalibration | None = None

        self.responses: list[str] = []
        self.context: str | None = None
        self.is_recording = False
        self.status_text = ""

        self._conversation_history: list[dict] = []  # [{question, answer}, ...]
        self._rejected_options: list[str] = []

        # patient identity & preference context
        self._patient_data = None          # PatientDataManager | None
        self._corpus_index = None          # CorpusIndex | None
        self._linguistic_profile: dict | None = None
        self._preference_rules: list[str] | None = None
        self._interaction_counter: int = 0
        self._preference_update_interval: int = 20
        self._rejection_round: int = 0

        self.timers = {k: 0.0 for k in ["top_left", "top_right", "bottom_left", "bottom_right", "none"]}
        self._last_tick = time.time()

        # prevent gc from killing qthreads mid-flight
        self._threads: list = []

        # without this, dwell timer fires multiple times during api call
        self._selection_locked = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def set_patient_context(self, patient_data, corpus_index=None):
        """Load patient identity context for prompt personalization."""
        self._patient_data = patient_data
        self._corpus_index = corpus_index
        self._linguistic_profile = patient_data.load_linguistic_profile() if patient_data else None
        pref = patient_data.load_preference_profile() if patient_data else None
        self._preference_rules = pref.get("rules", []) if pref else None
        self._interaction_counter = patient_data.interaction_count() if patient_data else 0
        print(f"patient context loaded: profile={'yes' if self._linguistic_profile else 'no'}, "
              f"index={'yes' if corpus_index else 'no'}, "
              f"prefs={len(self._preference_rules) if self._preference_rules else 0} rules, "
              f"interactions={self._interaction_counter}")

    def _get_identity_kwargs(self, include_prefs: bool = True) -> dict:
        """Build the identity keyword args for backend API calls."""
        kwargs = {}
        if self._linguistic_profile:
            kwargs["linguistic_profile_summary"] = self._linguistic_profile.get("summary")
        if self._corpus_index and self.context:
            try:
                kwargs["exemplars"] = self._corpus_index.retrieve(self.context, top_k=5)
            except Exception as e:
                print(f"exemplar retrieval failed: {e}")
        if include_prefs and self._preference_rules:
            kwargs["preference_rules"] = self._preference_rules
        return kwargs

    def _log_and_maybe_update(self, question, options_presented, selected, rejected, rejection_round):
        """Log interaction and trigger preference update if needed."""
        if not self._patient_data:
            return
        self._patient_data.log_interaction({
            "question": question,
            "options_presented": options_presented,
            "selected": selected,
            "rejected": rejected,
            "rejection_round": rejection_round,
        })
        self._interaction_counter += 1

        if (self._interaction_counter >= 20 and
                self._interaction_counter % self._preference_update_interval == 0):
            interactions = self._patient_data.load_interactions(last_n=100)

            def do_analysis(data):
                return self.backend.analyze_preferences(data)

            def on_prefs(result):
                if result:
                    self._preference_rules = result
                    self._patient_data.save_preference_profile({
                        "rules": result,
                        "last_updated": __import__("time").time(),
                        "interaction_count": self._interaction_counter,
                    })
                    print(f"preference rules updated: {result}")

            ref = _run_in_thread(do_analysis, on_prefs, interactions)
            self._threads.append(ref)

    def _update_geometry(self):
        self.sw = self.width()
        self.sh = self.height()
        # 30% felt right visually, not derived from anything scientific
        qw = self.sw * 0.3
        qh = self.sh * 0.3
        self.quads = {
            "top_left":     (0, 0, qw, qh),
            "top_right":    (self.sw - qw, 0, self.sw, qh),
            "bottom_left":  (0, self.sh - qh, qw, self.sh),
            "bottom_right": (self.sw - qw, self.sh - qh, self.sw, self.sh),
        }
        none_w = qw * 1.0
        # shorter than response quads so it doesnt compete for attention
        none_h = qh * 0.5
        half_w = self.sw / 2
        self.none_rect = (half_w - none_w / 2, 10,
                          half_w + none_w / 2, 10 + none_h)
        self._geometry_ready = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_geometry()

    def _tick(self):
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now

        ok, frame = self.gaze.get_frame()
        if not ok:
            return

        norm = self.gaze.process_frame(frame)
        detected = norm is not None

        if self.phase == "waiting":
            self.update()
            return

        if self.phase == "calibration":
            if self.calibration is None:
                self.update()
                return
            if detected and not self.calibration.is_complete():
                dot = self.calibration.get_dot_position()
                if dot is not None:
                    self.calibration.record_sample(norm, dot[0], dot[1])
            if self.calibration.is_complete():
                data = self.calibration.get_calibration_data()
                self.gaze.calibration_data = data
                self.gaze.train()
                if self.gaze.models is not None:
                    self.phase = "tracking"
                    self.status_text = "Calibration complete. Press SPACE to ask a question."
                    print("calibration done, tracking now")
                else:
                    self._restart_calibration()
            self.update()
            return

        if self.gaze.check_face_lost(detected):
            # force full recalibration because head probably moved
            self.status_text = "Face lost — press SPACE to recalibrate."
            self._restart_calibration()
            self.update()
            return

        gaze_pos = None
        if detected:
            gaze_pos = self.gaze.predict_gaze(norm[0], norm[1])

        if gaze_pos is not None and self.responses and not self._selection_locked:
            gx, gy = gaze_pos
            for key, (x1, y1, x2, y2) in self.quads.items():
                if x1 <= gx <= x2 and y1 <= gy <= y2:
                    self.timers[key] += dt
                    if self.timers[key] >= DWELL_THRESHOLD:
                        self._on_selection(key)
                        break
                else:
                    self.timers[key] = 0.0

            nx1, ny1, nx2, ny2 = self.none_rect
            if nx1 <= gx <= nx2 and ny1 <= gy <= ny2:
                self.timers["none"] += dt
                if self.timers["none"] >= DWELL_THRESHOLD:
                    self._on_none_of_these()
            else:
                self.timers["none"] = 0.0
        else:
            for k in self.timers:
                self.timers[k] = 0.0

        self.update()

    def _restart_calibration(self):
        self.phase = "waiting"
        self.gaze.reset_calibration()
        self.calibration = None
        for k in self.timers:
            self.timers[k] = 0.0

    def _begin_calibration(self):
        self.phase = "calibration"
        self.gaze.reset_calibration()
        self.calibration = MovingDotCalibration(self.sw, self.sh)
        self.calibration.start()

    def _on_selection(self, direction: str):
        idx = {"top_left": 0, "top_right": 1, "bottom_left": 2, "bottom_right": 3}.get(direction)
        if idx is None or idx >= len(self.responses):
            return
        selected = self.responses[idx]
        print(f"selected {direction}: {selected}")
        self._selection_locked = True
        self._reset_timers()
        self.status_text = f"Selected: {selected}"

        question = self.context
        history = list(self._conversation_history)
        options_snapshot = list(self.responses)
        self._conversation_history.append({"question": question, "answer": selected})
        rejected_snapshot = list(self._rejected_options)
        self._rejected_options = []

        # log interaction and maybe update preferences
        self._log_and_maybe_update(
            question, options_snapshot, selected, rejected_snapshot, self._rejection_round,
        )
        self._rejection_round = 0

        identity = self._get_identity_kwargs(include_prefs=False)

        def do_expand():
            return self.backend.expand_response(question, selected, history, **identity)

        def on_expanded(result):
            text = result if result else selected
            self.backend.speak(text)
            self.responses = []
            self.context = None
            self._selection_locked = False
            self.status_text = "Press SPACE to ask a new question."
            self.update()

        ref = _run_in_thread(do_expand, on_expanded)
        self._threads.append(ref)

    def _on_none_of_these(self):
        if not self.context:
            return
        print("none selected, fetching new options")
        self._selection_locked = True
        self._reset_timers()
        self.status_text = "Getting new options..."

        options_snapshot = list(self.responses)
        self._rejected_options.extend(self.responses)
        self.responses = []

        # log the rejection
        self._log_and_maybe_update(
            self.context, options_snapshot, None, options_snapshot, self._rejection_round,
        )
        self._rejection_round += 1

        rejected = list(self._rejected_options)
        history = list(self._conversation_history)
        identity = self._get_identity_kwargs(include_prefs=True)

        def do_generate():
            return self.backend.generate_options(
                self.context, history, rejected, **identity,
            )

        def on_new_options(result):
            if result:
                self.responses = result
            self._selection_locked = False
            self.status_text = ""
            self.update()

        ref = _run_in_thread(do_generate, on_new_options)
        self._threads.append(ref)

    def _reset_timers(self):
        for k in self.timers:
            self.timers[k] = 0.0

    def on_question_ready(self, question: str):
        self.context = question
        self.responses = []
        self._rejected_options = []
        self._rejection_round = 0
        self.status_text = "Getting responses..."
        self.update()

        history = list(self._conversation_history)
        identity = self._get_identity_kwargs(include_prefs=True)

        def do_generate():
            return self.backend.generate_options(question, history, **identity)

        def on_options(result):
            if result:
                self.responses = result
            self.status_text = ""
            self.update()

        ref = _run_in_thread(do_generate, on_options)
        self._threads.append(ref)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)

        if self.phase == "waiting":
            self._paint_waiting(p)
        elif self.phase == "calibration":
            self._paint_calibration(p)
        else:
            self._paint_tracking(p)
        p.end()

    def _paint_waiting(self, p: QPainter):
        p.setFont(QFont("Segoe UI", 22))
        p.setPen(TEXT_COLOR)
        p.drawText(self.rect(), Qt.AlignCenter, "Press SPACE to begin calibration")

        mode_label = "Eye Tracking" if self.gaze.mode == "eye" else "Head Tracking"
        p.setFont(QFont("Segoe UI", 14))
        p.setPen(QColor(100, 100, 100))
        p.drawText(QRectF(0, self.sh * 0.58, self.sw, 30), Qt.AlignCenter,
                   f"Mode: {mode_label}  \u2022  Press M to switch")

    def _paint_calibration(self, p: QPainter):
        if self.calibration is None:
            p.setFont(QFont("Segoe UI", 16))
            p.setPen(TEXT_COLOR)
            p.drawText(self.rect(), Qt.AlignCenter, "Starting calibration...")
            return

        p.setFont(QFont("Segoe UI", 20))
        p.setPen(TEXT_COLOR)
        cal_hint = "Follow the dot with your head" if self.gaze.mode == "head" else "Follow the dot with your eyes"
        p.drawText(QRectF(0, self.sh * 0.05, self.sw, 40), Qt.AlignCenter, cal_hint)

        dot = self.calibration.get_dot_position()
        if dot is not None:
            p.setBrush(QColor(50, 120, 220))
            p.setPen(Qt.NoPen)
            p.drawEllipse(dot[0] - 18, dot[1] - 18, 36, 36)

        prog = self.calibration.progress()
        bar_w = self.sw * 0.4
        bar_h = 16
        bx = (self.sw - bar_w) / 2
        by = self.sh * 0.92
        p.setPen(QPen(QColor(160, 160, 160), 1))
        p.setBrush(PROGRESS_BG)
        p.drawRoundedRect(QRectF(bx, by, bar_w, bar_h), 6, 6)
        p.setBrush(PROGRESS_FG)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(bx, by, bar_w * prog, bar_h), 6, 6)

    def _paint_tracking(self, p: QPainter):
        resp_font = QFont("Segoe UI", 16, QFont.Bold)
        small_font = QFont("Segoe UI", 12)
        dir_to_idx = {"top_left": 0, "top_right": 1, "bottom_left": 2, "bottom_right": 3}

        for key, (x1, y1, x2, y2) in self.quads.items():
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            p.setBrush(QBrush(QUAD_COLORS[key]))
            p.setPen(QPen(QColor(180, 180, 180), 1))
            p.drawRoundedRect(rect, 12, 12)

            idx = dir_to_idx[key]
            if idx < len(self.responses):
                p.setFont(resp_font)
                p.setPen(TEXT_COLOR)
                p.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, self.responses[idx])

            if self.timers[key] > 0:
                prog = min(1.0, self.timers[key] / DWELL_THRESHOLD)
                bar_len = (x2 - x1) * 0.5
                bar_h = 8
                bar_x = x1 + (x2 - x1 - bar_len) / 2
                bar_y = y2 - 20
                p.setBrush(PROGRESS_BG)
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(QRectF(bar_x, bar_y, bar_len, bar_h), 3, 3)
                p.setBrush(PROGRESS_FG)
                p.drawRoundedRect(QRectF(bar_x, bar_y, bar_len * prog, bar_h), 3, 3)

        # 4px inset so it doesnt visually merge with the corner quadrants
        nx1, ny1, nx2, ny2 = self.none_rect
        none_r = QRectF(nx1 + 4, ny1 + 4, nx2 - nx1 - 8, ny2 - ny1 - 8)
        p.setBrush(QBrush(NONE_COLOR))
        p.setPen(QPen(QColor(150, 130, 170), 1))
        p.drawRoundedRect(none_r, 8, 8)
        p.setFont(resp_font)
        p.setPen(TEXT_COLOR)
        p.drawText(none_r, Qt.AlignCenter, "None of these")

        if self.timers["none"] > 0:
            prog = min(1.0, self.timers["none"] / DWELL_THRESHOLD)
            p.setBrush(PROGRESS_FG)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(nx1 + 4, ny2 - 10, (nx2 - nx1 - 8) * prog, 5), 2, 2)

        positions = self.gaze.get_all_model_positions()
        for name in MODEL_NAMES:
            pos = positions.get(name)
            if pos is not None:
                p.setBrush(QBrush(BLOB_COLORS[name]))
                p.setPen(Qt.NoPen)
                p.drawEllipse(pos[0] - 20, pos[1] - 20, 40, 40)

        if self.is_recording:
            p.setBrush(QBrush(REC_COLOR))
            p.setPen(Qt.NoPen)
            p.drawEllipse(self.sw - 80, 20, 16, 16)
            p.setFont(QFont("Segoe UI", 13, QFont.Bold))
            p.setPen(REC_COLOR)
            p.drawText(self.sw - 60, 35, "REC")

        if self.context:
            p.setFont(QFont("Segoe UI", 14))
            p.setPen(QColor(80, 80, 80))
            q_rect = QRectF(self.sw * 0.2, self.sh * 0.46, self.sw * 0.6, 40)
            p.drawText(q_rect, Qt.AlignCenter, f"Q: {self.context}")

        if self.status_text:
            p.setFont(QFont("Segoe UI", 13))
            p.setPen(QColor(100, 100, 100))
            p.drawText(QRectF(0, self.sh * 0.52, self.sw, 30), Qt.AlignCenter, self.status_text)


class MainWindow(QMainWindow):
    def __init__(self, gaze_tracker: GazeTracker, backend: BackendClient,
                 cfg=None, save_config_fn=None,
                 patient_data=None, corpus_index=None):
        super().__init__()
        self.setWindowTitle("VisioALS")
        self.gaze = gaze_tracker
        self.backend = backend
        self._cfg = cfg
        self._save_config = save_config_fn
        self._patient_data = patient_data

        self.screen = GazeScreen(gaze_tracker, backend)
        if patient_data:
            self.screen.set_patient_context(patient_data, corpus_index)
        self.setCentralWidget(self.screen)

        self._session_start = time.time()
        self._threads: list = []

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Space and self.screen.phase == "waiting":
            self.screen._begin_calibration()
            self.screen.update()
            return

        if key == Qt.Key_Space and self.screen.phase == "tracking":
            if not self.screen.is_recording:
                self.backend.start_recording()
                self.screen.is_recording = True
                self.screen.status_text = "Recording... press SPACE to stop."
            else:
                audio = self.backend.stop_recording()
                self.screen.is_recording = False
                if audio is not None and len(audio) > 0:
                    self.screen.status_text = "Transcribing..."
                    self.screen.update()

                    def on_transcribed(text):
                        if text:
                            print(f"transcribed: {text}")
                            self.screen.on_question_ready(text)
                        else:
                            self.screen.status_text = "Could not transcribe. Try again."
                            self.screen.update()

                    ref = _run_in_thread(self.backend.transcribe, on_transcribed, audio)
                    self._threads.append(ref)
                else:
                    self.screen.status_text = "No audio recorded. Press SPACE to try again."
            self.screen.update()

        elif key == Qt.Key_M and self.screen.phase in ("waiting", "tracking"):
            new_mode = "head" if self.gaze.mode == "eye" else "eye"
            self.gaze.set_mode(new_mode)
            self.screen._restart_calibration()
            mode_label = "Head Tracking" if new_mode == "head" else "Eye Tracking"
            self.screen.status_text = f"Switched to {mode_label}. Press SPACE to calibrate."
            if self._cfg is not None and self._save_config is not None:
                self._cfg["tracking_mode"] = new_mode
                self._save_config(self._cfg)
            self.screen.update()

        elif key == Qt.Key_R:
            self.screen._restart_calibration()

        elif key == Qt.Key_F11:
            if self.isFullScreen():
                self.showMaximized()
            else:
                self.showFullScreen()

        elif key == Qt.Key_P and self.screen.phase in ("waiting", "tracking"):
            from main import PatientOnboardingDialog
            dlg = PatientOnboardingDialog(self.backend, parent=self)
            if dlg.exec() == 1:  # QDialog.Accepted
                patient_name = dlg.patient_name()
                if patient_name and self._cfg is not None:
                    self._cfg["active_patient"] = patient_name
                    if self._save_config:
                        self._save_config(self._cfg)
                    from patient_data import PatientDataManager
                    from embeddings import EmbeddingProvider, CorpusIndex
                    pd = PatientDataManager(patient_name)
                    ci = None
                    if pd.load_linguistic_profile() is not None:
                        provider = EmbeddingProvider()
                        ci = CorpusIndex(pd.embeddings_dir, provider)
                        if ci.is_built():
                            ci.load_index()
                        else:
                            ci = None
                    self._patient_data = pd
                    self.screen.set_patient_context(pd, ci)
                    self.screen.status_text = f"Patient profile loaded: {patient_name}"
                    self.screen.update()

        elif key == Qt.Key_V and self.screen.phase in ("waiting", "tracking"):
            if self.screen._patient_data:
                from main import PreferenceViewerDialog
                dlg = PreferenceViewerDialog(self.screen._patient_data, parent=self)
                dlg.exec()

        elif key == Qt.Key_Q or key == Qt.Key_Escape:
            self.close()

        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.screen._timer.stop()
        self.gaze.release_camera()
        # fire and forget telemetry, lazy import bc only needed on shutdown
        if hasattr(self, '_session_start'):
            import threading, requests as _req
            duration = time.time() - self._session_start
            url = self.backend.api_url + "/telemetry"
            threading.Thread(
                target=lambda: _req.post(url, json={"duration_seconds": round(duration)}, timeout=5),
                daemon=True,
            ).start()
            # ugly but need the request to actually leave before process dies
            import time as _t; _t.sleep(0.5)
        self.backend.shutdown()
        event.accept()
