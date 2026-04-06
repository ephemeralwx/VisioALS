import sys
import os
import json

from PySide6.QtWidgets import (
    QApplication, QWizard, QWizardPage, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QDialog, QProgressBar, QMessageBox,
    QWidget, QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QPixmap, QImage, QColor

import cv2
import numpy as np

from gaze import GazeTracker
from backend import BackendClient
from ui import MainWindow


def _config_dir() -> str:
    base = os.environ.get("APPDATA", os.path.expanduser("~"))
    d = os.path.join(base, "VisioALS")
    os.makedirs(d, exist_ok=True)
    return d


def _config_path() -> str:
    return os.path.join(_config_dir(), "config.json")


def _model_dir() -> str:
    d = os.path.join(_config_dir(), "models")
    os.makedirs(d, exist_ok=True)
    return d


_DEFAULTS = {
    "api_url": "https://visioals-backend.visioals.workers.dev",
    "first_run_complete": False,
    "tracking_mode": "eye",
}


def load_config() -> dict:
    p = _config_path()
    if os.path.exists(p):
        try:
            with open(p, "r") as f:
                cfg = json.load(f)
            for k, v in _DEFAULTS.items():
                cfg.setdefault(k, v)
            # force the canonical url regardless of what's saved, don't want
            # stale or hand-edited urls sneaking through
            cfg["api_url"] = _DEFAULTS["api_url"]
            return cfg
        except Exception:
            pass
    return dict(_DEFAULTS)


def save_config(cfg: dict):
    with open(_config_path(), "w") as f:
        json.dump(cfg, f, indent=2)


_STYLESHEET = """
QWizard {
    background-color: #FAFAF7;
}
QWizardPage {
    background-color: transparent;
}

QLabel {
    color: #1A1A1A;
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", sans-serif;
}
QLabel#pageTitle {
    font-size: 26px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: #111111;
    padding-bottom: 2px;
}
QLabel#pageSubtitle {
    font-size: 14px;
    font-weight: 400;
    color: #6B6B6B;
    padding-bottom: 8px;
}
QLabel#stepIndicator {
    font-size: 12px;
    font-weight: 600;
    color: #9B9B9B;
    letter-spacing: 1.5px;
}
QLabel#statusOk {
    font-size: 13px;
    font-weight: 500;
    color: #2D8A4E;
    padding: 6px 0px;
}
QLabel#statusErr {
    font-size: 13px;
    font-weight: 500;
    color: #C0392B;
    padding: 6px 0px;
}
QLabel#bulletItem {
    font-size: 14px;
    color: #3A3A3A;
    padding: 4px 0px 4px 8px;
}

QLineEdit {
    font-family: "Segoe UI", sans-serif;
    font-size: 14px;
    padding: 10px 14px;
    border: 1.5px solid #D5D5D0;
    border-radius: 10px;
    background-color: #FFFFFF;
    color: #1A1A1A;
    selection-background-color: #C8DBBE;
}
QLineEdit:focus {
    border-color: #4D8B55;
    background-color: #FFFFFF;
}
QLineEdit::placeholder {
    color: #ACACAC;
}

QPushButton {
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
    font-weight: 600;
    padding: 9px 24px;
    border: none;
    border-radius: 9px;
    background-color: #1A1A1A;
    color: #FAFAF7;
}
QPushButton:hover {
    background-color: #333333;
}
QPushButton:pressed {
    background-color: #000000;
}
QPushButton:disabled {
    background-color: #D5D5D0;
    color: #9B9B9B;
}
QPushButton#secondaryBtn {
    background-color: transparent;
    color: #1A1A1A;
    border: 1.5px solid #D5D5D0;
}
QPushButton#secondaryBtn:hover {
    background-color: #F0F0EB;
    border-color: #BBBBBB;
}

QProgressBar {
    border: none;
    border-radius: 6px;
    background-color: #E8E8E3;
    height: 10px;
    text-align: center;
    /* hides the default percentage text qt loves to show */
    font-size: 0px;
}
QProgressBar::chunk {
    border-radius: 6px;
    background-color: #4D8B55;
}

QLabel#webcamFrame {
    border: 2px solid #E0E0DB;
    border-radius: 14px;
    padding: 4px;
    background-color: #FFFFFF;
}
"""


def _make_step_label(step: int, total: int) -> QLabel:
    lbl = QLabel(f"STEP {step} OF {total}")
    lbl.setObjectName("stepIndicator")
    return lbl


def _make_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("pageTitle")
    lbl.setWordWrap(True)
    return lbl


def _make_subtitle(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("pageSubtitle")
    lbl.setWordWrap(True)
    return lbl


def _add_card_shadow(widget: QWidget):
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(24)
    shadow.setOffset(0, 4)
    shadow.setColor(QColor(0, 0, 0, 25))
    widget.setGraphicsEffect(shadow)


class WelcomePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("")
        self.setSubTitle("")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 36, 48, 24)
        layout.setSpacing(0)

        layout.addSpacing(20)
        layout.addWidget(_make_step_label(1, 2))
        layout.addSpacing(8)
        layout.addWidget(_make_title("Welcome to VisioALS"))
        layout.addSpacing(6)
        layout.addWidget(_make_subtitle(
            "Adaptive communication for ALS patients.\n"
            "We'll get you set up in just a moment."
        ))

        layout.addSpacing(28)

        steps = [
            "Check that your webcam is working",
        ]
        for i, text in enumerate(steps, 1):
            bullet = QLabel(f"{i}.  {text}")
            bullet.setObjectName("bulletItem")
            layout.addWidget(bullet)

        layout.addStretch(1)

        hint = QLabel("Press Next to begin")
        hint.setObjectName("pageSubtitle")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)


class WebcamPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("")
        self.setSubTitle("")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 36, 48, 24)
        layout.setSpacing(0)

        layout.addWidget(_make_step_label(2, 2))
        layout.addSpacing(8)
        layout.addWidget(_make_title("Webcam check"))
        layout.addSpacing(4)
        layout.addWidget(_make_subtitle(
            "We need camera access for tracking."
        ))
        layout.addSpacing(20)

        self._img_label = QLabel("Detecting camera...")
        self._img_label.setObjectName("webcamFrame")
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setMinimumHeight(220)
        _add_card_shadow(self._img_label)
        layout.addWidget(self._img_label, stretch=1)

        layout.addSpacing(12)
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status)

        self._ok = False

    def initializePage(self):
        cap = cv2.VideoCapture(0)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                # mirror so the preview feels like looking in a mirror
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(img).scaledToWidth(
                    380, Qt.SmoothTransformation
                )
                self._img_label.setPixmap(pix)
                self._status.setText("Camera detected — looking good.")
                self._status.setObjectName("statusOk")
                # qt won't re-apply objectName style without this hack
                self._status.setStyle(self._status.style())
                self._ok = True
                self.completeChanged.emit()
                return
        self._img_label.setText("No camera image")
        self._status.setText(
            "Could not open webcam. Please connect a camera and restart."
        )
        self._status.setObjectName("statusErr")
        self._status.setStyle(self._status.style())
        self._ok = False
        self.completeChanged.emit()

    def isComplete(self):
        return self._ok


class ApiUrlPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("")
        self.setSubTitle("")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 36, 48, 24)
        layout.setSpacing(0)

        layout.addWidget(_make_step_label(3, 3))
        layout.addSpacing(8)
        layout.addWidget(_make_title("Connect your server"))
        layout.addSpacing(4)
        layout.addWidget(_make_subtitle(
            "Enter the URL of your VisioALS backend so the app can "
            "generate responses and speak on your behalf."
        ))
        layout.addSpacing(24)

        field_label = QLabel("Server URL")
        field_label.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #6B6B6B; "
            "letter-spacing: 0.5px; padding-bottom: 4px;"
        )
        layout.addWidget(field_label)

        self._url_input = QLineEdit()
        self._url_input.setPlaceholderText("https://visioals-backend.visioals.workers.dev")
        self._url_input.textChanged.connect(self.completeChanged)
        layout.addWidget(self._url_input)

        layout.addSpacing(14)

        row = QHBoxLayout()
        row.setSpacing(12)
        self._test_btn = QPushButton("Test connection")
        self._test_btn.setObjectName("secondaryBtn")
        self._test_btn.clicked.connect(self._test)
        row.addWidget(self._test_btn)
        self._test_status = QLabel("")
        row.addWidget(self._test_status, stretch=1)
        layout.addLayout(row)

        layout.addStretch(1)

        self.registerField("api_url*", self._url_input)

    def _test(self):
        from backend import BackendClient as _BC
        url = self._url_input.text().strip()
        if not url:
            self._test_status.setText("Enter a URL first.")
            return
        self._test_btn.setEnabled(False)
        self._test_status.setText("Connecting...")
        self._test_status.setObjectName("pageSubtitle")
        self._test_status.setStyle(self._test_status.style())
        QApplication.processEvents()

        # skip __init__ since we just need the url for a quick health check
        tmp = _BC.__new__(_BC)
        tmp.api_url = url.rstrip("/")
        ok = False
        try:
            import requests
            r = requests.get(f"{tmp.api_url}/health", timeout=5)
            ok = r.status_code == 200
        except Exception:
            pass

        self._test_btn.setEnabled(True)
        if ok:
            self._test_status.setText("Connected")
            self._test_status.setObjectName("statusOk")
        else:
            self._test_status.setText("Could not connect — check the URL")
            self._test_status.setObjectName("statusErr")
        self._test_status.setStyle(self._test_status.style())

    def isComplete(self):
        return len(self._url_input.text().strip()) > 0


class SetupWizard(QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisioALS")
        self.setMinimumSize(560, 480)
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.setOption(QWizard.NoCancelButton, False)

        # empty pixmap kills the default header area qt insists on showing
        self.setPixmap(QWizard.LogoPixmap, QPixmap())

        self.addPage(WelcomePage())
        self.addPage(WebcamPage())

        self.setStyleSheet(_STYLESHEET)

        self.setButtonText(QWizard.NextButton, "Continue")
        self.setButtonText(QWizard.BackButton, "Back")
        self.setButtonText(QWizard.FinishButton, "Get started")
        self.setButtonText(QWizard.CancelButton, "Quit")


class _DownloadWorker(QObject):
    progress = Signal(int)
    finished = Signal(bool)

    def __init__(self, model_dir: str):
        super().__init__()
        self.model_dir = model_dir

    def run(self):
        try:
            from faster_whisper import WhisperModel
            self.progress.emit(30)
            WhisperModel("tiny.en", device="cpu", compute_type="int8",
                         download_root=self.model_dir)
            self.progress.emit(100)
            self.finished.emit(True)
        except Exception as e:
            print(f"whisper model download failed: {e}")
            self.finished.emit(False)


class ModelDownloadDialog(QDialog):
    def __init__(self, model_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VisioALS")
        self.setFixedSize(420, 220)
        self.setModal(True)
        self.setStyleSheet(_STYLESHEET + """
            QDialog { background-color: #FAFAF7; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 32, 36, 28)
        layout.setSpacing(0)

        title = QLabel("Downloading speech model")
        title.setObjectName("pageTitle")
        title.setStyleSheet("font-size: 20px;")
        layout.addWidget(title)
        layout.addSpacing(4)

        sub = QLabel("This only happens once. It'll just be a moment.")
        sub.setObjectName("pageSubtitle")
        layout.addWidget(sub)
        layout.addSpacing(20)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setFixedHeight(10)
        layout.addWidget(self._bar)
        layout.addSpacing(12)

        self._status = QLabel("Preparing download...")
        self._status.setObjectName("pageSubtitle")
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._thread = QThread()
        self._worker = _DownloadWorker(model_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._bar.setValue)
        self._worker.finished.connect(self._on_done)

    def showEvent(self, event):
        super().showEvent(event)
        self._thread.start()

    def _on_done(self, ok: bool):
        self._thread.quit()
        # 3s grace period so the thread doesn't get destroyed mid-cleanup
        self._thread.wait(3000)
        if ok:
            self._status.setText("Ready.")
            self._status.setObjectName("statusOk")
            self._status.setStyle(self._status.style())
            self.accept()
        else:
            self._status.setText("Download failed. Check your internet connection.")
            self._status.setObjectName("statusErr")
            self._status.setStyle(self._status.style())
            QMessageBox.warning(self, "Error", "Could not download the speech model.")
            self.reject()


def _model_exists(model_dir: str) -> bool:
    # rough check, just looks for .bin/.ct2 files anywhere under model dir.
    # not bulletproof but faster_whisper re-downloads if anything is actually missing
    if not os.path.isdir(model_dir):
        return False
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith((".bin", ".ct2")):
                return True
    return False


def main():
    app = QApplication(sys.argv)
    # fusion looks consistent across windows versions, native style is a mess
    app.setStyle("Fusion")

    cfg = load_config()

    # cli override for tracking mode, used by installer's first launch
    for arg in sys.argv[1:]:
        if arg.startswith("--tracking-mode="):
            cfg["tracking_mode"] = arg.split("=", 1)[1]

    if not cfg.get("first_run_complete"):
        wiz = SetupWizard()
        if wiz.exec() != QWizard.Accepted:
            sys.exit(0)
        cfg["first_run_complete"] = True
        save_config(cfg)

    mdir = _model_dir()
    if not _model_exists(mdir):
        dlg = ModelDownloadDialog(mdir)
        if dlg.exec() != QDialog.Accepted:
            sys.exit(0)

    gaze = GazeTracker(mode=cfg.get("tracking_mode", "eye"))
    if not gaze.open_camera():
        QMessageBox.critical(None, "Error", "Cannot open webcam.")
        sys.exit(1)

    backend = BackendClient(api_url=cfg["api_url"], model_dir=mdir)

    win = MainWindow(gaze, backend, cfg, save_config)
    win.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
