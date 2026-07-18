import sys
import os
import json

from PySide6.QtWidgets import (
    QApplication, QWizard, QWizardPage, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QDialog, QProgressBar, QMessageBox,
    QWidget, QGraphicsDropShadowEffect, QFileDialog, QTextEdit,
    QScrollArea, QFrame, QComboBox,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QRectF, QEvent, Property,
    QPropertyAnimation, QEasingCurve, QSignalBlocker,
)
from PySide6.QtGui import QPixmap, QImage, QColor, QPainter

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
    "active_patient": None,
    "preference_update_interval": 20,
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
QLabel#statusWarn {
    font-size: 13px;
    font-weight: 500;
    color: #A76400;
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


class _OnboardingWorker(QObject):
    progress = Signal(int)
    finished = Signal(object)  # emits the profile dict or None

    def __init__(self, patient_name: str, api_url: str, model_dir: str,
                 backend, elevenlabs_api_key: str):
        super().__init__()
        self._patient_name = patient_name
        self._api_url = api_url
        self._model_dir = model_dir
        self._backend = backend
        self._elevenlabs_api_key = (elevenlabs_api_key or "").strip()

    def run(self):
        try:
            from patient_data import PatientDataManager
            from linguistic_profile import LinguisticProfileExtractor
            from embeddings import EmbeddingProvider, CorpusIndex

            pm = PatientDataManager(self._patient_name)
            media_paths = pm.list_media_files()

            # Historical recordings contribute their transcripts to the same
            # corpus used by linguistic profiling and semantic retrieval.
            for index, media_path in enumerate(media_paths):
                transcript = self._backend.transcribe_file(media_path)
                pm.save_media_transcript(media_path, transcript)
                if media_paths:
                    self.progress.emit(5 + int(((index + 1) / len(media_paths)) * 25))

            voice_profile = pm.load_voice_profile()
            if media_paths:
                if not self._elevenlabs_api_key:
                    raise ValueError(
                        "Enter an ElevenLabs API key to create the patient's voice clone."
                    )
                fingerprint = pm.media_fingerprint()
                if not voice_profile or voice_profile.get("media_fingerprint") != fingerprint:
                    clone = self._backend.clone_voice(
                        self._patient_name,
                        media_paths,
                        self._elevenlabs_api_key,
                    )
                    voice_profile = {
                        "provider": "elevenlabs",
                        "voice_id": clone["voice_id"],
                        "requires_verification": bool(clone.get("requires_verification", False)),
                        "media_fingerprint": fingerprint,
                        "sample_count": len(media_paths),
                        "source_files": [os.path.basename(path) for path in media_paths],
                        "created_at": __import__("time").time(),
                    }
                    pm.save_voice_profile(voice_profile)
                self._backend.configure_elevenlabs(
                    self._elevenlabs_api_key,
                    voice_profile.get("voice_id", ""),
                )
            self.progress.emit(35)

            corpus = pm.load_corpus()
            if not corpus:
                self.finished.emit({"error": "Add at least one text, audio, or video file."})
                return

            # download spacy model if needed (first time only)
            try:
                import spacy
                spacy.load("en_core_web_sm")
            except OSError:
                print("downloading spacy en_core_web_sm...")
                from spacy.cli import download
                download("en_core_web_sm")
            self.progress.emit(42)

            # download embedding model if needed (first time only)
            provider = EmbeddingProvider(cache_dir=self._model_dir)
            if not provider.model_ready():
                provider.download_model(
                    progress_callback=lambda p: self.progress.emit(42 + int(p * 0.08))
                )
            self.progress.emit(50)

            # generate linguistic profile
            extractor = LinguisticProfileExtractor(corpus, self._api_url)
            profile = extractor.extract(
                progress_callback=lambda p: self.progress.emit(50 + int(p * 0.30))
            )
            pm.save_linguistic_profile(profile)
            self.progress.emit(80)

            # build embedding index
            index = CorpusIndex(pm.embeddings_dir, provider)
            index.build_index(
                corpus,
                progress_callback=lambda p: self.progress.emit(80 + int(p * 0.20))
            )
            self.progress.emit(100)
            self.finished.emit({
                "profile": profile,
                "voice_profile": voice_profile,
                "media_count": len(media_paths),
            })
        except Exception as e:
            print(f"onboarding failed: {e}")
            import traceback; traceback.print_exc()
            self.finished.emit({"error": str(e)})


_PATIENT_STUDIO_STYLESHEET = """
QDialog#patientStudio {
    background-color: #101011;
}
QFrame#studioShell {
    background-color: #1b1b1d;
    border: 1px solid #2d2d30;
    border-radius: 22px;
}
QFrame#studioRail {
    background-color: #151516;
    border: none;
    border-top-left-radius: 21px;
    border-bottom-left-radius: 21px;
}
QWidget#studioMain {
    background: transparent;
}
QLabel {
    color: #f3f3f5;
    font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue", sans-serif;
}
QLabel#studioRailTitle {
    font-size: 16px;
    font-weight: 650;
    color: #f4f4f6;
}
QLabel#studioRailCaption {
    color: #6d6d73;
    font-size: 10px;
    font-weight: 600;
}
QFrame#studioStageActive {
    background-color: #252527;
    border: none;
    border-radius: 12px;
}
QFrame#studioStageMuted {
    background-color: transparent;
    border: none;
    border-radius: 12px;
}
QLabel#studioStageIcon {
    min-width: 26px;
    max-width: 26px;
    min-height: 26px;
    max-height: 26px;
    border-radius: 8px;
    font-size: 10px;
    font-weight: 700;
}
QFrame#studioStageActive QLabel#studioStageIcon {
    color: #d7d7dc;
    background-color: #343438;
}
QFrame#studioStageMuted QLabel#studioStageIcon {
    color: #68686e;
    background-color: #222224;
}
QLabel#studioStageText {
    font-size: 12px;
    font-weight: 600;
}
QFrame#studioStageActive QLabel#studioStageText {
    color: #f4f4f5;
}
QFrame#studioStageMuted QLabel#studioStageText {
    color: #77777d;
}
QLabel#studioStageDetail {
    color: #66666d;
    font-size: 10px;
    font-weight: 500;
}
QFrame#studioStageActive QLabel#studioStageDetail {
    color: #9b98aa;
}
QLabel#studioRailMeta {
    color: #66666c;
    font-size: 10px;
}
QLabel#studioRailShortcut {
    min-width: 22px;
    max-width: 22px;
    min-height: 20px;
    max-height: 20px;
    border-radius: 6px;
    color: #aaaab0;
    background-color: #242426;
    border: 1px solid #303034;
    font-size: 10px;
    font-weight: 650;
}
QLabel#studioTitle {
    color: #fafafd;
    font-size: 31px;
    font-weight: 700;
}
QLabel#studioSubtitle {
    color: #8c8c93;
    font-size: 13px;
    line-height: 1.4;
}
QLabel#studioDraftBadge {
    color: #9a9aa1;
    background-color: #242426;
    border: 1px solid #343438;
    border-radius: 9px;
    padding: 5px 9px;
    font-size: 9px;
    font-weight: 700;
}
QLabel#studioEyebrow {
    color: #77747f;
    font-size: 9px;
    font-weight: 700;
}
QLabel#studioSectionTitle {
    color: #ebebee;
    font-size: 15px;
    font-weight: 650;
}
QLabel#studioSectionHint {
    color: #74747a;
    font-size: 11px;
}
QLabel#studioFieldLabel {
    color: #a1a1a7;
    font-size: 11px;
    font-weight: 600;
}
QFrame#studioSourceSurface {
    background-color: #222224;
    border: 1px solid #2b2b2f;
    border-radius: 16px;
}
QFrame#studioProgressCard {
    background-color: #171719;
    border: 1px solid #29292d;
    border-radius: 15px;
}
QLineEdit#studioInput, QComboBox#studioProfilePicker, QTextEdit#studioTextInput {
    background-color: #232325;
    color: #f0f0f2;
    border: 1px solid transparent;
    border-radius: 11px;
    padding: 10px 12px;
    font-family: "SF Pro Text", "Segoe UI", sans-serif;
    font-size: 13px;
    selection-background-color: #55555b;
}
QLineEdit#studioInput:focus, QComboBox#studioProfilePicker:focus, QTextEdit#studioTextInput:focus {
    border: 1px solid #68686f;
    background-color: #242428;
}
QComboBox#studioProfilePicker {
    padding: 7px 12px;
}
QComboBox#studioProfilePicker::drop-down {
    border: none;
    width: 28px;
}
QComboBox#studioProfilePicker QAbstractItemView {
    color: #f0f0f2;
    background-color: #232325;
    border: 1px solid #3a3a3f;
    selection-background-color: #3b3b40;
    outline: none;
}
QLineEdit#studioInput:disabled, QTextEdit#studioTextInput:disabled {
    color: #606066;
    background-color: #202022;
}
QPushButton {
    font-family: "SF Pro Text", "Segoe UI", sans-serif;
    border-radius: 10px;
    padding: 7px 13px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton#studioAction {
    min-height: 20px;
    color: #d7d7dc;
    background-color: #2e2e32;
    border: 1px solid #3a3a3f;
}
QPushButton#studioAction:hover {
    color: #ffffff;
    background-color: #38383d;
    border-color: #4b4b51;
}
QPushButton#studioAction:pressed {
    background-color: #29292d;
}
QPushButton#studioAction:disabled {
    color: #66666b;
    background-color: #28282b;
    border-color: #303034;
}
QPushButton#studioPrimary {
    min-width: 148px;
    min-height: 23px;
    color: #171719;
    background-color: #ededf0;
    border: 1px solid #ffffff;
}
QPushButton#studioPrimary:hover {
    background-color: #ffffff;
}
QPushButton#studioPrimary:pressed {
    background-color: #d4d4d8;
}
QPushButton#studioPrimary:disabled {
    color: #68686e;
    background-color: #29292c;
    border-color: #303034;
}
QPushButton#studioSecondary {
    color: #898990;
    background-color: transparent;
    border: 1px solid transparent;
}
QPushButton#studioSecondary:hover {
    color: #e5e5e8;
    background-color: #252527;
}
QLabel#studioStatus, QLabel#pageSubtitle {
    color: #7f7f86;
    font-size: 11px;
}
QLabel#statusOk {
    color: #69c99e;
    font-size: 11px;
    font-weight: 600;
}
QLabel#statusWarn {
    color: #dfae6c;
    font-size: 11px;
    font-weight: 600;
}
QLabel#statusErr {
    color: #f17178;
    font-size: 11px;
    font-weight: 600;
}
QLabel#studioProgressValue {
    color: #b8b8be;
    font-size: 10px;
    font-weight: 700;
}
QTextEdit#studioSummary {
    color: #d4d4d8;
    background-color: #202022;
    border: 1px solid #2e2e32;
    border-radius: 10px;
    padding: 8px 10px;
    font-size: 11px;
}
QScrollBar:vertical {
    background: transparent;
    width: 7px;
}
QScrollBar::handle:vertical {
    background: #45454a;
    border-radius: 4px;
    min-height: 24px;
}
QToolTip {
    color: #f4f4f5;
    background-color: #28282b;
    border: 1px solid #3a3a3f;
    padding: 6px 8px;
}
"""


class _InteractionEffect(QObject):
    """Small native animations for focus glow and hover elevation."""

    def __init__(self, widget: QWidget, mode: str):
        super().__init__(widget)
        self._widget = widget
        self._mode = mode
        self._effect = QGraphicsDropShadowEffect(widget)
        self._effect.setBlurRadius(0)
        self._effect.setOffset(0, 0 if mode == "focus" else 1)
        self._effect.setColor(
            QColor(10, 132, 255, 105) if mode == "focus" else QColor(0, 0, 0, 105)
        )
        widget.setGraphicsEffect(self._effect)
        widget.installEventFilter(self)

        self._blur = QPropertyAnimation(self._effect, b"blurRadius", self)
        self._blur.setDuration(150)
        self._blur.setEasingCurve(QEasingCurve.OutCubic)
        self._offset = QPropertyAnimation(self._effect, b"yOffset", self)
        self._offset.setDuration(150)
        self._offset.setEasingCurve(QEasingCurve.OutCubic)

    def _animate(self, active: bool):
        self._blur.stop()
        self._blur.setStartValue(self._effect.blurRadius())
        self._blur.setEndValue(18.0 if active and self._mode == "focus" else (14.0 if active else 0.0))
        self._blur.start()
        if self._mode == "hover":
            self._offset.stop()
            self._offset.setStartValue(self._effect.yOffset())
            self._offset.setEndValue(3.0 if active else 1.0)
            self._offset.start()

    def eventFilter(self, watched, event):
        if self._mode == "focus" and event.type() in (QEvent.FocusIn, QEvent.FocusOut):
            self._animate(event.type() == QEvent.FocusIn)
        elif self._mode == "hover" and event.type() in (QEvent.Enter, QEvent.Leave):
            self._animate(event.type() == QEvent.Enter)
        return False


def _add_interaction_effect(widget: QWidget, mode: str):
    # QObject parenting keeps the filter alive; the attribute helps debugging.
    widget._studio_interaction_effect = _InteractionEffect(widget, mode)


class _SegmentedProgressBar(QWidget):
    """A quiet, tick-based progress meter inspired by instrumentation UIs."""

    def __init__(self, parent=None, segments: int = 72):
        super().__init__(parent)
        self._value = 0
        self._minimum = 0
        self._maximum = 100
        self._segments = segments
        self._display_value = 0.0
        self._animation = QPropertyAnimation(self, b"animatedValue", self)
        self._animation.setDuration(280)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self.setFixedHeight(18)

    def setRange(self, minimum: int, maximum: int):
        self._minimum = minimum
        self._maximum = max(maximum, minimum + 1)
        self.update()

    def setValue(self, value: int):
        self._value = max(self._minimum, min(int(value), self._maximum))
        self._animation.stop()
        self._animation.setStartValue(self._display_value)
        self._animation.setEndValue(float(self._value))
        self._animation.start()

    def value(self) -> int:
        return self._value

    def _set_animated_value(self, value: float):
        self._display_value = value
        self.update()

    animatedValue = Property(float, lambda self: self._display_value, _set_animated_value)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        gap = 4.0
        segment_width = max(2.0, (width - gap * (self._segments - 1)) / self._segments)
        ratio = (self._display_value - self._minimum) / (self._maximum - self._minimum)
        active = round(ratio * self._segments)
        for index in range(self._segments):
            x = index * (segment_width + gap)
            if index < active:
                color = QColor("#b8b8be") if index < max(1, active - 2) else QColor("#eeeeef")
            else:
                color = QColor(62, 62, 66, 220)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(x, 4, segment_width, 10), 1.5, 1.5)


class PatientOnboardingDialog(QDialog):
    def __init__(self, backend, parent=None, selected_patient: str | None = None):
        super().__init__(parent)
        self.setObjectName("patientStudio")
        self.setWindowTitle("VisioALS — Patient Studio")
        self.setMinimumSize(980, 760)
        self.resize(1120, 780)
        self.setModal(True)
        self.setStyleSheet(_PATIENT_STUDIO_STYLESHEET)
        self._backend = backend
        self._name = ""
        self._thread = None
        self._is_generating = False
        self._recording_voice_sample = False
        self._has_external_source = False
        self._selected_saved_name = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        shell = QFrame()
        shell.setObjectName("studioShell")
        outer.addWidget(shell)

        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        rail = QFrame()
        rail.setObjectName("studioRail")
        rail.setFixedWidth(200)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(18, 22, 18, 18)
        rail_layout.setSpacing(8)

        rail_title = QLabel("Studio")
        rail_title.setObjectName("studioRailTitle")
        rail_layout.addWidget(rail_title)
        rail_layout.addSpacing(30)

        self._step_labels = []
        for index, (number, step) in enumerate((
            ("01", "Identity"),
            ("02", "Sources"),
            ("03", "Analysis"),
        )):
            stage = QFrame()
            stage.setObjectName("studioStageActive" if index == 0 else "studioStageMuted")
            stage.setFixedHeight(44)
            stage_layout = QHBoxLayout(stage)
            stage_layout.setContentsMargins(9, 7, 8, 7)
            stage_layout.setSpacing(9)
            stage_icon = QLabel(number)
            stage_icon.setObjectName("studioStageIcon")
            stage_icon.setAlignment(Qt.AlignCenter)
            stage_layout.addWidget(stage_icon)
            stage_text = QLabel(step)
            stage_text.setObjectName("studioStageText")
            stage_layout.addWidget(stage_text)
            stage_layout.addStretch()
            rail_layout.addWidget(stage)
            self._step_labels.append(stage)
        rail_layout.addStretch()

        rail_footer = QHBoxLayout()
        rail_footer.setSpacing(8)
        rail_tip = QLabel("Open studio")
        rail_tip.setObjectName("studioRailMeta")
        rail_footer.addWidget(rail_tip)
        rail_footer.addStretch()
        rail_shortcut = QLabel("P")
        rail_shortcut.setObjectName("studioRailShortcut")
        rail_shortcut.setAlignment(Qt.AlignCenter)
        rail_footer.addWidget(rail_shortcut)
        rail_layout.addLayout(rail_footer)
        shell_layout.addWidget(rail)

        main = QWidget()
        main.setObjectName("studioMain")
        layout = QVBoxLayout(main)
        layout.setContentsMargins(42, 30, 42, 24)
        layout.setSpacing(0)
        shell_layout.addWidget(main, 1)

        identity_fields = QHBoxLayout()
        identity_fields.setSpacing(12)

        name_column = QVBoxLayout()
        name_column.setSpacing(6)
        name_label = QLabel("Name")
        name_label.setObjectName("studioFieldLabel")
        name_column.addWidget(name_label)
        self._name_input = QLineEdit()
        self._name_input.setObjectName("studioInput")
        self._name_input.setPlaceholderText("e.g. Alex Morgan")
        self._name_input.setAccessibleName("Patient name")
        self._name_input.setFixedHeight(43)
        name_column.addWidget(self._name_input)
        self._profile_picker = QComboBox()
        self._profile_picker.setObjectName("studioProfilePicker")
        self._profile_picker.setAccessibleName("Saved patient profiles")
        self._profile_picker.setFixedHeight(36)
        self._profile_picker.addItem("Create a new profile…", None)
        from patient_data import PatientDataManager
        for patient_name in PatientDataManager.list_patient_names():
            self._profile_picker.addItem(patient_name, patient_name)
        name_column.addWidget(self._profile_picker)
        identity_fields.addLayout(name_column, 1)

        key_column = QVBoxLayout()
        key_column.setSpacing(6)
        key_label = QLabel("Voice cloning key  ·  Optional")
        key_label.setObjectName("studioFieldLabel")
        key_column.addWidget(key_label)
        self._elevenlabs_key_input = QLineEdit()
        self._elevenlabs_key_input.setObjectName("studioInput")
        self._elevenlabs_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._elevenlabs_key_input.setPlaceholderText("ElevenLabs API key")
        self._elevenlabs_key_input.setAccessibleName("ElevenLabs API key")
        self._elevenlabs_key_input.setFixedHeight(43)
        if self._backend.elevenlabs_api_key:
            self._elevenlabs_key_input.setText(self._backend.elevenlabs_api_key)
        key_column.addWidget(self._elevenlabs_key_input)
        identity_fields.addLayout(key_column, 1)
        layout.addLayout(identity_fields)
        layout.addSpacing(23)

        sources_surface = QFrame()
        sources_surface.setObjectName("studioSourceSurface")
        sources_surface.setMinimumHeight(174)
        sources_layout = QVBoxLayout(sources_surface)
        sources_layout.setContentsMargins(14, 12, 14, 13)
        sources_layout.setSpacing(7)

        source_toolbar = QHBoxLayout()
        source_toolbar.setSpacing(7)
        paste_label = QLabel("Writing sample")
        paste_label.setObjectName("studioSectionTitle")
        source_toolbar.addWidget(paste_label)
        source_toolbar.addStretch()

        self._import_btn = QPushButton("＋  Add files")
        self._import_btn.setObjectName("studioAction")
        self._import_btn.setToolTip("Import text, audio, or video sources")
        self._import_btn.clicked.connect(self._on_import)
        _add_interaction_effect(self._import_btn, "hover")
        source_toolbar.addWidget(self._import_btn)
        self._record_btn = QPushButton("●  Record voice")
        self._record_btn.setObjectName("studioAction")
        self._record_btn.setToolTip("Capture a microphone sample")
        self._record_btn.clicked.connect(self._toggle_voice_recording)
        _add_interaction_effect(self._record_btn, "hover")
        source_toolbar.addWidget(self._record_btn)
        sources_layout.addLayout(source_toolbar)

        self._import_status = QLabel("")
        self._import_status.setObjectName("studioStatus")
        self._import_status.setWordWrap(True)
        self._import_status.setVisible(False)
        sources_layout.addWidget(self._import_status)
        self._record_status = QLabel("")
        self._record_status.setObjectName("studioStatus")
        self._record_status.setWordWrap(True)
        self._record_status.setVisible(False)
        sources_layout.addWidget(self._record_status)

        self._paste_input = QTextEdit()
        self._paste_input.setObjectName("studioTextInput")
        self._paste_input.setPlaceholderText(
            "Paste messages, emails, notes, or any natural writing…"
        )
        self._paste_input.setMinimumHeight(94)
        self._paste_input.setAccessibleName("Patient writing samples")
        sources_layout.addWidget(self._paste_input)
        self._paste_status = QLabel("")
        self._paste_status.setObjectName("studioStatus")
        self._paste_status.setVisible(False)
        sources_layout.addWidget(self._paste_status)
        layout.addWidget(sources_surface)
        layout.addSpacing(20)

        progress_card = QFrame()
        progress_card.setObjectName("studioProgressCard")
        progress_card.setMinimumHeight(91)
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(15, 12, 15, 12)
        progress_layout.setSpacing(6)
        progress_header = QHBoxLayout()
        progress_label = QLabel("Ready to shape the profile")
        progress_label.setObjectName("studioFieldLabel")
        progress_header.addWidget(progress_label)
        progress_header.addStretch()
        self._progress_value = QLabel("Ready")
        self._progress_value.setObjectName("studioProgressValue")
        progress_header.addWidget(self._progress_value)
        progress_layout.addLayout(progress_header)

        self._bar = _SegmentedProgressBar()
        self._bar.setRange(0, 100)
        progress_layout.addWidget(self._bar)

        self._status = QLabel("Name the patient and add at least one source to begin.")
        self._status.setObjectName("studioStatus")
        self._status.setWordWrap(True)
        progress_layout.addWidget(self._status)

        self._summary = QTextEdit()
        self._summary.setObjectName("studioSummary")
        self._summary.setReadOnly(True)
        self._summary.setFixedHeight(54)
        self._summary.setVisible(False)
        progress_layout.addWidget(self._summary)
        layout.addWidget(progress_card)
        layout.addSpacing(12)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("studioSecondary")
        self._cancel_btn.clicked.connect(self.reject)
        _add_interaction_effect(self._cancel_btn, "hover")
        row.addWidget(self._cancel_btn)
        self._done_btn = QPushButton("Use this profile  →")
        self._done_btn.setObjectName("studioPrimary")
        self._done_btn.setVisible(False)
        self._done_btn.clicked.connect(self.accept)
        _add_interaction_effect(self._done_btn, "hover")
        row.addWidget(self._done_btn)
        self._generate_btn = QPushButton("Analyze sources  →")
        self._generate_btn.setObjectName("studioPrimary")
        self._generate_btn.clicked.connect(self._on_generate)
        self._generate_btn.setEnabled(False)
        self._generate_btn.setDefault(True)
        _add_interaction_effect(self._generate_btn, "hover")
        row.addWidget(self._generate_btn)
        layout.addLayout(row)

        self._name_input.textChanged.connect(self._on_name_changed)
        self._profile_picker.currentIndexChanged.connect(self._on_profile_selected)
        self._paste_input.textChanged.connect(self._on_pasted_text_changed)
        self.setTabOrder(self._name_input, self._profile_picker)
        self.setTabOrder(self._profile_picker, self._elevenlabs_key_input)
        self.setTabOrder(self._elevenlabs_key_input, self._import_btn)
        self.setTabOrder(self._import_btn, self._record_btn)
        self.setTabOrder(self._record_btn, self._paste_input)
        self.setTabOrder(self._paste_input, self._generate_btn)
        self.setFocusPolicy(Qt.StrongFocus)
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)
        if selected_patient:
            selected_index = self._profile_picker.findData(selected_patient)
            if selected_index >= 0:
                self._profile_picker.setCurrentIndex(selected_index)
            else:
                self._name_input.setFocus()
        else:
            self._name_input.setFocus()

    def patient_name(self) -> str:
        return self._name

    def _is_text_entry_target(self, widget: QWidget | None) -> bool:
        if widget is None:
            return False
        for editor in (self._name_input, self._elevenlabs_key_input, self._paste_input):
            if widget is editor or editor.isAncestorOf(widget):
                return True
        return False

    def eventFilter(self, watched, event):
        if event.type() == QEvent.MouseButtonPress and not self._is_text_entry_target(watched):
            if self._is_text_entry_target(QApplication.focusWidget()):
                self.setFocus(Qt.MouseFocusReason)
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_P and not self._is_text_entry_target(QApplication.focusWidget()):
            event.accept()
            self.reject()
            return
        super().keyPressEvent(event)

    def _set_active_step(self, active_index: int):
        for index, stage in enumerate(self._step_labels):
            stage.setObjectName(
                "studioStageActive" if index == active_index else "studioStageMuted"
            )
            for widget in (stage, *stage.findChildren(QLabel)):
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()

    def _on_profile_selected(self, index: int):
        name = self._profile_picker.itemData(index)
        if not name:
            self._start_new_profile(clear_name=True)
            return

        from patient_data import PatientDataManager
        pm = PatientDataManager(name)
        profile = pm.load_linguistic_profile() or {}
        voice_profile = pm.load_voice_profile() or {}
        corpus_count = len(pm.load_corpus())
        media_count = len(pm.list_media_files())

        self._selected_saved_name = name
        self._name = name
        self._name_input.setText(name)
        self._has_external_source = bool(corpus_count or media_count)
        paste_blocker = QSignalBlocker(self._paste_input)
        self._paste_input.clear()
        del paste_blocker

        source_parts = [f"{corpus_count} writing sample(s)"]
        if media_count:
            source_parts.append(f"{media_count} voice recording(s)")
        self._import_status.setText("Saved sources: " + ", ".join(source_parts) + ".")
        self._import_status.setObjectName("statusOk")
        self._import_status.setStyle(self._import_status.style())
        self._import_status.setVisible(True)
        self._record_status.setVisible(False)
        self._paste_status.setVisible(False)

        summary = profile.get("summary") or "Saved voice profile"
        self._summary.setPlainText(summary)
        self._summary.setVisible(True)
        self._bar.setValue(100)
        self._progress_value.setText("Saved")
        voice_note = " with cloned voice" if voice_profile.get("voice_id") else ""
        self._status.setText(f"{name}'s profile{voice_note} is ready to use.")
        self._status.setObjectName("statusOk")
        self._status.setStyle(self._status.style())
        self._set_active_step(2)

        self._generate_btn.setText("Update profile  →")
        self._generate_btn.setEnabled(True)
        self._generate_btn.setVisible(True)
        self._generate_btn.setDefault(False)
        self._done_btn.setText(f"Use {name}  →")
        self._done_btn.setVisible(True)
        self._done_btn.setDefault(True)
        self._cancel_btn.setVisible(True)

    def _on_name_changed(self):
        name = self._name_input.text().strip()
        if self._selected_saved_name and name != self._selected_saved_name:
            self._start_new_profile(clear_name=False)
        self._check_ready()

    def _start_new_profile(self, clear_name: bool):
        self._selected_saved_name = None
        self._name = ""
        picker_blocker = QSignalBlocker(self._profile_picker)
        self._profile_picker.setCurrentIndex(0)
        del picker_blocker
        if clear_name:
            self._name_input.clear()
        self._has_external_source = False
        self._import_status.setVisible(False)
        self._record_status.setVisible(False)
        self._summary.clear()
        self._summary.setVisible(False)
        self._bar.setValue(0)
        self._progress_value.setText("Ready")
        self._status.setObjectName("studioStatus")
        self._status.setStyle(self._status.style())
        self._set_active_step(0)
        self._generate_btn.setText("Analyze sources  →")
        self._generate_btn.setVisible(True)
        self._generate_btn.setDefault(True)
        self._done_btn.setVisible(False)
        self._done_btn.setDefault(False)
        self._cancel_btn.setVisible(True)
        self._check_ready()

    def _check_ready(self):
        name = self._name_input.text().strip()
        has_name = len(name) > 0
        has_pasted_text = bool(self._paste_input.toPlainText().strip())
        has_source = has_pasted_text or self._has_external_source
        self._generate_btn.setEnabled(has_name and not self._is_generating)
        if not has_name and not self._is_generating:
            self._status.setText("Name the patient and add at least one source to begin.")
        elif has_source and self._bar.value() == 0 and not self._is_generating:
            self._status.setText("Source ready — start analysis when everything looks right.")
        elif has_name and self._bar.value() == 0 and not self._is_generating:
            self._status.setText("Add a source, or analyze an existing corpus for this patient.")

    def _on_pasted_text_changed(self):
        text = self._paste_input.toPlainText().strip()
        if text:
            self._set_active_step(1)
            self._paste_status.setVisible(True)
            word_count = len(text.split())
            if word_count < 200:
                self._paste_status.setText(
                    f"{word_count} words — 200+ recommended for a reliable profile"
                )
                self._paste_status.setObjectName("statusWarn")
            else:
                self._paste_status.setText(f"{word_count} words ready to use")
                self._paste_status.setObjectName("statusOk")
        else:
            self._paste_status.clear()
            self._paste_status.setObjectName("studioStatus")
            self._paste_status.setVisible(False)
        self._paste_status.setStyle(self._paste_status.style())
        self._check_ready()

    def _on_import(self):
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Enter a patient name first.")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select patient files",
            "",
            (
                "Supported files (*.txt *.json *.csv *.mp3 *.wav *.m4a *.aac "
                "*.flac *.ogg *.opus *.mp4 *.mov *.avi *.mkv *.mpeg *.mpg *.webm);;"
                "Text files (*.txt *.json *.csv);;"
                "Audio files (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus);;"
                "Video files (*.mp4 *.mov *.avi *.mkv *.mpeg *.mpg *.webm)"
            ),
        )
        if not paths:
            return

        from patient_data import PatientDataManager

        pm = PatientDataManager(name)
        text_exts = {".txt", ".json", ".csv"}
        text_paths = [p for p in paths if os.path.splitext(p)[1].lower() in text_exts]
        media_paths = [p for p in paths if p not in text_paths]
        copied_text = pm.add_corpus_files(text_paths)
        copied_media = pm.add_media_files(media_paths)

        corpus = pm.load_corpus()
        self._import_status.setText(
            f"Added {len(copied_text)} text file(s) and {len(copied_media)} media file(s). "
            f"The current corpus contains {len(corpus)} text snippet(s); media will be "
            "transcribed when the profile is generated."
        )
        self._import_status.setObjectName("statusOk")
        self._import_status.setStyle(self._import_status.style())
        self._import_status.setVisible(True)
        self._has_external_source = bool(copied_text or copied_media)
        self._set_active_step(1)
        self._check_ready()

    def _toggle_voice_recording(self):
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Enter a patient name first.")
            return

        if not self._recording_voice_sample:
            try:
                self._backend.start_recording()
                self._recording_voice_sample = True
                self._set_active_step(1)
                self._record_btn.setText("■   Stop & save")
                self._record_status.setText("Recording… speak naturally, then press Stop.")
                self._record_status.setObjectName("statusWarn")
                self._record_status.setVisible(True)
            except Exception as e:
                QMessageBox.warning(self, "Microphone error", str(e))
                return
        else:
            audio = self._backend.stop_recording()
            self._recording_voice_sample = False
            self._record_btn.setText("●   Record another")
            if audio is None or len(audio) == 0:
                self._record_status.setText("No audio was captured. Try again.")
                self._record_status.setObjectName("statusErr")
            else:
                from patient_data import PatientDataManager
                path = PatientDataManager(name).save_recording(audio)
                self._has_external_source = True
                self._record_status.setText(
                    f"Saved {os.path.basename(path)}. It will be transcribed and cloned."
                )
                self._record_status.setObjectName("statusOk")
        self._record_status.setStyle(self._record_status.style())
        self._record_status.setVisible(True)
        self._check_ready()

    def _on_generate(self):
        name = self._name_input.text().strip()
        if not name:
            return
        pasted_text = self._paste_input.toPlainText().strip()
        if pasted_text:
            from patient_data import PatientDataManager
            PatientDataManager(name).add_pasted_text(pasted_text)

        from patient_data import PatientDataManager
        pm = PatientDataManager(name)
        api_key = self._elevenlabs_key_input.text().strip()
        if pm.list_media_files() and not api_key:
            QMessageBox.warning(
                self,
                "ElevenLabs API key required",
                "Enter an ElevenLabs API key to clone the imported voice recordings.",
            )
            return
        if api_key:
            from credential_store import save_elevenlabs_api_key
            if not save_elevenlabs_api_key(api_key):
                self._status.setText(
                    "The API key will work for this session, but the operating-system "
                    "credential store was unavailable."
                )
                self._status.setObjectName("statusWarn")
                self._status.setStyle(self._status.style())
        self._name = name
        self._is_generating = True
        self._set_active_step(2)
        self._bar.setValue(0)
        self._progress_value.setText("0%")
        self._status.setText("Preparing sources and building the patient profile…")
        self._status.setObjectName("studioStatus")
        self._status.setStyle(self._status.style())
        self._generate_btn.setText("Analyzing…")
        self._generate_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._name_input.setEnabled(False)
        self._profile_picker.setEnabled(False)
        self._import_btn.setEnabled(False)
        self._record_btn.setEnabled(False)
        self._elevenlabs_key_input.setEnabled(False)
        self._paste_input.setEnabled(False)

        self._thread = QThread()
        self._worker = _OnboardingWorker(
            name,
            self._backend.api_url,
            _model_dir(),
            self._backend,
            api_key,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_progress(self, value: int):
        self._bar.setValue(value)
        self._progress_value.setText(f"{value}%")

    def _on_done(self, result):
        self._is_generating = False
        profile = result.get("profile") if isinstance(result, dict) else None
        if profile:
            self._on_progress(100)
            self._progress_value.setText("Complete")
            voice_profile = result.get("voice_profile") or {}
            if voice_profile.get("requires_verification"):
                self._status.setText(
                    "Profile generated. ElevenLabs requires verification before the "
                    "cloned voice can be used."
                )
                self._status.setObjectName("statusWarn")
            elif voice_profile.get("voice_id"):
                self._status.setText(
                    "Profile generated and ElevenLabs cloned voice activated."
                )
                self._status.setObjectName("statusOk")
            else:
                self._status.setText(
                    "Profile generated. Add a recording later to enable cloned speech."
                )
                self._status.setObjectName("statusOk")
            self._summary.setVisible(True)
            self._summary.setPlainText(profile.get("summary", "(no summary)"))
            self._generate_btn.setVisible(False)
            self._done_btn.setText(f"Use {self._name}  →")
            self._done_btn.setVisible(True)
            self._done_btn.setFocus()
            self._cancel_btn.setVisible(False)
        else:
            error = result.get("error") if isinstance(result, dict) else None
            self._status.setText(f"Failed — {error or 'check the imported files.'}")
            self._status.setObjectName("statusErr")
            self._set_active_step(1)
            self._progress_value.setText("Needs attention")
            self._generate_btn.setText("Try analysis again   →")
            self._generate_btn.setEnabled(True)
            self._cancel_btn.setEnabled(True)
            self._name_input.setEnabled(True)
            self._profile_picker.setEnabled(True)
            self._import_btn.setEnabled(True)
            self._record_btn.setEnabled(True)
            self._elevenlabs_key_input.setEnabled(True)
            self._paste_input.setEnabled(True)
        self._status.setStyle(self._status.style())

    def reject(self):
        if self._is_generating:
            self._status.setText("Analysis is still running. This window will be ready shortly.")
            self._status.setObjectName("statusWarn")
            self._status.setStyle(self._status.style())
            return
        if self._recording_voice_sample:
            self._backend.stop_recording()
            self._recording_voice_sample = False
        super().reject()


class PreferenceViewerDialog(QDialog):
    def __init__(self, patient_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Learned Preferences")
        self.setFixedSize(460, 380)
        self.setModal(True)
        self.setStyleSheet(_STYLESHEET + "QDialog { background-color: #FAFAF7; }")
        self._patient_data = patient_data

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 20)
        layout.setSpacing(0)

        title = QLabel("Preference Rules")
        title.setObjectName("pageTitle")
        title.setStyleSheet("font-size: 20px;")
        layout.addWidget(title)
        layout.addSpacing(4)

        sub = QLabel(f"Patient: {patient_data.patient_name}")
        sub.setObjectName("pageSubtitle")
        layout.addWidget(sub)
        layout.addSpacing(12)

        # load preferences
        pref = patient_data.load_preference_profile()
        rules = pref.get("rules", []) if pref else []
        last_updated = pref.get("last_updated") if pref else None
        interaction_count = pref.get("interaction_count", 0) if pref else 0

        if last_updated:
            import time as _t
            ts = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(last_updated))
            meta = QLabel(f"Last updated: {ts}  \u00b7  Based on {interaction_count} interactions")
        else:
            meta = QLabel("No preferences learned yet.")
        meta.setObjectName("pageSubtitle")
        meta.setStyleSheet("font-size: 12px;")
        layout.addWidget(meta)
        layout.addSpacing(12)

        # rules list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #D5D5D0; border-radius: 8px; background: #FFF; }")
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(16, 12, 16, 12)
        inner_layout.setSpacing(6)

        if rules:
            for rule in rules:
                lbl = QLabel(f"\u2022  {rule}")
                lbl.setWordWrap(True)
                lbl.setStyleSheet("font-size: 13px; color: #2A2A2A;")
                inner_layout.addWidget(lbl)
        else:
            lbl = QLabel("No preference rules yet. They will be generated after 20+ interactions.")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 13px; color: #888;")
            inner_layout.addWidget(lbl)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        layout.addSpacing(12)

        # buttons
        row = QHBoxLayout()
        if rules:
            clear_btn = QPushButton("Clear Preferences")
            clear_btn.setObjectName("secondaryBtn")
            clear_btn.clicked.connect(self._on_clear)
            row.addWidget(clear_btn)
        row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        layout.addLayout(row)

    def _on_clear(self):
        reply = QMessageBox.question(
            self, "Clear Preferences",
            "This will delete all learned preference rules. The system will re-learn from future interactions.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._patient_data.delete_preference_profile()
            self.accept()


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


def _smoke_test_dependencies() -> None:
    """Import packaged runtime components without opening hardware or windows."""
    import ctranslate2  # noqa: F401
    import en_core_web_sm
    import faster_whisper  # noqa: F401
    import keyring  # noqa: F401
    import mediapipe  # noqa: F401
    import onnxruntime  # noqa: F401
    import sounddevice  # noqa: F401
    import spacy  # noqa: F401
    import tokenizers  # noqa: F401

    # Loading the small English pipeline verifies its bundled model data, not
    # merely the generated Python package wrapper.
    en_core_web_sm.load()
    print("VisioALS bundle smoke test passed")


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("VisioALS")
    app.setOrganizationDomain("visioals.org")
    app.setApplicationName("VisioALS")
    app.setApplicationDisplayName("VisioALS")
    # fusion looks consistent across windows versions, native style is a mess
    app.setStyle("Fusion")

    if "--smoke-test" in sys.argv:
        _smoke_test_dependencies()
        return

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

    from credential_store import load_elevenlabs_api_key
    elevenlabs_api_key = load_elevenlabs_api_key()
    backend = BackendClient(
        api_url=cfg["api_url"],
        model_dir=mdir,
        elevenlabs_api_key=elevenlabs_api_key,
    )

    # load active patient context if one is configured
    patient_data = None
    corpus_index = None
    active_patient = cfg.get("active_patient")
    if active_patient:
        from patient_data import PatientDataManager
        from embeddings import EmbeddingProvider, CorpusIndex
        patient_data = PatientDataManager(active_patient)
        voice_profile = patient_data.load_voice_profile() or {}
        backend.configure_elevenlabs(
            elevenlabs_api_key,
            voice_profile.get("voice_id", ""),
        )
        if patient_data.load_linguistic_profile() is not None:
            provider = EmbeddingProvider(cache_dir=mdir)
            corpus_index = CorpusIndex(patient_data.embeddings_dir, provider)
            if corpus_index.is_built():
                corpus_index.load_index()
            else:
                corpus_index = None
        print(f"active patient: {active_patient}")

    win = MainWindow(gaze, backend, cfg, save_config, patient_data, corpus_index)
    win.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
