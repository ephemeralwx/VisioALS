"""
backend.py — API client, audio recording, speech-to-text, text-to-speech.
"""

import os
import json
import wave
import tempfile
import threading
import numpy as np
import sounddevice as sd
import pyttsx3
import requests
from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt


# ---------------------------------------------------------------------------
# TTS Worker — lives on a dedicated QThread so pyttsx3 is always called from
# the same thread (it is not thread-safe).
# ---------------------------------------------------------------------------

class TTSWorker(QObject):
    """Owns the pyttsx3 engine; receives speak requests via signal."""
    request_speak = Signal(str)

    def __init__(self):
        super().__init__()
        self._engine = None
        # NOTE: signal is connected in BackendClient after moveToThread()

    @Slot(str)
    def _speak(self, text: str):
        try:
            # Recreate engine every call — pyttsx3's runAndWait() often
            # fails silently on the second invocation if the engine is reused.
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            print(f"[TTS] Spoke: {text[:60]}")
        except Exception as e:
            print(f"[TTS] Error: {e}")


# ---------------------------------------------------------------------------
# Backend Client
# ---------------------------------------------------------------------------

class BackendClient:
    """Handles Railway API calls, audio recording, transcription, and TTS."""

    def __init__(self, api_url: str, model_dir: str):
        self.api_url = api_url.rstrip("/")
        self.model_dir = model_dir

        # Audio recording state
        self._stream = None
        self._audio_frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()

        # STT model (lazy-loaded)
        self._whisper_model = None

        # TTS on dedicated thread — connect AFTER moveToThread with queued connection
        self._tts_thread = QThread()
        self._tts_worker = TTSWorker()
        self._tts_worker.moveToThread(self._tts_thread)
        self._tts_worker.request_speak.connect(
            self._tts_worker._speak, Qt.ConnectionType.QueuedConnection
        )
        self._tts_thread.start()

    # -- Railway API ---------------------------------------------------------

    def generate_options(self, question: str) -> list[str]:
        """POST /generate-options → list of 4 response strings."""
        try:
            r = requests.post(
                f"{self.api_url}/generate-options",
                json={"question": question},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            options = data.get("options", [])
            # Guarantee exactly 4
            while len(options) < 4:
                options.append("(no response)")
            return options[:4]
        except Exception as e:
            print(f"[API] generate_options error: {e}")
            return ["(error)", "(error)", "(error)", "(error)"]

    def expand_response(self, question: str, response: str) -> str:
        """POST /expand-response → expanded sentence."""
        try:
            r = requests.post(
                f"{self.api_url}/expand-response",
                json={"question": question, "response": response},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("expanded", response)
        except Exception as e:
            print(f"[API] expand_response error: {e}")
            return response

    def health_check(self) -> bool:
        """GET /health — returns True if backend is reachable."""
        try:
            r = requests.get(f"{self.api_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # -- Audio recording (sounddevice) --------------------------------------

    def start_recording(self):
        """Open a 16 kHz mono input stream."""
        with self._lock:
            if self._recording:
                return
            self._audio_frames = []
            self._recording = True
            self._stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            )
            self._stream.start()
            print("[Audio] Recording started.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Audio] {status}")
        self._audio_frames.append(indata.copy())

    def stop_recording(self) -> np.ndarray | None:
        """Stop recording and return the audio as int16 numpy array."""
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            print("[Audio] Recording stopped.")
            if not self._audio_frames:
                return None
            return np.concatenate(self._audio_frames, axis=0)

    # -- Transcription (faster-whisper) -------------------------------------

    def _load_whisper(self):
        if self._whisper_model is not None:
            return
        from faster_whisper import WhisperModel
        self._whisper_model = WhisperModel(
            "tiny.en",
            device="cpu",
            compute_type="int8",
            download_root=self.model_dir,
        )
        print("[STT] faster-whisper tiny.en loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        """Save audio to temp WAV, run faster-whisper, return text."""
        self._load_whisper()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        try:
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(16000)
                wf.writeframes(audio.tobytes())
            segments, _ = self._whisper_model.transcribe(tmp_path)
            text = " ".join(seg.text for seg in segments).strip()
            print(f"[STT] Transcribed: {text}")
            return text
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # -- TTS ----------------------------------------------------------------

    def speak(self, text: str):
        """Queue text for the TTS worker thread."""
        self._tts_worker.request_speak.emit(text)

    # -- Cleanup ------------------------------------------------------------

    def shutdown(self):
        if self._recording:
            self.stop_recording()
        self._tts_thread.quit()
        self._tts_thread.wait(2000)
