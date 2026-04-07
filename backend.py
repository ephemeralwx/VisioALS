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


# pyttsx3 is not threadsafe, needs its own qthread
class TTSWorker(QObject):
    request_speak = Signal(str)

    def __init__(self):
        super().__init__()
        self._engine = None

    @Slot(str)
    def _speak(self, text: str):
        try:
            # recreate engine every call - reusing it causes silent failures
            # after the first runAndWait(), seems like a pyttsx3 bug
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            print(f"tts spoke: {text[:60]}")
        except Exception as e:
            print(f"tts error: {e}")


class BackendClient:

    def __init__(self, api_url: str, model_dir: str):
        self.api_url = api_url.rstrip("/")
        self.model_dir = model_dir

        self._stream = None
        self._audio_frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()

        self._whisper_model = None

        self._tts_thread = QThread()
        self._tts_worker = TTSWorker()
        self._tts_worker.moveToThread(self._tts_thread)
        # must connect after moveToThread or qt delivers to wrong thread
        self._tts_worker.request_speak.connect(
            self._tts_worker._speak, Qt.ConnectionType.QueuedConnection
        )
        self._tts_thread.start()

    def generate_options(
        self,
        question: str,
        history: list[dict] | None = None,
        rejected: list[str] | None = None,
        linguistic_profile_summary: str | None = None,
        exemplars: list[str] | None = None,
        preference_rules: list[str] | None = None,
    ) -> list[str]:
        try:
            body: dict = {"question": question}
            if history:
                body["history"] = history
            if rejected:
                body["rejected"] = rejected
            if linguistic_profile_summary:
                body["linguistic_profile_summary"] = linguistic_profile_summary
            if exemplars:
                body["exemplars"] = exemplars
            if preference_rules:
                body["preference_rules"] = preference_rules
            r = requests.post(
                f"{self.api_url}/generate-options",
                json=body,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            options = data.get("options", [])
            # pad to 4 so the UI grid always has something to show
            while len(options) < 4:
                options.append("(no response)")
            return options[:4]
        except Exception as e:
            print(f"api generate_options failed: {e}")
            return ["(error)", "(error)", "(error)", "(error)"]

    def expand_response(
        self,
        question: str,
        response: str,
        history: list[dict] | None = None,
        linguistic_profile_summary: str | None = None,
        exemplars: list[str] | None = None,
    ) -> str:
        try:
            body: dict = {"question": question, "response": response}
            if history:
                body["history"] = history
            if linguistic_profile_summary:
                body["linguistic_profile_summary"] = linguistic_profile_summary
            if exemplars:
                body["exemplars"] = exemplars
            r = requests.post(
                f"{self.api_url}/expand-response",
                json=body,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("expanded", response)
        except Exception as e:
            print(f"api expand_response failed: {e}")
            return response

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.api_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def start_recording(self):
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
            print("recording started")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"audio callback status: {status}")
        self._audio_frames.append(indata.copy())

    def stop_recording(self) -> np.ndarray | None:
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            print("recording stopped")
            if not self._audio_frames:
                return None
            return np.concatenate(self._audio_frames, axis=0)

    def _load_whisper(self):
        if self._whisper_model is not None:
            return
        from faster_whisper import WhisperModel
        # tiny.en is good enough for short phrases and doesn't kill cpu
        self._whisper_model = WhisperModel(
            "tiny.en",
            device="cpu",
            compute_type="int8",
            download_root=self.model_dir,
        )
        print("whisper tiny.en loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        self._load_whisper()
        # faster-whisper needs a file path, can't take raw numpy audio
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()  # close handle so whisper can read the file on windows
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio.tobytes())
            segments, _ = self._whisper_model.transcribe(tmp_path)
            text = " ".join(seg.text for seg in segments).strip()
            print(f"transcribed: {text}")
            return text
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def analyze_style(self, sample_texts: list[str]) -> dict:
        """Call /analyze-style for subjective linguistic analysis."""
        try:
            r = requests.post(
                f"{self.api_url}/analyze-style",
                json={"sample_texts": sample_texts},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"api analyze_style failed: {e}")
            return {
                "humor_style": "unknown",
                "tone_description": "unknown",
                "emotional_valence": "neutral",
                "personality_notes": "",
            }

    def analyze_preferences(self, interactions: list[dict]) -> list[str]:
        """Call /analyze-preferences to extract preference rules from interaction history."""
        try:
            r = requests.post(
                f"{self.api_url}/analyze-preferences",
                json={"interactions": interactions},
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("rules", [])
        except Exception as e:
            print(f"api analyze_preferences failed: {e}")
            return []

    def speak(self, text: str):
        self._tts_worker.request_speak.emit(text)

    def shutdown(self):
        if self._recording:
            self.stop_recording()
        self._tts_thread.quit()
        # 2s is arbitrary, just don't block forever on exit
        self._tts_thread.wait(2000)
