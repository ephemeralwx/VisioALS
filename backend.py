import os
import json
import contextlib
import mimetypes
import platform
import queue
import shutil
import subprocess
import wave
import tempfile
import threading
import numpy as np
import sounddevice as sd
import requests

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


# pyttsx3 is not threadsafe and can silently fail when driven from Qt threads.
# Keep speech on one plain Python worker thread and fall back to OS TTS.
class TTSWorker:
    def __init__(self, elevenlabs_api_key: str = "", voice_id: str = ""):
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._config_lock = threading.Lock()
        self._elevenlabs_api_key = (elevenlabs_api_key or "").strip()
        self._elevenlabs_voice_id = (voice_id or "").strip()
        self._thread = threading.Thread(target=self._run, name="VisioALS-TTS", daemon=True)
        self._thread.start()

    def configure_elevenlabs(self, api_key: str, voice_id: str):
        with self._config_lock:
            self._elevenlabs_api_key = (api_key or "").strip()
            self._elevenlabs_voice_id = (voice_id or "").strip()

    def speak(self, text: str):
        text = (text or "").strip()
        if text:
            self._queue.put(text)

    def shutdown(self, timeout: float = 2.0):
        self._queue.put(None)
        self._thread.join(timeout)

    def _run(self):
        while True:
            text = self._queue.get()
            if text is None:
                return

            if self._speak_with_elevenlabs(text):
                print(f"tts spoke via ElevenLabs clone: {text[:60]}")
                continue

            # NSSpeechSynthesizer (used by pyttsx3 on macOS) can return from
            # runAndWait() without producing audio when called off the main
            # thread.  The built-in `say` command is reliable from this worker.
            if platform.system() == "Darwin":
                if self._speak_with_system(text):
                    print(f"tts spoke via system voice: {text[:60]}")
                    continue
                print("tts system voice unavailable, trying pyttsx3")

            if self._speak_with_pyttsx3(text):
                print(f"tts spoke: {text[:60]}")
                continue

            # macOS already tried the system voice above.
            if platform.system() != "Darwin" and self._speak_with_system(text):
                print(f"tts spoke via system voice: {text[:60]}")
            else:
                print("tts error: no working text-to-speech backend")

    def _speak_with_elevenlabs(self, text: str) -> bool:
        lock = getattr(self, "_config_lock", None)
        if lock is None:
            api_key = getattr(self, "_elevenlabs_api_key", "")
            voice_id = getattr(self, "_elevenlabs_voice_id", "")
        else:
            with lock:
                api_key = self._elevenlabs_api_key
                voice_id = self._elevenlabs_voice_id

        if not api_key or not voice_id:
            return False

        try:
            # macOS' native player handles ElevenLabs MP3 output reliably and
            # avoids feeding a low-rate raw PCM stream through PortAudio. The
            # latter can sound like static on some CoreAudio device setups.
            use_native_macos_player = (
                platform.system() == "Darwin" and shutil.which("afplay") is not None
            )
            output_format = (
                "mp3_44100_128" if use_native_macos_player else "pcm_24000"
            )
            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                params={"output_format": output_format},
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/octet-stream",
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.8,
                        "use_speaker_boost": True,
                    },
                },
                timeout=60,
            )
            response.raise_for_status()
            if not response.content:
                raise RuntimeError("ElevenLabs returned empty audio")

            if use_native_macos_player:
                self._play_mp3_with_afplay(response.content)
            else:
                self._play_pcm(response.content, sample_rate=24000)
            return True
        except Exception as e:
            print(f"ElevenLabs TTS failed, using local fallback: {e}")
            return False

    @staticmethod
    def _play_mp3_with_afplay(content: bytes) -> None:
        """Play encoded ElevenLabs audio with macOS' native audio stack."""
        if content.startswith((b"RIFF", b"OggS", b"fLaC")):
            raise RuntimeError("ElevenLabs returned an unexpected audio format")

        temp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        path = temp.name
        try:
            temp.write(content)
            temp.close()
            subprocess.run(
                [shutil.which("afplay") or "/usr/bin/afplay", path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            if not temp.closed:
                temp.close()
            try:
                os.unlink(path)
            except OSError:
                pass

    @staticmethod
    def _play_pcm(content: bytes, sample_rate: int) -> None:
        """Validate signed 16-bit little-endian PCM before playback."""
        encoded_signatures = (b"ID3", b"RIFF", b"OggS", b"fLaC")
        if content.startswith(encoded_signatures):
            raise RuntimeError(
                "ElevenLabs returned encoded audio when raw PCM was requested"
            )
        if len(content) % 2:
            raise RuntimeError("ElevenLabs returned malformed 16-bit PCM audio")

        audio = np.frombuffer(content, dtype="<i2")
        if audio.size == 0:
            raise RuntimeError("ElevenLabs returned empty PCM audio")

        # Float32 is PortAudio's native interchange format and avoids device-
        # specific integer conversion issues that can produce crunchy output.
        normalized = audio.astype(np.float32) / 32768.0
        sd.play(normalized, samplerate=sample_rate, blocking=True)

    def _speak_with_pyttsx3(self, text: str) -> bool:
        if pyttsx3 is None:
            print("tts pyttsx3 unavailable")
            return False

        pythoncom = None
        try:
            if platform.system() == "Windows":
                try:
                    import pythoncom as _pythoncom
                    pythoncom = _pythoncom
                    pythoncom.CoInitialize()
                except Exception as e:
                    print(f"tts COM init warning: {e}")

            # recreate engine every call - reusing it causes silent failures
            # after the first runAndWait(), seems like a pyttsx3 bug
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return True
        except Exception as e:
            print(f"tts pyttsx3 error: {e}")
            return False
        finally:
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def _speak_with_system(self, text: str) -> bool:
        system = platform.system()
        command = None

        if system == "Windows":
            powershell = shutil.which("powershell") or shutil.which("pwsh")
            if powershell:
                command = [
                    powershell,
                    "-NoProfile",
                    "-Command",
                    (
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        "$s.Speak($args[0])"
                    ),
                    text,
                ]
        elif system == "Darwin" and shutil.which("say"):
            command = ["say", text]
        elif shutil.which("spd-say"):
            command = ["spd-say", text]
        elif shutil.which("espeak"):
            command = ["espeak", text]

        if not command:
            return False

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as e:
            print(f"tts system voice error: {e}")
            return False


class BackendClient:

    def __init__(self, api_url: str, model_dir: str,
                 elevenlabs_api_key: str = "", voice_id: str = ""):
        self.api_url = api_url.rstrip("/")
        self.model_dir = model_dir

        self._stream = None
        self._audio_frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()

        self._whisper_model = None

        self._elevenlabs_api_key = (elevenlabs_api_key or "").strip()
        self._elevenlabs_voice_id = (voice_id or "").strip()
        self._tts_worker = TTSWorker(self._elevenlabs_api_key, self._elevenlabs_voice_id)

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

    def transcribe_file(self, media_path: str) -> str:
        """Transcribe an imported audio or video file with local Whisper."""
        self._load_whisper()
        segments, _ = self._whisper_model.transcribe(media_path, vad_filter=True)
        text = " ".join(segment.text for segment in segments).strip()
        print(f"transcribed media {os.path.basename(media_path)}: {text[:100]}")
        return text

    @property
    def elevenlabs_api_key(self) -> str:
        return self._elevenlabs_api_key

    @property
    def elevenlabs_voice_id(self) -> str:
        return self._elevenlabs_voice_id

    def configure_elevenlabs(self, api_key: str, voice_id: str):
        self._elevenlabs_api_key = (api_key or "").strip()
        self._elevenlabs_voice_id = (voice_id or "").strip()
        self._tts_worker.configure_elevenlabs(
            self._elevenlabs_api_key,
            self._elevenlabs_voice_id,
        )

    @staticmethod
    def _write_pcm_wav(path: str, audio: np.ndarray, sample_rate: int = 16000):
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

    def clone_voice(self, patient_name: str, media_paths: list[str],
                    api_key: str | None = None) -> dict:
        """Create an ElevenLabs Instant Voice Clone from imported media."""
        key = (api_key or self._elevenlabs_api_key or "").strip()
        if not key:
            raise ValueError("An ElevenLabs API key is required to clone a voice.")
        if not media_paths:
            raise ValueError("At least one voice recording is required.")

        # ElevenLabs expects audio files. Convert video containers to a
        # temporary mono WAV using faster-whisper's bundled PyAV decoder.
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".webm"}
        with tempfile.TemporaryDirectory(prefix="visioals_clone_") as temp_dir:
            audio_paths: list[str] = []
            for index, source in enumerate(media_paths):
                ext = os.path.splitext(source)[1].lower()
                if ext in video_extensions:
                    from faster_whisper.audio import decode_audio
                    audio = decode_audio(source, sampling_rate=16000)
                    converted = os.path.join(temp_dir, f"video_{index}.wav")
                    self._write_pcm_wav(converted, audio, 16000)
                    audio_paths.append(converted)
                else:
                    audio_paths.append(source)

            with contextlib.ExitStack() as stack:
                files = []
                for path in audio_paths:
                    handle = stack.enter_context(open(path, "rb"))
                    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
                    files.append(("files", (os.path.basename(path), handle, mime)))

                response = requests.post(
                    "https://api.elevenlabs.io/v1/voices/add",
                    headers={"xi-api-key": key},
                    data={
                        "name": f"VisioALS - {patient_name}",
                        "description": "Voice clone created by VisioALS from user-provided recordings.",
                        "remove_background_noise": "false",
                    },
                    files=files,
                    timeout=300,
                )
                if not response.ok:
                    try:
                        detail = response.json().get("detail", response.text)
                    except Exception:
                        detail = response.text
                    raise RuntimeError(f"ElevenLabs voice cloning failed ({response.status_code}): {detail}")

                result = response.json()
                voice_id = (result.get("voice_id") or "").strip()
                if not voice_id:
                    raise RuntimeError("ElevenLabs did not return a voice ID.")
                result["voice_id"] = voice_id
                return result

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
                "language_variety": "unknown",
                "slang_and_regionalisms": [],
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
        self._tts_worker.speak(text)

    def shutdown(self):
        if self._recording:
            self.stop_recording()
        # 2s is arbitrary, just don't block forever on exit
        self._tts_worker.shutdown(2.0)
