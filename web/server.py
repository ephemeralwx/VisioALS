#!/usr/bin/env python3
"""Small web server and API relay for the VisioALS browser demo."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import warnings
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import cgi


WEB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_ROOT.parent
DEFAULT_UPSTREAM = "https://visioals-backend.visioals.workers.dev"
MAX_REQUEST_BYTES = 512 * 1024
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_POST_ENDPOINTS = {
    "/api/generate-options": "/generate-options",
    "/api/expand-response": "/expand-response",
    "/api/analyze-style": "/analyze-style",
    "/api/analyze-preferences": "/analyze-preferences",
    "/api/telemetry": "/telemetry",
}


class SlidingWindowLimiter:
    """A modest per-client guard against an accidentally shared demo URL."""

    def __init__(self, limit: int = 40, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits[client]
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


RATE_LIMITER = SlidingWindowLimiter()
_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()

_MEDIA_SUFFIXES = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}


def _load_elevenlabs_api_key() -> str:
    """Load the server-owned ElevenLabs credential without exposing it to clients."""

    env_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        for raw_line in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "ELEVENLABS_API_KEY":
                return value.strip().strip('"').strip("'")
    except (FileNotFoundError, OSError):
        pass
    try:
        import keyring

        return (keyring.get_password("VisioALS", "elevenlabs_api_key") or "").strip()
    except Exception:
        return ""


def _media_suffix(filename: str | None, content_type: str | None) -> str:
    """Choose an extension that agrees with the bytes produced by MediaRecorder."""

    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime in _MEDIA_SUFFIXES:
        return _MEDIA_SUFFIXES[mime]
    return Path(filename or "sample.webm").suffix.lower() or ".webm"


def _voice_clone_file(filename: str | None, content_type: str | None, content: bytes) -> tuple[str, bytes, str]:
    """Convert browser/container audio to a predictable PCM WAV for ElevenLabs."""

    suffix = _media_suffix(filename, content_type)
    # Regular WAV and MP3 imports are already reliable ElevenLabs inputs. Browser
    # recordings use container formats whose codecs vary by browser, so normalize
    # those before cloning.
    if suffix not in {".webm", ".ogg", ".opus", ".m4a", ".mp4"}:
        return Path(filename or f"sample{suffix}").name, content, content_type or "application/octet-stream"

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to prepare browser voice recordings")
    with tempfile.TemporaryDirectory(prefix="visioals-audio-") as directory:
        source = Path(directory) / f"source{suffix}"
        output = Path(directory) / "voice-sample.wav"
        source.write_bytes(content)
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(output)],
            capture_output=True,
            check=False,
            timeout=120,
        )
        if result.returncode or not output.is_file() or not output.stat().st_size:
            raise ValueError("the browser recording could not be converted to WAV")
        return f"{Path(filename or 'voice-sample').stem}.wav", output.read_bytes(), "audio/wav"


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _validate_payload(path: str, payload: dict[str, Any]) -> str | None:
    """Return a user-safe validation error, or None when the body is valid."""

    if path == "/api/generate-options":
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            return "A question is required."
        if len(question) > 1_000:
            return "The question is too long."
    elif path == "/api/expand-response":
        if not isinstance(payload.get("question"), str) or not payload["question"].strip():
            return "A question is required."
        if not isinstance(payload.get("response"), str) or not payload["response"].strip():
            return "A selected response is required."
    elif path == "/api/analyze-style":
        samples = payload.get("sample_texts")
        if not isinstance(samples, list) or not samples:
            return "At least one writing sample is required."
        if len(samples) > 50 or any(not isinstance(item, str) for item in samples):
            return "Writing samples must be a list of at most 50 text items."
        if sum(len(item) for item in samples) > 100_000:
            return "The writing sample is too large for this demo."
    elif path == "/api/analyze-preferences":
        interactions = payload.get("interactions")
        if not isinstance(interactions, list) or len(interactions) > 100:
            return "Interactions must be a list of at most 100 items."
    elif path == "/api/telemetry":
        duration = payload.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            return "A positive session duration is required."
    return None


class VisioALSHandler(SimpleHTTPRequestHandler):
    server_version = "VisioALSWeb/1.0"

    @property
    def upstream(self) -> str:
        return str(getattr(self.server, "upstream", DEFAULT_UPSTREAM)).rstrip("/")

    @property
    def elevenlabs_api_key(self) -> str:
        key = str(getattr(self.server, "elevenlabs_api_key", "")).strip()
        if not key:
            # The desktop app may save the owner key after this server starts.
            # Retry the server-side sources so visitors never need a key field or
            # a server restart.
            key = _load_elevenlabs_api_key()
            if key:
                self.server.elevenlabs_api_key = key  # type: ignore[attr-defined]
        return key

    def log_message(self, fmt: str, *args: object) -> None:
        forwarded = self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        client = forwarded or self.client_address[0]
        print(f"[{self.log_date_time_string()}] {client} {fmt % args}", flush=True)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(self), microphone=(self)")
        self.send_header("Cross-Origin-Resource-Policy", "same-site")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net 'wasm-unsafe-eval'; "
            "style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "connect-src 'self' https://cdn.jsdelivr.net https://storage.googleapis.com; "
            "worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
        )

    def end_headers(self) -> None:
        self._security_headers()
        super().end_headers()

    def _send_json(self, status: int, payload: Any) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _client_key(self) -> str:
        forwarded = self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        return forwarded or self.client_address[0]

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        if path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/api/health":
            self._proxy_health()
            return

        decoded = urllib.parse.unquote(path)
        normalized = posixpath.normpath(decoded).lstrip("/")
        if normalized in ("", "."):
            normalized = "index.html"
        candidate = (WEB_ROOT / normalized).resolve()
        if WEB_ROOT not in candidate.parents and candidate != WEB_ROOT:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            # The app is intentionally a single page.
            candidate = WEB_ROOT / "index.html"

        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.suffix in {".html", ".js", ".css"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Presentation builds change frequently; stale JS would strand audience
        # members on an older behavior while the Mac is serving newer HTML.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        if path in {"/api/transcribe-media", "/api/clone-voice"}:
            if not RATE_LIMITER.allow(self._client_key()):
                self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "This demo is busy. Please wait a minute and try again."})
                return
            self._handle_media_upload(path)
            return
        if path == "/api/tts":
            if not RATE_LIMITER.allow(self._client_key()):
                self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "This demo is busy. Please wait a minute and try again."})
                return
            self._handle_tts()
            return
        upstream_path = ALLOWED_POST_ENDPOINTS.get(path)
        if upstream_path is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        if not RATE_LIMITER.allow(self._client_key()):
            self._send_json(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": "This demo is busy. Please wait a minute and try again."},
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Invalid request size."})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "The request body must be JSON."})
            return
        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "The request must be a JSON object."})
            return
        validation_error = _validate_payload(path, payload)
        if validation_error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": validation_error})
            return
        self._proxy_json(upstream_path, payload)

    def _read_json_body(self, maximum: int = MAX_REQUEST_BYTES) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > maximum:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Invalid request size."})
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "The request body must be JSON."})
            return None
        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "The request must be a JSON object."})
            return None
        return payload

    def _multipart_form(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "The uploaded media is too large."})
            return None
        if not self.headers.get("Content-Type", "").startswith("multipart/form-data"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "A multipart media upload is required."})
            return None
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": str(length),
            },
            keep_blank_values=True,
        )

    @staticmethod
    def _form_files(form) -> list:
        if "files" not in form:
            return []
        value = form["files"]
        return value if isinstance(value, list) else [value]

    def _handle_media_upload(self, path: str) -> None:
        form = self._multipart_form()
        if form is None:
            return
        files = [item for item in self._form_files(form) if getattr(item, "file", None)]
        if not files:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "At least one media file is required."})
            return
        if path == "/api/transcribe-media":
            item = files[0]
            suffix = _media_suffix(item.filename, item.type)
            temp_path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                    temp_path = temp.name
                    while chunk := item.file.read(1024 * 1024):
                        temp.write(chunk)
                text = self._transcribe_file(temp_path)
                self._send_json(HTTPStatus.OK, {"text": text})
            except Exception as exc:
                print(f"media transcription failed: {exc}", flush=True)
                self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "The media could not be transcribed."})
            finally:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
            return

        api_key = self.elevenlabs_api_key
        patient_name = form.getfirst("patient_name", "Patient").strip()[:80] or "Patient"
        if not api_key:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "An ElevenLabs API key is required."})
            return
        try:
            request_files = []
            for item in files:
                content = item.file.read()
                if content:
                    request_files.append(("files", _voice_clone_file(item.filename, item.type, content)))
            if not request_files:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "At least one non-empty media file is required."})
                return
            response = requests.post(
                "https://api.elevenlabs.io/v1/voices/add",
                headers={"xi-api-key": api_key},
                data={"name": f"VisioALS - {patient_name}", "description": "Voice clone created by VisioALS from user-provided recordings.", "remove_background_noise": "false"},
                files=request_files,
                timeout=300,
            )
            if not response.ok:
                self._send_json(HTTPStatus.BAD_GATEWAY, {"error": f"ElevenLabs voice cloning failed ({response.status_code})."})
                return
            result = response.json()
            voice_id = str(result.get("voice_id", "")).strip()
            if not voice_id:
                raise ValueError("missing voice id")
            self._send_json(HTTPStatus.OK, {"voice_id": voice_id, "requires_verification": bool(result.get("requires_verification", False))})
        except (requests.RequestException, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            print(f"voice cloning failed: {exc}", flush=True)
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "The voice recording could not be prepared or cloned."})

    @staticmethod
    def _transcribe_file(path: str) -> str:
        global _WHISPER_MODEL
        with _WHISPER_LOCK:
            if _WHISPER_MODEL is None:
                from faster_whisper import WhisperModel

                model_dir = os.environ.get("VISIOALS_MODEL_DIR", str(Path.home() / "VisioALS" / "models"))
                _WHISPER_MODEL = WhisperModel("tiny.en", device="cpu", compute_type="int8", download_root=model_dir)
            segments, _ = _WHISPER_MODEL.transcribe(path, vad_filter=True)
            return " ".join(segment.text for segment in segments).strip()

    def _handle_tts(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        api_key = self.elevenlabs_api_key
        voice_id = str(payload.get("voice_id", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not api_key or not voice_id or not text or len(text) > 5_000:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "A valid voice, key, and text are required."})
            return
        try:
            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{urllib.parse.quote(voice_id, safe='')}",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                json={"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "use_speaker_boost": True}},
                timeout=60,
            )
            if not response.ok or not response.content:
                raise requests.RequestException(f"status {response.status_code}")
            self._send_binary(HTTPStatus.OK, response.content, "audio/mpeg")
        except requests.RequestException as exc:
            print(f"ElevenLabs TTS failed: {exc}", flush=True)
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "Cloned speech is temporarily unavailable."})

    def _proxy_health(self) -> None:
        try:
            response = requests.get(f"{self.upstream}/health", timeout=8)
            data = response.json()
            self._send_json(response.status_code, data)
        except (requests.RequestException, ValueError):
            self._send_json(HTTPStatus.BAD_GATEWAY, {"status": "unavailable"})

    def _proxy_json(self, upstream_path: str, payload: dict[str, Any]) -> None:
        try:
            response = requests.post(
                f"{self.upstream}{upstream_path}",
                json=payload,
                headers={"User-Agent": "VisioALS-Web/1.0"},
                timeout=45,
            )
            response_body = response.content[:MAX_REQUEST_BYTES]
            status = response.status_code
        except requests.RequestException:
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": "The AI service is temporarily unavailable. Please try again."},
            )
            return
        try:
            response_data = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "The AI service returned an invalid response."})
            return
        if status >= 400:
            # Do not expose upstream/provider diagnostics or credentials in the public demo.
            message = response_data.get("error", "The AI request failed.") if isinstance(response_data, dict) else "The AI request failed."
            if "OpenRouter returned" in str(message):
                message = "The AI service could not complete that request."
            self._send_json(status, {"error": message})
            return
        self._send_json(status, response_data)


def make_server(host: str, port: int, upstream: str, elevenlabs_api_key: str | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), VisioALSHandler)
    server.daemon_threads = True
    server.upstream = upstream  # type: ignore[attr-defined]
    server.elevenlabs_api_key = _load_elevenlabs_api_key() if elevenlabs_api_key is None else elevenlabs_api_key  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the VisioALS browser demo.")
    parser.add_argument("--host", default=os.environ.get("VISIOALS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument(
        "--upstream",
        default=os.environ.get("VISIOALS_API_URL", DEFAULT_UPSTREAM),
        help="Existing VisioALS API worker URL.",
    )
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.upstream)
    print(f"VisioALS web demo: http://{args.host}:{args.port}", flush=True)
    print(f"AI relay: {args.upstream.rstrip('/')}", flush=True)
    if not server.elevenlabs_api_key:  # type: ignore[attr-defined]
        print("Warning: ELEVENLABS_API_KEY is not configured; voice cloning and cloned speech are unavailable.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
