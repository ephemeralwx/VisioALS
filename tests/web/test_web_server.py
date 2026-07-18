import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web"))

import server as web_server  # noqa: E402
from server import make_server  # noqa: E402


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *_args):
        return

    def do_GET(self):  # noqa: N802
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        self._send(200, {"status": "ok"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.calls.append((self.path, body))
        if self.path == "/generate-options":
            self._send(200, {"options": ["Yes", "No", "Maybe", "Later"]})
        elif self.path == "/expand-response":
            self._send(200, {"expanded": "Yes, that sounds good."})
        else:
            self._send(404, {"error": "missing"})

    def _send(self, status, payload):
        content = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class WebServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstreamHandler)
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()
        upstream_url = f"http://127.0.0.1:{cls.upstream.server_address[1]}"
        cls.server = make_server("127.0.0.1", 0, upstream_url, elevenlabs_api_key="server-eleven-key")
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.upstream.shutdown()
        cls.upstream.server_close()

    def request(self, path, payload=None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, response.headers, response.read()

    def test_index_is_served_with_browser_security_headers(self):
        status, headers, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"VisioALS", body)
        self.assertIn(b"Press P to open Studio", body)
        self.assertIn(b"Press C to recalibrate at any time", body)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("camera=(self)", headers["Permissions-Policy"])

    def test_setup_reserves_a_visible_footer_row(self):
        status, _, body = self.request("/styles.css")
        self.assertEqual(status, 200)
        css = body.decode()
        self.assertIn("grid-template-rows: 30px minmax(0, 1fr) 58px", css)
        self.assertIn("overflow-y: auto", css)

    def test_gaze_core_module_is_served_as_javascript(self):
        status, headers, body = self.request("/gaze-core.mjs")
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers["Content-Type"])
        self.assertIn(b"StandardScaler", body)

    def test_local_and_upstream_health(self):
        self.assertEqual(json.loads(self.request("/healthz")[2]), {"status": "ok"})
        self.assertEqual(json.loads(self.request("/api/health")[2]), {"status": "ok"})

    def test_generate_options_is_validated_and_relayed(self):
        payload = {"question": "Would you like some tea?", "history": []}
        status, _, body = self.request("/api/generate-options", payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["options"], ["Yes", "No", "Maybe", "Later"])
        self.assertEqual(FakeUpstreamHandler.calls[-1], ("/generate-options", payload))

    def test_missing_question_is_rejected_before_upstream(self):
        before = len(FakeUpstreamHandler.calls)
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request("/api/generate-options", {"question": ""})
        self.assertEqual(context.exception.code, 400)
        self.assertEqual(len(FakeUpstreamHandler.calls), before)

    def test_unknown_api_endpoint_is_not_relayed(self):
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request("/api/not-allowed", {"question": "test"})
        self.assertEqual(context.exception.code, 404)

    def test_web_ui_does_not_request_or_send_an_elevenlabs_key(self):
        html = self.request("/index.html")[2].decode()
        javascript = self.request("/app.js")[2].decode()
        self.assertNotIn("studio-key", html)
        self.assertNotIn('form.append("api_key"', javascript)
        self.assertNotIn("api_key:state", javascript)

    def test_tts_uses_server_owned_elevenlabs_key(self):
        response = mock.Mock(ok=True, content=b"mp3")
        with mock.patch.object(web_server.requests, "post", return_value=response) as post:
            status, _, body = self.request("/api/tts", {"voice_id": "voice-123", "text": "Hello"})
        self.assertEqual(status, 200)
        self.assertEqual(body, b"mp3")
        self.assertEqual(post.call_args.kwargs["headers"]["xi-api-key"], "server-eleven-key")

    def test_empty_cached_key_is_reloaded_from_server_side_credentials(self):
        handler = object.__new__(web_server.VisioALSHandler)
        handler.server = mock.Mock(elevenlabs_api_key="")
        with mock.patch.object(web_server, "_load_elevenlabs_api_key", return_value="new-owner-key"):
            self.assertEqual(handler.elevenlabs_api_key, "new-owner-key")
        self.assertEqual(handler.server.elevenlabs_api_key, "new-owner-key")

    def test_browser_recording_is_normalized_to_wav_for_voice_cloning(self):
        def fake_ffmpeg(args, **_kwargs):
            Path(args[-1]).write_bytes(b"RIFF-normalized-wave")
            return mock.Mock(returncode=0)

        with mock.patch.object(web_server.shutil, "which", return_value="/usr/bin/ffmpeg"), mock.patch.object(
            web_server.subprocess, "run", side_effect=fake_ffmpeg
        ) as run:
            name, content, content_type = web_server._voice_clone_file(
                "recording.webm", "audio/webm;codecs=opus", b"browser-audio"
            )

        self.assertEqual(name, "recording.wav")
        self.assertEqual(content, b"RIFF-normalized-wave")
        self.assertEqual(content_type, "audio/wav")
        self.assertIn("pcm_s16le", run.call_args.args[0])

    def test_media_suffix_uses_mime_type_over_a_wrong_browser_extension(self):
        self.assertEqual(web_server._media_suffix("recording.webm", "audio/mp4"), ".m4a")


if __name__ == "__main__":
    unittest.main()
