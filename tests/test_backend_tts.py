import queue
import sys
import threading
import types

import numpy as np

sys.modules.setdefault("sounddevice", types.SimpleNamespace())
import backend


def test_tts_falls_back_to_windows_system_speech(monkeypatch):
    worker = object.__new__(backend.TTSWorker)
    calls = []

    monkeypatch.setattr(backend, "pyttsx3", None)
    monkeypatch.setattr(backend.platform, "system", lambda: "Windows")
    monkeypatch.setattr(backend.shutil, "which", lambda name: "powershell" if name == "powershell" else None)

    def fake_run(command, check, stdout, stderr):
        calls.append(command)

    monkeypatch.setattr(backend.subprocess, "run", fake_run)

    assert worker._speak_with_pyttsx3("hello") is False
    assert worker._speak_with_system("hello") is True
    assert calls
    assert calls[0][0] == "powershell"
    assert calls[0][-1] == "hello"


def test_tts_speak_ignores_empty_text():
    worker = object.__new__(backend.TTSWorker)
    worker._queue = queue.Queue()

    worker.speak("   ")

    assert worker._queue.empty()


def test_tts_prefers_system_say_on_macos(monkeypatch):
    worker = object.__new__(backend.TTSWorker)
    worker._queue = queue.Queue()
    worker._queue.put("hello")
    worker._queue.put(None)
    calls = []

    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        worker,
        "_speak_with_system",
        lambda text: calls.append(("system", text)) or True,
    )
    monkeypatch.setattr(
        worker,
        "_speak_with_pyttsx3",
        lambda text: calls.append(("pyttsx3", text)) or True,
    )

    worker._run()

    assert calls == [("system", "hello")]


def test_tts_uses_pyttsx3_if_say_is_unavailable(monkeypatch):
    worker = object.__new__(backend.TTSWorker)
    worker._queue = queue.Queue()
    worker._queue.put("hello")
    worker._queue.put(None)
    calls = []

    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        worker,
        "_speak_with_system",
        lambda text: calls.append(("system", text)) or False,
    )
    monkeypatch.setattr(
        worker,
        "_speak_with_pyttsx3",
        lambda text: calls.append(("pyttsx3", text)) or True,
    )

    worker._run()

    assert calls == [("system", "hello"), ("pyttsx3", "hello")]


def test_tts_prefers_elevenlabs_clone(monkeypatch):
    worker = object.__new__(backend.TTSWorker)
    worker._queue = queue.Queue()
    worker._queue.put("hello")
    worker._queue.put(None)
    calls = []

    monkeypatch.setattr(
        worker,
        "_speak_with_elevenlabs",
        lambda text: calls.append(("elevenlabs", text)) or True,
    )
    monkeypatch.setattr(
        worker,
        "_speak_with_system",
        lambda text: calls.append(("system", text)) or True,
    )
    monkeypatch.setattr(
        worker,
        "_speak_with_pyttsx3",
        lambda text: calls.append(("pyttsx3", text)) or True,
    )

    worker._run()

    assert calls == [("elevenlabs", "hello")]


def test_elevenlabs_pcm_is_played(monkeypatch):
    worker = object.__new__(backend.TTSWorker)
    worker._config_lock = threading.Lock()
    worker._elevenlabs_api_key = "test-key"
    worker._elevenlabs_voice_id = "voice-123"
    played = []

    class Response:
        content = np.array([0, 100, -100], dtype=np.int16).tobytes()

        @staticmethod
        def raise_for_status():
            return None

    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")
    monkeypatch.setattr(backend.requests, "post", fake_post)
    monkeypatch.setattr(
        backend.sd,
        "play", lambda audio, samplerate, blocking: played.append(
            (audio.tolist(), samplerate, blocking)
        ),
        raising=False,
    )

    assert worker._speak_with_elevenlabs("hello") is True
    assert captured["params"] == {"output_format": "pcm_24000"}
    assert np.allclose(played[0][0], [0.0, 100 / 32768, -100 / 32768])
    assert played[0][1:] == (24000, True)


def test_elevenlabs_uses_native_mp3_playback_on_macos(monkeypatch):
    worker = object.__new__(backend.TTSWorker)
    worker._config_lock = threading.Lock()
    worker._elevenlabs_api_key = "test-key"
    worker._elevenlabs_voice_id = "voice-123"
    captured = {}

    class Response:
        content = b"ID3-valid-mp3-data"

        @staticmethod
        def raise_for_status():
            return None

    def fake_post(*args, **kwargs):
        captured.update(kwargs)
        return Response()

    played = []
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        backend.shutil,
        "which",
        lambda command: "/usr/bin/afplay" if command == "afplay" else None,
    )
    monkeypatch.setattr(backend.requests, "post", fake_post)
    monkeypatch.setattr(
        worker,
        "_play_mp3_with_afplay",
        lambda content: played.append(content),
    )

    assert worker._speak_with_elevenlabs("hello") is True
    assert captured["params"] == {"output_format": "mp3_44100_128"}
    assert played == [Response.content]


def test_pcm_playback_rejects_encoded_audio(monkeypatch):
    played = []
    monkeypatch.setattr(
        backend.sd,
        "play",
        lambda *args, **kwargs: played.append((args, kwargs)),
        raising=False,
    )

    try:
        backend.TTSWorker._play_pcm(b"ID3-not-pcm", sample_rate=24000)
    except RuntimeError as error:
        assert "encoded audio" in str(error)
    else:
        raise AssertionError("Expected encoded audio to be rejected")

    assert played == []


def test_clone_voice_posts_audio_files(tmp_path, monkeypatch):
    sample = tmp_path / "voice.wav"
    sample.write_bytes(b"RIFF-test")
    captured = {}

    class Response:
        ok = True

        @staticmethod
        def json():
            return {"voice_id": "voice-abc", "requires_verification": False}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(backend.requests, "post", fake_post)
    client = object.__new__(backend.BackendClient)
    client._elevenlabs_api_key = ""

    result = client.clone_voice("Alice", [str(sample)], api_key="test-key")

    assert result["voice_id"] == "voice-abc"
    assert captured["url"].endswith("/v1/voices/add")
    assert captured["headers"]["xi-api-key"] == "test-key"
    assert captured["data"]["name"] == "VisioALS - Alice"
    assert len(captured["files"]) == 1
