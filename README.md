# VisioALS

Eye-gaze communication tool for ALS patients. Uses a standard webcam to track eye movements and lets patients select responses by looking at on-screen options.

This software only works on Windows devices.

<video src="https://github.com/user-attachments/assets/55762027-76bf-466a-aeee-92e8c25af38f" width="100%" controls>
</video>

## How It Works

1. A caregiver asks a question via voice (recorded through the microphone)
2. Speech is transcribed locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
3. An AI backend generates 4 possible response options
4. The patient selects a response by gazing at one of four screen quadrants
5. The selected response is expanded into a natural sentence and spoken aloud via text-to-speech

## Download

Download the latest installer from the [Releases](../../releases) page.

## Architecture

```
VisioALS (desktop app)
  main.py          — Setup wizard, model download, app entry point
  ui.py            — Gaze tracking UI, answer selection quadrants
  gaze.py          — Eye tracking via MediaPipe, calibration, ML prediction
  backend.py       — API client, audio recording, STT, TTS

visioals-worker/   — Cloudflare Worker (API proxy + telemetry)
RailwayAPI/        — FastAPI backend (GPT-powered response generation)
```

## Tech Stack

- **Eye Tracking**: MediaPipe Face Landmarker + scikit-learn (Random Forest, Gradient Boosting, KNN ensemble)
- **Speech-to-Text**: faster-whisper (tiny.en model, runs locally)
- **Text-to-Speech**: pyttsx3 (Windows SAPI5)
- **Response Generation**: GPT-4.1-nano via OpenRouter
- **Desktop UI**: PySide6 (Qt)
- **Backend**: FastAPI on Railway + Cloudflare Worker proxy

## Building from Source

### Prerequisites

- Python 3.12+
- A webcam
- Windows 10/11

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/VisioALS.git
cd VisioALS
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Building the Installer

1. Install [Inno Setup 6](https://jrsoftware.org/isdl.php)
2. Run `build.bat`, or manually:

```bash
env\Scripts\activate
pip install pyinstaller
pyinstaller --noconfirm VisioALS.spec
```

Then open `installer.iss` in Inno Setup and compile.

### Backend Setup

The Railway backend requires an `OPENROUTER_API_KEY` environment variable. See `RailwayAPI/.env.example`.

## License

MIT
