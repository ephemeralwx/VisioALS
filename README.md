# VisioALS

**Free, open-source eye-gaze communication software for people living with ALS.**

VisioALS uses a standard webcam (no special hardware!) to let patients select AI-generated responses just by looking at the screen. Over time it learns the patient's preferences and preserves their unique voice so responses sound like *them*, not a machine.

<video src="https://github.com/user-attachments/assets/1184c544-43ea-4be5-8b80-68aae07f99c7" width="100%" controls></video>

---

## Quick Start (For Caregivers)

VisioALS runs on Windows 10/11 and macOS 12 or newer with a webcam. I tried my best to make it as easy to set up as possible.

1. **Download** the latest installer from the [Releases](../../releases) page
2. **Install the app:** run the Windows installer, or open the matching Mac DMG and drag VisioALS to Applications
3. **Launch VisioALS** and follow a setup wizard that will walk you through everything:
   - Webcam check
   - Server connection (pre-configured, just click next)
   - Patient profile creation (add past writing, audio, or video and optionally record a voice sample)
   - ElevenLabs voice cloning (the API key is stored in the operating system credential store)
4. **Calibration:** The patient looks at a moving dot on screen for a brief 14-second calibration phase
5. **Start talking:** The caregiver speaks and the user is able to respond simply by looking

Press **C** at any time to immediately recalibrate the gaze tracker.

That's it. If the setup provides trouble, contact me at kevin.w.xia@gmail.com and I can help! 
There are no configuration files and no terminal commands for normal use.

### Installing on a Mac

The GitHub release has two Mac downloads:

- **Apple Silicon** for Macs with an M1, M2, M3, M4, or later Apple chip
- **Intel** for older Intel-based Macs

Because the Mac app is released for free without an Apple Developer Program
membership, it is ad-hoc signed but not notarized. On first launch, macOS may
say that Apple could not verify the app is free of malware. Open it without
using Terminal:

1. Drag **VisioALS.app** from the DMG into **Applications**.
2. In Finder, Control-click or right-click VisioALS and choose **Open**.
3. Choose **Open** again in the confirmation dialog.

If macOS does not show the second Open button, try launching once, then go to
**System Settings → Privacy & Security** and choose **Open Anyway** for
VisioALS. The app asks for Camera and Microphone access when those features are
first used. These permissions can be changed later under **Privacy & Security**.

---

## How It Works

1. A caregiver asks a question out loud (recorded through the microphone)
2. Speech is transcribed locally on the device
3. AI generates 4 response options tailored to the patient's voice and preferences
4. The patient selects a response by gazing at one of four screen quadrants (or looking at "None of these" to reject all options and generate new ones)
5. The selected response is spoken through the patient's ElevenLabs cloned voice
   (with local system text-to-speech as a fallback)

The system supports both **eye tracking** and **head tracking**, so users can use whichever works better for their current level of mobility.

---

## Cool Features 

### Linguistic Identity Preservation

ALS takes away the ability to speak, but it shouldn't take away *how* someone speaks. During setup, users have the option to import  samples of their past writing/communication patterns - old texts, emails, social media posts, anything. VisioALS builds a detailed linguistic profile extracts:

- **(1)Vocabulary pattern** - word choices, favorite phrases, & sentence length
- **(2)Tone/register** - how formal casual they are, use of contractions, & humor style
- **(3)Signature expressions** - catchphrases, filler words, how they open and close messages

This profile is used every time the system generates response options, so the output sounds like the patient. Each response should represent their words, their style, and their personality.

Past audio and video recordings are transcribed locally with Whisper. Their
transcripts are added to the same corpus, allowing the profile to learn from
how the patient spoke as well as how they wrote. The original recordings are
also used to create an ElevenLabs Instant Voice Clone, so every selected
response is spoken with the active patient's cloned voice.

### Adaptive Preference Learning

Every time the patient picks a response (or rejects all four options), the system records that feedback and adapts future suggestions:

- Rejected responses are tracked and avoided in follow-up suggestions
- After enough interactions, the system extracts preference rules automatically (e.g., "prefers direct answers over emotional ones" or "avoids formal language")
- These rules are stored and applied to every future conversation

The result is that VisioALS gets better the more it's used. What starts as generic AI suggestions gradually becomes personalized to exactly how the patient wants to communicate.

### No Special Hardware

Most eye-tracking AAC devices cost thousands of dollars and require dedicated hardware. VisioALS uses a standard laptop webcam and an ensemble of machine learning models (Linear Regression, Polynomial Ridge Regression, Support Vector Regression, and K-Nearest Neighbors) to track where the patient is looking. The user just has to undergo a short calibration phase before each session.

---

## Technical Overview

### Architecture

```
VisioALS (desktop app)
  main.py               — Setup wizard, model downloads, app entry point
  ui.py                 — Gaze tracking UI, answer selection quadrants, interaction logging
  gaze.py               — Eye/head tracking via MediaPipe, calibration, ML prediction
  backend.py            — API client, audio recording, STT, TTS
  linguistic_profile.py — Linguistic identity extraction (vocabulary, tone, structure)
  patient_data.py       — Patient data management, corpus loading, preference storage
  embeddings.py         — Semantic similarity search over patient's writing corpus

visioals-worker/        — Cloudflare Worker (API proxy)
```

### Tech Stack

| Component | Technology |
|---|---|
| Eye/Head Tracking | MediaPipe Face Landmarker + scikit-learn ensemble |
| Speech-to-Text | faster-whisper (tiny.en, runs locally) |
| Text-to-Speech | ElevenLabs cloned voice; pyttsx3/system voice fallback |
| Response Generation | GPT-5.6 Luna via OpenRouter |
| Semantic Retrieval | all-MiniLM-L6-v2 sentence embeddings |
| Linguistic Analysis | spaCy + TF-IDF (scikit-learn) |
| Desktop UI | PySide6 (Qt) |
| Backend | Cloudflare Worker proxy |

### How the Linguistic Profile Works

VisioALS includes a profile extractor module that analyzes a patient's writing corpus and extracts the following:

avg sentence length, vocabulary diversity, sentence type, average response length, formality, recurring catchphrases, humor style, tone

The resulting profile is stored as JSON and summarized into a short description that's included in every response generation prompt.

### How Preference Learning Works

Interactions are logged to a JSONL file with the question, all options shown, what was selected, and what was rejected. After 20+ interactions, the system periodically sends recent interaction history to the backend, which extracts human-readable preference rules. These rules are then included in future generation prompts, creating a feedback loop that continuously improves response quality.

---

## Building from Source

### Prerequisites

- Python 3.12+
- Laptop with webcam
- Windows 10/11 or macOS 12+

### Setup

```bash
git clone https://github.com/ephemeralwx/VisioALS.git
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

### Building the free macOS downloads

On a Mac, the build script creates an architecture-specific `.app`, applies an
ad-hoc signature, smoke-tests the frozen dependencies, and packages the app in
a DMG:

```bash
chmod +x macos/build_macos.sh
VISIOALS_VERSION=1.0.0 ./macos/build_macos.sh
```

The result is written to `release/` and is labeled `Apple-Silicon` or `Intel`
based on the Mac that performed the build. No Apple account, certificate, or
paid program membership is used.

GitHub Actions builds both architectures on GitHub's hosted Mac runners. A
manual run of **Build macOS apps** verifies both builds without publishing
anything. Pushing a version tag creates a GitHub Release and attaches both
DMGs directly, without using temporary Actions artifact storage:

```bash
./macos/publish_release.sh 1.3.0
```

The publishing script requires a clean working tree. It pushes the current
branch, creates the numeric version tag, and pushes the tag to start the GitHub
build. If you prefer to do that manually, run `git tag -a v1.3.0 -m "VisioALS
v1.3.0"` followed by `git push origin v1.3.0`. Since these builds are not
notarized, users must follow the first-launch steps above.

## Browser Demo (macOS + ngrok)

The repository also includes a browser version for presentations. Each visitor
uses the camera, microphone, and speech output on their own device; the Mac only
serves the site and relays requests to the existing VisioALS API.

Prerequisites:

- Python 3.10 or newer
- The Python `requests` package (`pip install requests`; already included in `requirements.txt`)
- `ffmpeg` (`brew install ffmpeg`) to normalize Chrome/Safari recordings to WAV for voice cloning
- [ngrok](https://ngrok.com/download), signed in once with `ngrok config add-authtoken ...`
- A current Chrome or Edge browser is recommended for voice input

Start the local server and public HTTPS tunnel with one command:

```bash
chmod +x run_web_demo.sh
./run_web_demo.sh
```

Share `https://glutton-grab-squall.ngrok-free.dev`. This is the account's
permanent development domain and the launcher selects it automatically. Keep
that terminal and the Mac awake for the duration of the presentation; press
`Ctrl+C` when finished. The URL remains the same but is reachable only while
the server and ngrok are running.

The ngrok authtoken installed on the Mac must belong to the same dashboard
account that owns this domain. If ngrok reports `ERR_NGROK_320`, copy the
authtoken from that account's **Getting Started → Your Authtoken** page and run
`ngrok config add-authtoken YOUR_TOKEN` once before relaunching.

To use an assigned/reserved domain supported by your ngrok account, override it:

```bash
NGROK_URL=https://your-domain.ngrok.app ./run_web_demo.sh
```

The browser port intentionally matches the desktop application rather than
using a separate website interface. It has the same setup wizard, moving-dot
calibration, full-window response quadrants, gaze dwell behavior, Patient
Studio, learned preferences, speech flow, and keyboard controls:

- `Space`: begin calibration; in tracking mode, start/stop the question recording
- `M`: switch between eye and head tracking
- `R`: reset to the pre-calibration screen
- `C`: recalibrate immediately
- `F11`: enter/leave fullscreen
- `P`: open Patient Studio
- `V`: view learned preferences for the active patient
- `Q` or `Escape`: close VisioALS

Camera frames are processed locally in the visitor's browser and never
uploaded. Imported patient media is sent to the host Mac only when the visitor
explicitly generates a Patient Studio profile, so it can be transcribed and
normalized to WAV for voice cloning. The owner's ElevenLabs key remains on the
Mac server and is never sent to visitors. Profiles and interaction history are
saved in the visitor's browser. The target is a current desktop Chrome or Edge
window at least 980 × 720; the port is deliberately not redesigned for mobile.

For local-only development without a tunnel:

```bash
python3 web/server.py --port 8000
```

Then open `http://127.0.0.1:8000`. Camera access on audience devices requires
the HTTPS ngrok URL, not the Mac's LAN address.

### Backend Setup

The backend requires an `OPENROUTER_API_KEY` environment variable. See `RailwayAPI/.env.example`.

Voice cloning requires an ElevenLabs API key. The desktop app loads it from the
operating system credential manager, while the web server loads it from the
`ELEVENLABS_API_KEY` environment variable or the git-ignored project `.env`
file. The web credential remains server-side and is never requested from or
sent to visitors' browsers.

---

## License

MIT
