# VisioALS

**Free, open-source eye-gaze communication software for people living with ALS.**

VisioALS uses a standard webcam — no special hardware — to let patients select AI-generated responses just by looking at the screen. Over time it learns the patient's preferences and preserves their unique voice so responses sound like *them*, not a machine.

<video src="https://github.com/user-attachments/assets/a585578b-41c8-445c-963a-4a036e3e8994" width="100%" controls></video>

---

## Quick Start (For Caregivers)

VisioALS runs on any Windows 10/11 computer with a webcam. No technical knowledge required.

1. **Download** the latest installer from the [Releases](../../releases) page
2. **Run the installer** and follow the prompts
3. **Launch VisioALS** — a setup wizard walks you through everything:
   - Webcam check
   - Server connection (pre-configured, just click next)
   - Patient profile creation (import some of the patient's past writing — texts, emails, anything they've written)
4. **Calibrate** — the patient looks at dots on screen so the system learns their eye movement
5. **Start talking** — the caregiver speaks, the patient responds by looking

That's it. No API keys, no configuration files, no terminal commands.

---

## How It Works

1. A caregiver asks a question out loud (recorded through the microphone)
2. Speech is transcribed locally on the device — nothing leaves the computer for this step
3. AI generates 4 response options tailored to the patient's voice and preferences
4. The patient selects a response by gazing at one of four screen quadrants (or looking at "None of these" to reject all options)
5. The selected response is spoken aloud through text-to-speech

The system supports both **eye tracking** and **head tracking**, so patients can use whichever works better for their current level of mobility.

---

## Key Features

### Linguistic Identity Preservation

ALS takes away the ability to speak, but it shouldn't take away *how* someone speaks. During setup, caregivers import samples of the patient's past writing — old texts, emails, social media posts, anything. VisioALS builds a detailed linguistic profile that captures:

- **Vocabulary patterns** — word choices, favorite phrases, sentence length
- **Tone and register** — how formal or casual they are, use of contractions, humor style
- **Signature expressions** — catchphrases, filler words, how they open and close messages

This profile is used every time the system generates response options, so the output sounds like the patient — their words, their style, their personality.

### Adaptive Preference Learning (RLHF)

Every time the patient picks a response (or rejects all four options), the system learns. This is a form of reinforcement learning from human feedback:

- Rejected responses are tracked and avoided in follow-up suggestions
- After enough interactions, the system extracts preference rules automatically (e.g., "prefers direct answers over emotional ones" or "avoids formal language")
- These rules are stored and applied to every future conversation

The result is that VisioALS gets better the more it's used. What starts as generic AI suggestions gradually becomes personalized to exactly how the patient wants to communicate.

### No Special Hardware

Most eye-tracking AAC devices cost thousands of dollars and require dedicated hardware. VisioALS uses a standard laptop webcam and an ensemble of machine learning models (Random Forest, Gradient Boosting, KNN) to track where the patient is looking. A short calibration step at the start of each session is all that's needed.

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
| Text-to-Speech | pyttsx3 (Windows SAPI5) |
| Response Generation | GPT-4.1-nano via OpenRouter |
| Semantic Retrieval | all-MiniLM-L6-v2 sentence embeddings |
| Linguistic Analysis | spaCy + TF-IDF (scikit-learn) |
| Desktop UI | PySide6 (Qt) |
| Backend | Cloudflare Worker proxy |

### How the Linguistic Profile Works

The `LinguisticProfileExtractor` analyzes a patient's writing corpus across five dimensions:

1. **Vocabulary metrics** — average sentence length, type-token ratio (vocabulary diversity), distinctive words and phrases identified via TF-IDF analysis
2. **Structural patterns** — distribution of sentence types (declarative, questions, fragments, imperatives), average response length
3. **Register and tone** — formality score computed from contraction rate, average word length, and passive voice usage
4. **Signature phrases** — most common filler words, sentence openers/closers, and recurring n-gram catchphrases
5. **Subjective style** — humor style, emotional valence, and tone description via LLM analysis of representative samples

The resulting profile is stored as JSON and summarized into a natural-language description that's included in every response generation prompt.

### How Preference Learning Works

Interactions are logged to a JSONL file with the question, all options shown, what was selected, and what was rejected. After 20+ interactions, the system periodically sends recent interaction history to the backend, which extracts human-readable preference rules. These rules are then included in future generation prompts, creating a feedback loop that continuously improves response quality.

---

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

The backend requires an `OPENROUTER_API_KEY` environment variable. See `RailwayAPI/.env.example`.

---

## License

MIT
