# VisioALS

**Free, open-source eye-gaze communication software for people living with ALS.**

VisioALS uses a standard webcam (no special hardware!) to let patients select AI-generated responses just by looking at the screen. Over time it learns the patient's preferences and preserves their unique voice so responses sound like *them*, not a machine.

<video src="https://github.com/user-attachments/assets/a585578b-41c8-445c-963a-4a036e3e8994" width="100%" controls></video>

---

## Quick Start (For Caregivers)

VisioALS runs on any Windows 10/11 laptop with a webcam. I tried my best to make it as easy to set up as possible. 

1. **Download** the latest installer from the [Releases](../../releases) page
2. **Run the installer** and follow the prompts
3. **Launch VisioALS** and follow a setup wizard that will walk you through everything:
   - Webcam check
   - Server connection (pre-configured, just click next)
   - Patient profile creation (import some of the patient's past writing (texts, emails, anything they've written that sounds like them)
4. **Calibration:** The patient looks at dots on screen for a brief ~30 second calibration phase
5. **Start talking:** The caregiver speaks and the user is able to respond simply by looking

That's it. If the setup provides trouble, contact me at kevin.w.xia@gmail.com and I can help! 
There are no no configuration files and no terminal commands.

---

## How It Works

1. A caregiver asks a question out loud (recorded through the microphone)
2. Speech is transcribed locally on the device
3. AI generates 4 response options tailored to the patient's voice and preferences
4. The patient selects a response by gazing at one of four screen quadrants (or looking at "None of these" to reject all options and generate new ones)
5. The selected response is spoken aloud through text-to-speech

The system supports both **eye tracking** and **head tracking**, so users can use whichever works better for their current level of mobility.

---

## Cool Features 

### Linguistic Identity Preservation

ALS takes away the ability to speak, but it shouldn't take away *how* someone speaks. During setup, users have the option to import  samples of their past writing/communication patterns - old texts, emails, social media posts, anything. VisioALS builds a detailed linguistic profile extracts:

- **(1)Vocabulary pattern** - word choices, favorite phrases, & sentence length
- **(2)Tone/register** - how formal casual they are, use of contractions, & humor style
- **(3)Signature expressions** - catchphrases, filler words, how they open and close messages

This profile is used every time the system generates response options, so the output sounds like the patient. Each response should represent their words, their style, and their personality.

### Adaptive Preference Learning (RLHF)

Every time the patient picks a response (or rejects all four options), the system learns. This is a form of reinforcement learning from human feedback:

- Rejected responses are tracked and avoided in follow-up suggestions
- After enough interactions, the system extracts preference rules automatically (e.g., "prefers direct answers over emotional ones" or "avoids formal language")
- These rules are stored and applied to every future conversation

The result is that VisioALS gets better the more it's used. What starts as generic AI suggestions gradually becomes personalized to exactly how the patient wants to communicate.

### No Special Hardware

Most eye-tracking AAC devices cost thousands of dollars and require dedicated hardware. VisioALS uses a standard laptop webcam and an ensemble of machine learning models (Random Forest, Gradient Boosting, KNN) to track where the patient is looking. The user just has to undergo a short calibration phase before each session. 

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
| Response Generation | GPT-5.4-nano via OpenRouter |
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
- Windows 10/11

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

### Backend Setup

The backend requires an `OPENROUTER_API_KEY` environment variable. See `RailwayAPI/.env.example`.

---

## License

MIT
