"""
VisioALS Railway Backend — 3 endpoints for the desktop app.
"""

import os
import json
import re
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="VisioALS Backend")

# Lazy-init: created on first request, not at import time.
_client = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client

MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4.1-nano")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class QAPair(BaseModel):
    question: str
    answer: str

class OptionsRequest(BaseModel):
    question: str
    history: list[QAPair] = []
    rejected: list[str] = []
    linguistic_profile_summary: str | None = None
    exemplars: list[str] | None = None
    preference_rules: list[str] | None = None

class OptionsResponse(BaseModel):
    options: list[str]

class ExpandRequest(BaseModel):
    question: str
    response: str
    history: list[QAPair] = []
    linguistic_profile_summary: str | None = None
    exemplars: list[str] | None = None

class ExpandResponse(BaseModel):
    expanded: str

class StyleAnalysisRequest(BaseModel):
    sample_texts: list[str]

class StyleAnalysisResponse(BaseModel):
    humor_style: str
    tone_description: str
    emotional_valence: str
    personality_notes: str

class PreferenceAnalysisRequest(BaseModel):
    interactions: list[dict]

class PreferenceAnalysisResponse(BaseModel):
    rules: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_array(raw: str) -> list:
    """Parse a JSON array from GPT output, stripping markdown fences if present."""
    # Strip ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    return parsed


def _build_identity_block(
    profile_summary: str | None = None,
    exemplars: list[str] | None = None,
    preference_rules: list[str] | None = None,
) -> str:
    """Build the patient communication model block for prompt injection."""
    if not profile_summary and not exemplars and not preference_rules:
        return ""
    parts = []
    if profile_summary:
        parts.append(f"Voice: {profile_summary}")
    if preference_rules:
        parts.append("Preferences:\n" + "\n".join(f"- {r}" for r in preference_rules))
    if exemplars:
        parts.append(
            "Style examples from the patient's past writing:\n"
            + "\n".join(f'  "{e}"' for e in exemplars[:5])
        )
    return (
        "Patient Communication Model:\n"
        + "\n".join(parts) + "\n"
        "Generated answers MUST match this patient's voice and preferences.\n\n"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-options", response_model=OptionsResponse)
def generate_options(req: OptionsRequest):
    history_block = ""
    if req.history:
        pairs = req.history[-5:]  # last 5 Q&A pairs to keep prompt short
        lines = [f"  Q: \"{p.question}\" → A: \"{p.answer}\"" for p in pairs]
        history_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

    rejected_block = ""
    if req.rejected:
        rejected_block = (
            "The patient already rejected these options — do NOT repeat or rephrase them:\n"
            + ", ".join(f'"{r}"' for r in req.rejected) + "\n"
            "Generate completely different answers.\n\n"
        )

    identity_block = _build_identity_block(
        req.linguistic_profile_summary, req.exemplars, req.preference_rules,
    )

    prompt = (
        f"{history_block}"
        f"{identity_block}"
        f"A caregiver asked an ALS patient: \"{req.question}\"\n\n"
        f"{rejected_block}"
        "Generate exactly 4 short possible answers the patient might want to give.\n"
        "Rules:\n"
        "- Each answer must be a brief phrase (2-8 words).\n"
        "- All 4 answers MUST be meaningfully different from each other.\n"
        "- Cover a spread: one positive, one negative, one practical, one emotional.\n"
        "- No two answers should convey the same sentiment or meaning.\n"
        "Return ONLY a JSON array of exactly 4 strings. No markdown, no explanation."
    )
    try:
        completion = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You respond with raw JSON only. No markdown fences, no explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
        )
        raw = completion.choices[0].message.content.strip()
        print(f"[generate-options] Raw GPT response: {raw}")
        options = _extract_json_array(raw)
        options = [str(o) for o in options]
        while len(options) < 4:
            options.append("(no response)")
        return OptionsResponse(options=options[:4])
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/expand-response", response_model=ExpandResponse)
def expand_response(req: ExpandRequest):
    history_block = ""
    if req.history:
        pairs = req.history[-5:]
        lines = [f"  Q: \"{p.question}\" → A: \"{p.answer}\"" for p in pairs]
        history_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

    identity_block = _build_identity_block(
        req.linguistic_profile_summary, req.exemplars,
    )

    prompt = (
        f"{history_block}"
        f"{identity_block}"
        f"A caregiver asked: \"{req.question}\"\n"
        f"The ALS patient selected this short answer: \"{req.response}\"\n\n"
        "Turn the patient's short answer into a natural, complete sentence that answers the question. "
        "Keep it brief (1-2 sentences). Speak from the patient's perspective (first person). "
        "Match the patient's voice if a communication model is provided above."
    )
    try:
        completion = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        text = completion.choices[0].message.content.strip()
        print(f"[expand-response] Raw GPT response: {text}")
        return ExpandResponse(expanded=text)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/analyze-style", response_model=StyleAnalysisResponse)
def analyze_style(req: StyleAnalysisRequest):
    texts_block = "\n---\n".join(req.sample_texts[:50])
    prompt = (
        "Analyze the following writing samples from a single person. "
        "Characterize their communication style.\n\n"
        f"Samples:\n{texts_block}\n\n"
        "Respond with a JSON object containing exactly these fields:\n"
        '- "humor_style": describe their humor style (e.g. "dry", "sarcastic", "self-deprecating", "none")\n'
        '- "tone_description": describe their overall tone in 3-6 words\n'
        '- "emotional_valence": do they tend toward "positive", "negative", or "neutral" framing?\n'
        '- "personality_notes": 1-2 sentences about notable personality traits in their writing\n'
        "Return ONLY the JSON object. No markdown, no explanation."
    )
    try:
        completion = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You respond with raw JSON only. No markdown fences, no explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        raw = completion.choices[0].message.content.strip()
        print(f"[analyze-style] Raw GPT response: {raw}")
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        return StyleAnalysisResponse(
            humor_style=data.get("humor_style", "unknown"),
            tone_description=data.get("tone_description", "unknown"),
            emotional_valence=data.get("emotional_valence", "neutral"),
            personality_notes=data.get("personality_notes", ""),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/analyze-preferences", response_model=PreferenceAnalysisResponse)
def analyze_preferences(req: PreferenceAnalysisRequest):
    log_lines = []
    for entry in req.interactions[-100:]:
        selected = entry.get("selected") or "(none — all rejected)"
        rejected = ", ".join(f'"{r}"' for r in entry.get("rejected", []))
        log_lines.append(
            f"Q: \"{entry.get('question', '')}\"\n"
            f"  Selected: \"{selected}\"\n"
            f"  Rejected: [{rejected}]"
        )

    log_block = "\n\n".join(log_lines)
    prompt = (
        "Here is a log of an ALS patient's response selections and rejections "
        "in a communication aid. The most recent interactions are at the end — "
        "prioritize recent patterns over older ones.\n\n"
        f"{log_block}\n\n"
        "Analyze the patterns in what they reject vs. what they select. "
        "Extract 3-8 concrete preference rules. Each rule should describe a "
        "pattern, not a specific response.\n\n"
        "Examples of good rules:\n"
        '- "Avoids sentimental or emotionally effusive language"\n'
        '- "Prefers responses under 5 words"\n'
        '- "Never selects responses that minimize or dismiss the question"\n'
        '- "Tends to pick the most direct/practical option"\n\n'
        "Return ONLY a JSON array of rule strings. No markdown, no explanation."
    )
    try:
        completion = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You respond with raw JSON only. No markdown fences, no explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        raw = completion.choices[0].message.content.strip()
        print(f"[analyze-preferences] Raw GPT response: {raw}")
        rules = _extract_json_array(raw)
        rules = [str(r) for r in rules][:8]
        return PreferenceAnalysisResponse(rules=rules)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=str(e))
