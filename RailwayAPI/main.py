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

class OptionsResponse(BaseModel):
    options: list[str]

class ExpandRequest(BaseModel):
    question: str
    response: str
    history: list[QAPair] = []

class ExpandResponse(BaseModel):
    expanded: str


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

    prompt = (
        f"{history_block}"
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

    prompt = (
        f"{history_block}"
        f"A caregiver asked: \"{req.question}\"\n"
        f"The ALS patient selected this short answer: \"{req.response}\"\n\n"
        "Turn the patient's short answer into a natural, complete sentence that answers the question. "
        "Keep it brief (1-2 sentences). Speak from the patient's perspective (first person)."
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
