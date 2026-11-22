"""Pluggable Reviewer for Shorts Scripts and Frames (Ant Ling + Gemini).

Features:
- Pluggable provider: 'antling' (OpenAI-compatible, default), 'gemini' (fallback), or 'both'.
- Automatic fallback: if Ant Ling encounters an error, timeout, rate-limit, or quota limit,
  it logs and falls back to Gemini without failing the run.
- Spend guard: Hard cap of 4 review calls per video / run.
- Token spend tracking: logs every review call with token counts and estimated costs to history/reviews.jsonl.
- Frame resizing: downscales frames to max 720px on the long side before base64 encoding.
- Strict JSON output: {"pass": bool, "problems": [...], "worst_sentence": "...", "fix_suggestion": "..."}.
- Never leaks API keys, tokens or personal info in prompts.
"""

import base64
import datetime
import io
import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

try:
    from .config import get_gemini_key
    from .log import log
except Exception:
    try:
        from config import get_gemini_key
        from log import log
    except Exception:
        def get_gemini_key():
            return os.environ.get("GEMINI_API_KEY", "")
        def log(msg):
            print(f"[Reviewer] {msg}", file=sys.stderr)

# ── Config & Models ──────────────────────────────────────────────────────────

ANTLING_BASE_URL = os.environ.get("ANTLING_BASE_URL", "https://api.ant-ling.com/v1").rstrip("/")
ANTLING_TEXT_MODEL = os.environ.get("ANTLING_TEXT_MODEL", "Ling-3.1-flash")
ANTLING_VISION_MODEL = os.environ.get("ANTLING_VISION_MODEL", "Ling-3.2-flash")

# Default REVIEWER=antling for scripts and frames, with gemini as automatic fallback
REVIEWER = os.environ.get("REVIEWER", "antling").strip().lower()

# Pricing constants in CNY / 1M tokens
# Ling-3.1-flash promo is 0.50 in / 1.50 out (standard is 2.00 / 6.00)
# Ling-3.2-flash is 2.00 in / 6.00 out
PRICE_CNY_PER_M = {
    "Ling-3.1-flash": {"input": 0.50, "output": 1.50},  # current promo rate
    "Ling-3.1-flash-standard": {"input": 2.00, "output": 6.00},
    "Ling-3.2-flash": {"input": 2.00, "output": 6.00},
    "gemini": {"input": 0.54, "output": 2.16},  # approx CNY equivalent for Flash Lite
}
CNY_TO_USD = 0.14

MAX_CALLS_PER_VIDEO = 4
_call_counter = 0


def reset_review_call_count():
    global _call_counter
    _call_counter = 0


def get_review_call_count() -> int:
    return _call_counter


def _get_antling_key() -> str:
    key = os.environ.get("ANTLING_API_KEY", "").strip()
    if not key:
        cfg_file = pathlib.Path.home() / ".verticals" / "config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                key = str(cfg.get("ANTLING_API_KEY", "")).strip()
            except Exception:
                pass
    return key


def _get_reviews_log_path() -> pathlib.Path:
    # Look for history/reviews.jsonl relative to cwd, my-shorts-bot, or ~/.verticals
    candidates = [
        pathlib.Path("history/reviews.jsonl"),
        pathlib.Path("my-shorts-bot/history/reviews.jsonl"),
        pathlib.Path(__file__).resolve().parent.parent / "my-shorts-bot" / "history" / "reviews.jsonl",
        pathlib.Path(__file__).resolve().parent / "history" / "reviews.jsonl",
        pathlib.Path.home() / ".verticals" / "reviews.jsonl",
    ]
    for c in candidates:
        if c.parent.exists():
            return c
    # Fallback to local history
    p = pathlib.Path("history/reviews.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _log_review_entry(entry: dict):
    path = _get_reviews_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"Failed to log review to {path}: {e}")


def _parse_json_verdict(raw: str) -> dict:
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    s, e = text.find("{"), text.rfind("}") + 1
    if s >= 0 and e > s:
        text = text[s:e]
    try:
        data = json.loads(text)
        return {
            "pass": bool(data.get("pass", False)),
            "problems": list(data.get("problems", [])),
            "worst_sentence": str(data.get("worst_sentence") or ""),
            "fix_suggestion": str(data.get("fix_suggestion") or ""),
        }
    except Exception as e:
        log(f"JSON parsing error from LLM reply: {e} -- raw: {text[:200]}")
        # If parsing fails, fall back to checking if text contains pass: true
        is_pass = '"pass": true' in text.lower() or '"pass":true' in text.lower()
        return {
            "pass": is_pass,
            "problems": ["Unparseable review verdict"] if not is_pass else [],
            "worst_sentence": "",
            "fix_suggestion": "Ensure LLM returns strict valid JSON",
        }


# ── Frame Processing ─────────────────────────────────────────────────────────

def resize_frame_to_720_base64(frame_path: pathlib.Path | str) -> str:
    """Resize image to max 720px on the long side and return base64 JPEG string."""
    p = pathlib.Path(frame_path)
    with Image.open(p) as img:
        img = img.convert("RGB")
        w, h = img.size
        long_side = max(w, h)
        if long_side > 720:
            scale = 720.0 / long_side
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── API Calls ────────────────────────────────────────────────────────────────

def _call_antling_chat(messages: list, model: str, max_tokens: int = 800) -> Tuple[Optional[str], dict]:
    """Call Ant Ling via OpenAI-compatible endpoint. Returns (content, usage_dict)."""
    api_key = _get_antling_key()
    if not api_key:
        raise ValueError("ANTLING_API_KEY not configured")

    url = f"{ANTLING_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=(10, 45))
    if r.status_code == 402 or "quota" in r.text.lower() or "billing" in r.text.lower():
        log(f"Ant Ling quota/billing error HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError(f"Ant Ling quota/billing error: {r.text[:200]}")
    if r.status_code != 200:
        raise RuntimeError(f"Ant Ling API error {r.status_code}: {r.text[:200]}")

    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return content, usage


def _call_gemini_chat(parts: list, model_kind: str = "text", max_tokens: int = 800) -> Tuple[Optional[str], dict]:
    """Call Gemini via Google API with fallback picking."""
    try:
        from .gemini_model import _config, pick_model, REQUEST_TIMEOUT, API
    except Exception:
        from gemini_model import _config, pick_model, REQUEST_TIMEOUT, API

    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not configured")

    model = pick_model(model_kind)
    url = f"{API}/models/{model}:generateContent"
    cfg = _config(model, max_tokens, 0.1)
    r = requests.post(url, json={"contents": [{"parts": parts}], "generationConfig": cfg},
                      headers={"x-goog-api-key": key}, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:200]}")

    data = r.json()
    candidates = data.get("candidates") or [{}]
    out_parts = candidates[0].get("content", {}).get("parts", [])
    text = " ".join(p.get("text", "") for p in out_parts if not p.get("thought")).strip()
    usage_meta = data.get("usageMetadata", {})
    usage = {
        "prompt_tokens": usage_meta.get("promptTokenCount", 0),
        "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
        "total_tokens": usage_meta.get("totalTokenCount", 0),
    }
    return text, usage


# ── Review Execution Engine ──────────────────────────────────────────────────

def _execute_review(
    stage: str,
    prompt: str,
    images_b64: Optional[List[str]] = None,
) -> dict:
    """Execute review with provider routing, spend guard, and token logging."""
    global _call_counter

    if _call_counter >= MAX_CALLS_PER_VIDEO:
        log(f"Hard cap of {MAX_CALLS_PER_VIDEO} reviews reached for this video. Skipping review.")
        return {
            "pass": True,
            "problems": [],
            "worst_sentence": "",
            "fix_suggestion": "Review skipped: hard cap reached",
            "skipped": True,
        }

    provider_choice = REVIEWER
    log(f"Running {stage} review via provider '{provider_choice}' (call {_call_counter + 1}/{MAX_CALLS_PER_VIDEO})...")

    def try_antling():
        is_vision = bool(images_b64)
        model = ANTLING_VISION_MODEL if is_vision else ANTLING_TEXT_MODEL
        if not images_b64:
            messages = [
                {"role": "system", "content": "You are a strict, objective quality auditor for 40-second viral YouTube Shorts. Respond only in strict JSON."},
                {"role": "user", "content": prompt},
            ]
        else:
            content_items = [{"type": "text", "text": prompt}]
            for b64 in images_b64:
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            messages = [{"role": "user", "content": content_items}]
        raw, usage = _call_antling_chat(messages, model=model)
        return raw, usage, model

    def try_gemini():
        is_vision = bool(images_b64)
        parts = [{"text": prompt}]
        if images_b64:
            for b64 in images_b64:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64
                    }
                })
        raw, usage = _call_gemini_chat(parts, model_kind="vision" if is_vision else "text")
        return raw, usage, "gemini"

    raw_response = None
    used_provider = None
    used_model = None
    token_usage = {}

    providers_to_try = []
    if provider_choice == "antling":
        providers_to_try = [("antling", try_antling), ("gemini", try_gemini)]
    elif provider_choice == "gemini":
        providers_to_try = [("gemini", try_gemini), ("antling", try_antling)]
    elif provider_choice == "both":
        providers_to_try = [("antling", try_antling), ("gemini", try_gemini)]
    else:
        providers_to_try = [("antling", try_antling), ("gemini", try_gemini)]

    results = []
    for prov_name, prov_func in providers_to_try:
        try:
            _call_counter += 1
            raw_response, token_usage, used_model = prov_func()
            used_provider = prov_name
            log(f"Review call succeeded using {used_provider} ({used_model}). Tokens: {token_usage}")
            results.append((prov_name, raw_response, token_usage, used_model))
            if provider_choice != "both":
                break
        except Exception as e:
            log(f"Provider '{prov_name}' review call failed or hit quota: {e} -- proceeding to next provider/step")
            if "quota" in str(e).lower() or "billing" in str(e).lower():
                # On quota/billing error, log and skip gracefully
                continue

    if not results:
        log("All review providers failed or are unavailable. Continuing without failing the run.")
        return {
            "pass": True,
            "problems": [],
            "worst_sentence": "",
            "fix_suggestion": "Review unavailable; run continued safely",
            "skipped": True,
        }

    # Aggregate verdict (if 'both', both must pass)
    final_verdict = {"pass": True, "problems": [], "worst_sentence": "", "fix_suggestion": ""}
    for prov_name, raw_resp, usage, model_name in results:
        v = _parse_json_verdict(raw_resp)
        p_in = usage.get("prompt_tokens", 0)
        p_out = usage.get("completion_tokens", 0)
        p_tot = usage.get("total_tokens", p_in + p_out)

        rates = PRICE_CNY_PER_M.get(model_name, PRICE_CNY_PER_M.get(prov_name, {"input": 1.0, "output": 2.0}))
        cost_cny = round((p_in * rates["input"] + p_out * rates["output"]) / 1_000_000, 6)
        cost_usd = round(cost_cny * CNY_TO_USD, 6)

        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stage": stage,
            "provider": prov_name,
            "model": model_name,
            "pass": v["pass"],
            "problems": v["problems"],
            "worst_sentence": v["worst_sentence"],
            "fix_suggestion": v["fix_suggestion"],
            "prompt_tokens": p_in,
            "completion_tokens": p_out,
            "total_tokens": p_tot,
            "cost_cny": cost_cny,
            "cost_usd": cost_usd,
        }
        _log_review_entry(entry)

        if not v["pass"]:
            final_verdict["pass"] = False
            final_verdict["problems"].extend(v["problems"])
            if not final_verdict["worst_sentence"] and v["worst_sentence"]:
                final_verdict["worst_sentence"] = v["worst_sentence"]
            if not final_verdict["fix_suggestion"] and v["fix_suggestion"]:
                final_verdict["fix_suggestion"] = v["fix_suggestion"]

    final_verdict["problems"] = list(dict.fromkeys(final_verdict["problems"]))
    return final_verdict


# ── Public Review Functions ──────────────────────────────────────────────────

def review_script(script: str, topic: str, sources: Optional[List[Dict[str, Any]]] = None) -> dict:
    """Run strict script review before TTS.

    Checks:
    1. Subject named explicitly in the first two sentences.
    2. Every proper noun and piece of jargon explained.
    3. A viewer who has never heard of the topic can follow it.
    4. The hook names the subject.
    5. Every number appears in the sources.
    6. The closing question is specific, not a reused generic line.
    """
    sources_summary = "\n".join(
        f"- {s.get('title', '')}: {s.get('summary', '')}" for s in (sources or [])
    ) or "(No separate sources provided; topic headline is the source)"

    prompt = f"""You are a strict, objective second-opinion quality reviewer auditing a 40-second faceless YouTube Short script before voiceover generation.
The script must be completely self-contained for a casual viewer who has NEVER heard of this topic before.

TOPIC:
\"\"\"{topic}\"\"\"

FETCHED SOURCE HEADLINES / CONTEXT:
\"\"\"{sources_summary}\"\"\"

SCRIPT TO AUDIT:
\"\"\"{script}\"\"\"

AUDIT CHECKS (ALL MUST PASS):
1. SUBJECT NAMED IN FIRST TWO SENTENCES: The specific subject (the show, place, landmark, company, event) MUST be named by its actual name in the first two sentences.
2. HOOK NAMES SUBJECT: The opening hook sentence MUST name the subject directly. Vague hooks like 'This game-changing twist' or 'This landmark' without naming the subject FAIL.
3. FOLLOWABLE BY FIRST-TIME VIEWER: Anyone who has never watched the show, visited the place, or heard of the topic must follow it effortlessly.
4. PROPER NOUNS & JARGON EXPLAINED: Every proper noun, jargon term, or format rule is immediately explained in a clear clause.
5. NUMBERS APPEAR IN SOURCES: Any specific statistic, year, measurement, or dollar figure in the script must be grounded in the provided sources/topic.
6. CLOSING QUESTION IS SPECIFIC: The ending question MUST be specific to this exact story, NOT a generic reused placeholder like 'What do you think? Comment below.'

You must output ONLY valid JSON matching this exact structure:
{{
  "pass": true or false,
  "problems": ["list of specific failed checks or issues found"],
  "worst_sentence": "the single most confusing, vague, or flawed sentence from the script, or empty string if pass is true",
  "fix_suggestion": "concrete instruction on how to fix the worst sentence and pass all checks"
}}
"""
    return _execute_review("script", prompt)


def review_frames(frame_paths: List[pathlib.Path | str], script: str, topic: str) -> dict:
    """Run strict visual frame review after rendering and before upload.

    Sends 6 frames spread across the video duration (resized to max 720px on the long side).
    Checks:
    1. Subject visible in the first 2 seconds (frame 1).
    2. Each frame matches what is said at that moment in the script.
    3. No cut-off or overlapping text.
    4. No logos, brand art or real identifiable faces.
    """
    frames_b64 = []
    for fp in frame_paths[:6]:
        try:
            frames_b64.append(resize_frame_to_720_base64(fp))
        except Exception as e:
            log(f"Failed to process frame {fp}: {e}")

    if not frames_b64:
        log("No valid frames could be encoded for review. Bypassing frame review.")
        return {"pass": True, "problems": [], "worst_sentence": "", "fix_suggestion": "No frames", "skipped": True}

    prompt = f"""You are a strict, objective visual quality reviewer auditing 6 sampled frames from a finished 40-second vertical YouTube Short before upload.
The frames are sampled in chronological order from beginning to end.

TOPIC: \"\"\"{topic}\"\"\"
SCRIPT: \"\"\"{script}\"\"\"

AUDIT CHECKS (ALL MUST PASS):
1. SUBJECT VISIBLE IN FIRST 2 SECONDS: Frame 1 must show the actual subject or a clear visual representation of it (not generic filler or an unrelated scene).
2. SCRIPT-FRAME ALIGNMENT: Each frame must accurately match the narration content occurring at that moment in the story.
3. NO CUT-OFF OR OVERLAPPING TEXT: Any on-screen text, captions, or stickers must be clean, readable, not cut off by vertical screen bounds (9:16), and not overlapping other text.
4. NO LOGOS, BRAND ART OR IDENTIFIABLE REAL FACES: No copyrighted watermarks, TV channel bugs, commercial brand art, or recognizable real people's faces (to avoid copyright/impersonation flags).

You must output ONLY valid JSON matching this exact structure:
{{
  "pass": true or false,
  "problems": ["list of visual issues found, referencing frame numbers 1-6"],
  "worst_sentence": "the script sentence corresponding to the worst frame, or empty string",
  "fix_suggestion": "what visual correction or replacement is needed"
}}
"""
    return _execute_review("frame", prompt, images_b64=frames_b64)
