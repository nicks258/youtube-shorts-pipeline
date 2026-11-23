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
        import os, pathlib, json
        def get_gemini_key():
            k = os.environ.get("GEMINI_API_KEY", "")
            if not k:
                try:
                    cfg_path = pathlib.Path.home() / ".verticals" / "config.json"
                    if cfg_path.exists():
                        k = json.loads(cfg_path.read_text()).get("GEMINI_API_KEY", "")
                except Exception:
                    pass
            return k
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
    "gemini-decision-1": {"input": 0.714, "output": 0.0},  # $0.10 / 1M input tokens ($0 output)
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

def _call_antling_chat(messages: list, model: str, max_tokens: int = 4000) -> Tuple[Optional[str], dict]:
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
    msg = data.get("choices", [{}])[0].get("message", {})
    content = msg.get("content", "") or ""
    if not content.strip() and msg.get("reasoning_content"):
        content = msg.get("reasoning_content")
    usage = data.get("usage", {})
    return content, usage


def _call_gemini_chat(parts: list, model_kind: str = "text", max_tokens: int = 2000) -> Tuple[Optional[str], dict]:
    """Call Gemini via Google API with fallback picking."""
    try:
        from .gemini_model import generate
    except Exception:
        from gemini_model import generate

    text = generate(parts, max_tokens=max_tokens, temperature=0.1, kind=model_kind)
    return text, {}


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
            log(f"Review call completed using {used_provider} ({used_model}). Tokens: {token_usage}")

            # If response is empty or whitespace, fail over to the next provider
            if not raw_response or not raw_response.strip():
                log(f"Provider '{prov_name}' returned empty response content. Falling back to next reviewer...")
                continue

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


# ── Google Decisions API Reviewer ─────────────────────────────────────────────

def _get_genai_client():
    """Create a Google GenAI Client configured for Decisions API."""
    from google import genai
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY not configured")
    return genai.Client(api_key=key, http_options={"api_version": "v1beta"})


def _review_script_decisions(
    script: str,
    topic: str,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    """Evaluate script quality checks using Google's Decisions API (gemini-decision-1).

    All checks evaluated in a single request against a shared context:
    - 5 noul questions:
      1. subject named in first two sentences
      2. every proper noun and jargon term explained
      3. hook names the subject
      4. closing question is specific not generic
      5. every number appears in the provided sources
    - 1 score question:
      clarity score ("How well could someone who has never heard of this topic follow this script?")
      with 4 levels from "completely lost" (0) to "fully clear" (3).

    Branching logic:
    - Fail if any noul < 0.5 OR clarity score < level 2.
    - Where confidence < 0.6 on failing check, treat as soft fail.
    """
    client = _get_genai_client()

    sources_summary = "\n".join(
        f"- {s.get('title', '')}: {s.get('summary', '')}" for s in (sources or [])
    ) or "(No separate sources provided; topic headline is the source)"

    shared_input = f"Topic: {topic}\nSource Headlines: {sources_summary}\nScript: {script}"

    questions = {
        "subject_named_in_first_two": {
            "type": "noul",
            "instructions": "Is the specific subject (the show, place, landmark, company, or person) explicitly named by its actual specific name in the first two sentences?",
        },
        "every_proper_noun_and_jargon_explained": {
            "type": "noul",
            "instructions": "Is every proper noun, jargon term, or format rule clearly explained for a viewer who has never heard of this topic?",
        },
        "hook_names_subject": {
            "type": "noul",
            "instructions": "Does the opening hook sentence explicitly name the specific subject by name (e.g. not a vague pronoun or generic placeholder like 'This twist' or 'A bridge')?",
        },
        "closing_question_specific": {
            "type": "noul",
            "instructions": "Is the closing question specific to this exact story rather than a generic reused placeholder like 'What do you think? Comment below.'?",
        },
        "numbers_in_sources": {
            "type": "noul",
            "instructions": "Does every number, date, measurement, or statistic in the script appear in the provided sources or topic?",
        },
        "clarity_score": {
            "type": "score",
            "instructions": "How well could someone who has never heard of this topic follow this script?",
            "criteria": ["completely lost", "somewhat confused", "mostly clear", "fully clear"],
        },
    }

    resp = client.decisions.create(
        model="gemini-decision-1",
        input=shared_input,
        questions=questions,
    )

    checks = {}
    problems = []
    is_pass = True
    is_soft_fail = False
    min_confidence = 1.0

    noul_labels = {
        "subject_named_in_first_two": "Subject named in the first two sentences",
        "every_proper_noun_and_jargon_explained": "Every proper noun and jargon term explained",
        "hook_names_subject": "Hook names the subject",
        "closing_question_specific": "Closing question is specific not generic",
        "numbers_in_sources": "Every number appears in the provided sources",
    }

    for q_key, label in noul_labels.items():
        ans = resp.answers.get(q_key)
        if ans:
            prob = float(ans.noul)
            conf = float(ans.confidence)
            chk_pass = prob >= 0.5
            checks[q_key] = {
                "name": label,
                "probability": prob,
                "confidence": conf,
                "pass": chk_pass,
            }
            if conf < min_confidence:
                min_confidence = conf
            if not chk_pass:
                is_pass = False
                problems.append(f"{label} failed (probability: {prob:.4f}, confidence: {conf:.4f})")
                if conf < 0.6:
                    is_soft_fail = True

    clarity_ans = resp.answers.get("clarity_score")
    clarity_score = 0.0
    clarity_conf = 0.0
    if clarity_ans:
        clarity_score = float(clarity_ans.score)
        clarity_conf = float(clarity_ans.confidence)
        clarity_probs = dict(clarity_ans.probabilities or {})
        prob_lost_or_confused = float(clarity_probs.get("0", 0.0) + clarity_probs.get("1", 0.0))

        # Pass requires score >= 2.5 AND combined probability of levels 0 & 1 <= 0.35
        clarity_pass = (clarity_score >= 2.5) and (prob_lost_or_confused <= 0.35)
        checks["clarity_score"] = {
            "name": "Clarity score (first-time viewer comprehension)",
            "score": clarity_score,
            "confidence": clarity_conf,
            "pass": clarity_pass,
            "prob_levels_0_1": prob_lost_or_confused,
            "probabilities": clarity_probs,
            "legend": dict(clarity_ans.legend or {}),
        }

        if not clarity_pass:
            is_pass = False
            if clarity_score < 2.5:
                problems.append(f"Clarity score below threshold 2.5 (score: {clarity_score:.2f}/3.0)")
            if prob_lost_or_confused > 0.35:
                problems.append(f"High viewer confusion risk (combined level 0+1 probability: {prob_lost_or_confused:.2f} > 0.35)")
            # Note: Do NOT set is_soft_fail based on score confidence.
            # Confidence on score question represents the top level's probability (~0.35-0.50).
            # Soft fail (< 0.6) applies exclusively to noul checks.

    # Parse sentences for worst_sentence and fix_suggestion
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script) if s.strip()]
    worst_sentence = ""
    fix_suggestions = []

    if not is_pass:
        if not checks.get("hook_names_subject", {}).get("pass", True) and sentences:
            worst_sentence = sentences[0]
            fix_suggestions.append("Rewrite the hook to explicitly name the subject.")
        elif not checks.get("closing_question_specific", {}).get("pass", True) and sentences:
            worst_sentence = sentences[-1]
            fix_suggestions.append("Replace the generic closing question with a specific question about this topic.")
        elif not checks.get("subject_named_in_first_two", {}).get("pass", True) and len(sentences) > 1:
            worst_sentence = sentences[1]
            fix_suggestions.append("Name the subject explicitly in the first two sentences.")
        elif not checks.get("every_proper_noun_and_jargon_explained", {}).get("pass", True):
            worst_sentence = sentences[0] if sentences else ""
            fix_suggestions.append("Define proper nouns and jargon terms in a clear clause.")
        elif not checks.get("clarity_score", {}).get("pass", True):
            worst_sentence = sentences[0] if sentences else ""
            fix_suggestions.append("Clarify the story narrative so a first-time viewer can follow effortlessly.")

    fix_suggestion = " ".join(fix_suggestions) if fix_suggestions else ""

    usage = resp.usage
    p_in = usage.total_input_tokens if usage and usage.total_input_tokens else len(shared_input) // 4
    p_out = 0
    p_tot = p_in
    cost_usd = round(p_in * (0.10 / 1_000_000), 6)
    cost_cny = round(cost_usd * 7.14, 6)

    verdict = {
        "pass": is_pass,
        "soft_fail": is_soft_fail,
        "problems": problems,
        "worst_sentence": worst_sentence,
        "fix_suggestion": fix_suggestion,
        "checks": checks,
        "clarity_score": clarity_score,
        "confidence": min_confidence,
        "prompt_tokens": p_in,
        "completion_tokens": p_out,
        "total_tokens": p_tot,
        "cost_usd": cost_usd,
        "cost_cny": cost_cny,
        "provider": "google-decisions",
        "model": "gemini-decision-1",
    }

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": "script",
        "provider": "google-decisions",
        "model": "gemini-decision-1",
        "pass": is_pass,
        "soft_fail": is_soft_fail,
        "clarity_score": clarity_score,
        "clarity_confidence": clarity_conf,
        "checks": checks,
        "problems": problems,
        "worst_sentence": worst_sentence,
        "fix_suggestion": fix_suggestion,
        "prompt_tokens": p_in,
        "completion_tokens": 0,
        "total_tokens": p_tot,
        "cost_cny": cost_cny,
        "cost_usd": cost_usd,
    }
    _log_review_entry(entry)

    return verdict


# ── Public Review Functions ──────────────────────────────────────────────────

def review_script(script: str, topic: str, sources: Optional[List[Dict[str, Any]]] = None) -> dict:
    """Run strict script review before TTS.

    Primary: Google's Decisions API (gemini-decision-1) evaluating all 5 noul checks
    plus clarity scoring in a single request.
    Fallback: Ling-3.2-flash / Ling-3.1-flash / Gemini chat if Decisions errors.
    Never fails a run because a reviewer is unavailable.
    """
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

    # Primary reviewer: Google's Decisions API (gemini-decision-1)
    try:
        _call_counter += 1
        log(f"Running script review via Google Decisions API (call {_call_counter}/{MAX_CALLS_PER_VIDEO})...")
        verdict = _review_script_decisions(script, topic, sources)
        log(f"Google Decisions review finished: pass={verdict['pass']}, clarity={verdict.get('clarity_score', 0.0):.2f}, soft_fail={verdict.get('soft_fail', False)}")
        return verdict
    except Exception as e:
        log(f"Google Decisions API review failed or unavailable ({e}). Falling back to Ling-3.2-flash/Gemini review.")

    # Fallback to existing LLM JSON review
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
