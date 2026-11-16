"""Pick a Gemini model that exists for this API key, and call it.

Google shuts old Gemini models down regularly (gemini-2.0-flash is gone), so
instead of hard-coding one name this asks the API which models the key can
use and picks the best free Flash model from a preference list.

Override with GEMINI_MODEL (text) or GEMINI_VISION_MODEL (image checks).
"""
import os
import time

import requests

from .config import get_gemini_key
from .log import log

API = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT = (10, 45)  # connect, read timeouts in seconds

TEXT_PREFS = [
    "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest",
    "gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest", "gemini-2.5-flash-lite",
]
# Shot planning and image checks are many small calls. Flash Lite models get
# 500 free requests/day (regular Flash only 20), so they go first here.
VISION_PREFS = [
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
] + TEXT_PREFS

_cache: dict = {}


def _available() -> set:
    if "models" in _cache:
        return _cache["models"]
    names: set = set()
    token = None
    try:
        for _ in range(10):
            params = {"pageSize": 1000}
            if token:
                params["pageToken"] = token
            r = requests.get(f"{API}/models", params=params,
                             headers={"x-goog-api-key": get_gemini_key()}, timeout=30)
            r.raise_for_status()
            data = r.json()
            for m in data.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    names.add(m["name"].split("/", 1)[-1])
            token = data.get("nextPageToken")
            if not token:
                break
    except Exception as e:
        log(f"Could not list Gemini models ({e})")
    _cache["models"] = names
    return names


def _ordered_models(kind: str) -> list[str]:
    """All usable models for this key, best first. Previews last: their free quota is tiny."""
    available = _available()
    prefs = VISION_PREFS if kind == "vision" else TEXT_PREFS
    ordered = [m for m in prefs if m in available]
    skip = ("image", "tts", "audio", "live", "embedding", "robotics", "omni", "computer-use", "pro")
    rest = [m for m in available if "flash" in m and m not in ordered and not any(x in m for x in skip)]
    lite_first = kind == "vision"
    rest.sort(key=lambda m: ("preview" in m or "exp" in m,
                             ("lite" not in m) if lite_first else ("lite" in m),
                             [-ord(ch) for ch in m]))
    ordered = list(dict.fromkeys(ordered + rest))
    return ordered or ["gemini-flash-latest"]


def pick_model(kind: str = "text") -> str:
    env = os.environ.get("GEMINI_VISION_MODEL" if kind == "vision" else "GEMINI_MODEL", "").strip()
    if env:
        return env
    exhausted = _cache.setdefault("exhausted", set())
    for model in _ordered_models(kind):
        if model not in exhausted:
            if _cache.get(kind) != model:
                log(f"Using Gemini model {model} for {kind}")
                _cache[kind] = model
            return model
    raise RuntimeError("Gemini free quota used up on every available model -- try again tomorrow")


def _config(model: str, max_tokens: int, temperature: float) -> dict:
    # newer models "think" first and that counts toward the output limit,
    # so leave extra room and ask for light thinking
    cfg = {"maxOutputTokens": max_tokens + 4096, "temperature": temperature}
    if model.startswith("gemini-2.5-flash"):
        cfg["thinkingConfig"] = {"thinkingBudget": 0}
    elif model.startswith("gemini-3"):
        cfg["thinkingConfig"] = {"thinkingLevel": "low"}
    return cfg


def _quota_info(r) -> tuple[bool, float]:
    """(is_daily_limit, seconds_to_wait) from a 429 response."""
    text = r.text or ""
    daily = "PerDay" in text or "per day" in text.lower() or "RequestsPerDay" in text
    wait = 0.0
    try:
        for d in r.json().get("error", {}).get("details", []):
            if "retryDelay" in d:
                wait = float(str(d["retryDelay"]).rstrip("s"))
    except Exception:
        pass
    return daily, wait


def generate(parts: list, max_tokens: int = 1500, temperature: float = 0.7, kind: str = "text") -> str:
    key = get_gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    exhausted = _cache.setdefault("exhausted", set())
    no_thinking = _cache.setdefault("no_thinking", set())
    retried_5xx: set[str] = set()
    waited = 0.0
    last = ""
    max_attempts = max(16, len(_ordered_models(kind)) * 2)
    for _ in range(max_attempts):
        try:
            model = pick_model(kind)  # raises when every model is skipped/exhausted
        except RuntimeError as e:
            if last and "Gemini API 429" not in last:
                raise RuntimeError(last) from e
            raise
        if model in exhausted:
            break
        cfg = _config(model, max_tokens, temperature)
        if model in no_thinking:
            cfg.pop("thinkingConfig", None)
        try:
            r = requests.post(f"{API}/models/{model}:generateContent",
                              json={"contents": [{"parts": parts}], "generationConfig": cfg},
                              headers={"x-goog-api-key": key}, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last = f"Gemini request failed ({model}): {e}"
            log(f"{model}: request failed ({type(e).__name__}) -- switching model")
            exhausted.add(model)
            continue
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                data = {}
            out = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            text = " ".join(p.get("text", "") for p in out if not p.get("thought")).strip()
            if text:
                return text
            last = f"empty response from {model}"
            exhausted.add(model)
            continue
        last = f"Gemini API {r.status_code} ({model}): {r.text[:200]}"
        if r.status_code == 400 and "thinkingConfig" in cfg:
            no_thinking.add(model)
            continue
        if r.status_code == 429:
            daily, wait = _quota_info(r)
            if daily or wait > 60 or waited > 90:
                log(f"{model}: free quota used up -- switching model")
                exhausted.add(model)
            else:
                wait = wait or 15
                waited += wait
                time.sleep(wait)
            continue
        if r.status_code in (404, 403):
            exhausted.add(model)  # shut down or not allowed for this key
            continue
        if r.status_code in (500, 503):
            if model not in retried_5xx and waited < 20:
                retried_5xx.add(model)
                waited += 5
                time.sleep(5)
                continue
            log(f"{model}: server error {r.status_code} -- switching model")
            exhausted.add(model)
            continue
        exhausted.add(model)
    raise RuntimeError(last or "Gemini call failed")