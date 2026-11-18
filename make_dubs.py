"""Make dubbed audio tracks for the Short that was just produced.

For every language enabled in languages.json this:
  1. translates the script with Gemini Flash Lite, aiming at a spoken length that
     matches the English audio,
  2. speaks it with Kokoro in that language's voice,
  3. fits the result to the video's exact duration (YouTube requires each audio
     track to be roughly the same length as the video),
  4. writes out/dubs/<code>.m4a plus translated title/description.

YouTube has no API for multi-language audio, so the files are added by hand in
YouTube Studio -> Languages. out/dubs/INSTRUCTIONS.txt lists the steps and the
translated metadata for each language.

Env: GEMINI_API_KEY, TOPIC_CATEGORY (optional), DUB_LANGUAGES (optional
comma-separated override, e.g. "hi,es"), SKIP_DUBS=1 to do nothing.
Usage: python make_dubs.py [path to the video]
"""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if (ROOT / "pipeline").exists():
    sys.path.insert(0, str(ROOT / "pipeline"))
if (ROOT.parent / "verticals").exists():
    sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

LANGS_FILE = ROOT / "languages.json"
DRAFTS = pathlib.Path.home() / ".verticals" / "drafts"
OUT = ROOT / "out" / "dubs"
WORK = ROOT / "out" / "_dubwork"

TEMPO_MIN, TEMPO_MAX = 0.86, 1.18   # beyond this a stretched voice sounds wrong
TOLERANCE = 0.25                     # seconds we accept without stretching
MAX_TRANSLATION_TRIES = 2


def log(msg: str):
    print(f"  [dubs] {msg}", file=sys.stderr)


# ---------------------------------------------------------------- audio helpers

def duration_of(path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def atempo_chain(factor: float) -> str:
    """ffmpeg's atempo takes 0.5-2.0 per filter; chain them for anything outside that."""
    parts, f = [], factor
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        parts.append("atempo=0.5")
        f /= 0.5
    parts.append(f"atempo={f:.6f}")
    return ",".join(parts)


def fit_plan(duration: float, target: float) -> dict:
    """How to make `duration` seconds of speech fill exactly `target` seconds."""
    if target <= 0:
        return {"tempo": 1.0, "pad": 0.0, "ok": False, "note": "no target duration"}
    if abs(duration - target) <= TOLERANCE:
        return {"tempo": 1.0, "pad": max(0.0, target - duration), "ok": True, "note": "already matches"}
    wanted = duration / target                      # >1 means speak faster
    tempo = min(TEMPO_MAX, max(TEMPO_MIN, wanted))
    after = duration / tempo
    pad = max(0.0, target - after)
    over = after - target
    if over > TOLERANCE:
        return {"tempo": tempo, "pad": 0.0, "ok": False,
                "note": f"still {over:.1f}s too long at max speed; translation needs to be shorter"}
    return {"tempo": tempo, "pad": pad, "ok": True,
            "note": f"speed x{tempo:.3f}" + (f", {pad:.1f}s silence added" if pad > 0.05 else "")}


def render_track(src: pathlib.Path, dest: pathlib.Path, plan: dict, target: float):
    """Apply the fit plan and write an AAC audio-only file of exactly `target` seconds."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if abs(plan["tempo"] - 1.0) > 0.001:
        filters.append(atempo_chain(plan["tempo"]))
    if plan["pad"] > 0.01:
        filters.append(f"apad=pad_dur={plan['pad'] + 1:.3f}")
    else:
        filters.append("apad=pad_dur=1")            # pad then trim: guarantees the exact length
    filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-af", ",".join(filters), "-t", f"{target:.3f}",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "1", str(dest)], check=True)


# ---------------------------------------------------------------- Gemini

def gemini(prompt: str, max_tokens: int = 2000) -> str:
    try:
        from verticals.gemini_model import generate
    except ImportError:
        from gemini_model import generate
    # kind="vision" selects a Flash Lite model: 500 free requests a day
    return generate([{"text": prompt}], max_tokens=max_tokens, temperature=0.3, kind="vision")


def translate_script(script: str, lang: dict, target_words: int) -> str:
    reply = gemini(f"""Translate this YouTube Shorts narration into {lang['name']} ({lang['native_name']}).

Rules:
- Natural spoken {lang['name']}, the way a narrator would say it, not a literal translation.
- Keep the same facts, numbers and order. Keep the hook punchy.
- Aim for about {target_words} words so it takes the same time to say as the English.
- Write numbers as words where a speaker would say them.
- No emoji, no hashtags, no notes, no quotation marks around the whole text.

Narration:
{script}

Reply with ONLY the translated narration.""")
    return " ".join(reply.strip().strip('"').split())


def translate_meta(title: str, description: str, lang: dict) -> dict:
    reply = gemini(f"""Translate this YouTube Short's title and description into {lang['name']}.
Keep hashtags in English but translate the words around them. Keep the title under 90 characters.

TITLE: {title}

DESCRIPTION:
{description}

Reply with ONLY JSON: {{"title": "...", "description": "..."}}""")
    try:
        m = re.search(r"\{.*\}", reply, re.S)
        data = json.loads(m.group(0))
        return {"title": str(data.get("title", ""))[:100], "description": str(data.get("description", ""))}
    except Exception as e:
        log(f"{lang['name']}: could not translate the title/description ({e})")
        return {"title": "", "description": ""}


# ---------------------------------------------------------------- speech

def speak(script: str, lang: dict, work: pathlib.Path) -> pathlib.Path:
    """Use the pipeline's Kokoro helper with this language's voice."""
    try:
        from kokoro_tts import generate_kokoro
    except ImportError:
        from verticals.kokoro_tts import generate_kokoro
    work.mkdir(parents=True, exist_ok=True)
    before = dict(os.environ)
    os.environ["KOKORO_VOICE"] = lang["voice"]
    os.environ["KOKORO_LANG"] = lang["kokoro_lang"]
    try:
        return pathlib.Path(generate_kokoro(script, work, lang["code"]))
    finally:
        os.environ.clear()
        os.environ.update(before)


# ---------------------------------------------------------------- main

def enabled_languages(cfg: dict) -> list:
    override = [c.strip() for c in os.environ.get("DUB_LANGUAGES", "").split(",") if c.strip()]
    out = []
    for code, lang in cfg["languages"].items():
        if (code in override) if override else lang.get("enabled"):
            out.append({**lang, "code": code})
    return out


def newest_draft() -> dict:
    drafts = sorted(DRAFTS.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(drafts[-1].read_text(encoding="utf-8")) if drafts else {}


def find_video(argv: list) -> pathlib.Path | None:
    if len(argv) > 1 and pathlib.Path(argv[1]).exists():
        return pathlib.Path(argv[1])
    draft = newest_draft()
    for key in ("video_en", "video"):
        p = draft.get(key)
        if p and pathlib.Path(p).exists():
            return pathlib.Path(p)
    videos = sorted((ROOT / "out").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return videos[-1] if videos else None


def main():
    if os.environ.get("SKIP_DUBS") == "1":
        log("SKIP_DUBS=1, nothing to do")
        return
    if not LANGS_FILE.exists():
        log("languages.json not found, nothing to do")
        return
    cfg = json.loads(LANGS_FILE.read_text(encoding="utf-8"))
    langs = enabled_languages(cfg)
    if not langs:
        log("no languages enabled")
        return

    video = find_video(sys.argv)
    if not video:
        log("no finished video found, nothing to do")
        return
    target = duration_of(video)
    draft = newest_draft()
    script = " ".join(str(draft.get("script", "")).split())
    if not script:
        log("the draft has no script, nothing to do")
        return
    english_words = len(script.split())

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    results = []

    for lang in langs:
        log(f"{lang['name']} ({lang['voice']})...")
        entry = {"code": lang["code"], "name": lang["name"], "voice": lang["voice"],
                 "youtube_language": lang.get("youtube_language", lang["code"]), "ok": False}
        try:
            words = english_words
            for attempt in range(MAX_TRANSLATION_TRIES):
                text = translate_script(script, lang, words)
                raw = speak(text, lang, WORK / lang["code"])
                spoken = duration_of(raw)
                plan = fit_plan(spoken, target)
                if plan["ok"] or attempt == MAX_TRANSLATION_TRIES - 1:
                    break
                words = max(20, int(words * target / spoken * 0.95))
                log(f"  {lang['name']}: {spoken:.1f}s vs {target:.1f}s, retrying with ~{words} words")
            dest = OUT / f"{lang['code']}.m4a"
            render_track(raw, dest, plan, target)
            meta = translate_meta(draft.get("youtube_title", ""), draft.get("youtube_description", ""), lang)
            entry.update(ok=True, file=dest.name, script=text, seconds=round(duration_of(dest), 2),
                         fit=plan["note"], fit_clean=plan["ok"], **meta)
            log(f"  {lang['name']}: {entry['seconds']}s ({plan['note']})")
            if not plan["ok"]:
                log(f"  WARNING {lang['name']}: {plan['note']}")
        except Exception as e:
            entry["error"] = str(e)[:300]
            log(f"  {lang['name']} failed: {e}")
        results.append(entry)

    shutil.rmtree(WORK, ignore_errors=True)
    (OUT / "manifest.json").write_text(json.dumps(
        {"video": str(video), "video_seconds": round(target, 2),
         "title": draft.get("youtube_title", ""), "languages": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    done = [r for r in results if r["ok"]]
    lines = ["HOW TO ADD THESE AUDIO TRACKS", "",
             "YouTube has no API for multi-language audio, so this part is manual (about 1 minute per language).",
             "", "1. Post the Short as usual (its audio stays English).",
             "2. On a computer, open YouTube Studio -> Languages (left menu) and click this video.",
             "3. Next to Audio, click Add, pick the language, and upload the matching file below.",
             "4. If YouTube already made an automatic dub for that language, delete it first.",
             "5. Paste the translated title and description for that language on the same page.", "",
             f"Video: {video.name} ({target:.1f}s)", ""]
    for r in results:
        lines.append("-" * 60)
        if not r["ok"]:
            lines += [f"{r['name']}: FAILED ({r.get('error', 'unknown')})", ""]
            continue
        lines += [f"{r['name']} ({r['youtube_language']}) -> {r['file']}  [{r['seconds']}s, {r['fit']}]",
                  "" if r["fit_clean"] else "  WARNING: length does not match well; check before uploading.",
                  f"TITLE: {r.get('title', '')}", "", "DESCRIPTION:", r.get("description", ""), ""]
    (OUT / "INSTRUCTIONS.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"{len(done)}/{len(results)} languages ready in {OUT}")


if __name__ == "__main__":
    main()
