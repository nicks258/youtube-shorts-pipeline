"""Fast-paced stock-footage visuals for the Shorts pipeline.

Instead of 3 still images for the whole video, this splits the script into
short beats (about 2-3 seconds of speech each), finds a matching stock video
clip for every beat, and cuts between them in sync with the voiceover.

Stock source: Pixabay (PIXABAY_API_KEY, free) or Pexels (PEXELS_API_KEY).
For each beat, Gemini plans 3 searches (specific -> general), the results are
ranked by tag match, and Gemini looks at the thumbnails to pick one that really
shows what is being said (or rejects them all).
Fallback order: stock video -> stock photo (with slow zoom)
-> AI image of the exact scene (Pollinations, then Gemini) -> plain colour frame.

Without either key, the pipeline's original 3-image behaviour is used.
"""

import json
import os
import random
import re
from pathlib import Path
import subprocess

import requests
from PIL import Image

from . import broll as _broll
from .assemble import _ffmpeg_has_libass, get_audio_duration
from .assemble import assemble_video as _original_assemble
from .config import MEDIA_DIR, VIDEO_HEIGHT, VIDEO_WIDTH, get_gemini_key, run_cmd
from .log import log

W, H = VIDEO_WIDTH, VIDEO_HEIGHT
PLAN_FILE = "visual_plan.json"
WORDS_PER_SHOT = int(os.environ.get("WORDS_PER_SHOT", "7"))  # ~2.5 s of speech
PEXELS = "https://api.pexels.com"
STOPWORDS = set("""a an the and or but if then so of to in on at by for with from as is are was were be been
being it its this that these those you your we our they their he she his her i me my not no yes do does did
can could will would should just very really more most than too also into about over after before here there
what which who whom how why when where all any some every each one two three new now get got make makes""".split())


# ---------------------------------------------------------------- planning

def _split_script(script: str) -> list[str]:
    """Split the script into short beats that each get their own shot."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(script.split()))
    beats = []
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        parts = max(1, round(len(words) / WORDS_PER_SHOT))
        size = -(-len(words) // parts)  # ceiling division
        for i in range(0, len(words), size):
            beats.append(" ".join(words[i:i + size]))
    merged = []
    for beat in beats:
        if merged and len(beat.split()) < 3:
            merged[-1] += " " + beat
        else:
            merged.append(beat)
    return merged


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def _keyword_query(text: str) -> str:
    words = sorted(_words(text), key=len, reverse=True)
    return " ".join(words[:2]) or "technology"


# ---------------------------------------------------------------- motion scenes

MOTION_KIT_DIR = os.environ.get("MOTION_KIT_DIR", "").strip()
MOTION_SHARE = float(os.environ.get("MOTION_SHARE", "0.3"))  # max share of shots that are animated
BRAND_NAME = os.environ.get("CHANNEL_NAME", "TECH UNSEEN").upper()
MOTION_TYPES = ("stat", "keyword", "compare", "steps")


def _motion_enabled() -> bool:
    kit = Path(MOTION_KIT_DIR) if MOTION_KIT_DIR else None
    import shutil
    return bool(kit and (kit / "render.mjs").exists()
                and (kit / "node_modules" / "puppeteer-core").exists() and shutil.which("node"))


def _motion_prompt(n: int) -> str:
    most = max(1, round(n * MOTION_SHARE))
    return f"""
Some lines are better as an ANIMATED GRAPHIC than as footage: a number or statistic,
a comparison of two things, a short list or process, or one key term/concept that
is hard to film. For up to {most} such lines (never line 1, never two in a row),
add "motion" with ONE of these shapes (keep every text short):
  {{"type": "stat", "data": {{"value": "47%", "label": "max 8 words", "kicker": "1-2 words"}}}}
  {{"type": "keyword", "data": {{"word": "1-3 words", "sub": "max 7 words"}}}}
  {{"type": "compare", "data": {{"left": {{"title": "1-2 words", "detail": "max 6 words"}},
                                  "right": {{"title": "1-2 words", "detail": "max 6 words"}}}}}}
  {{"type": "steps", "data": {{"title": "1-3 words", "items": ["max 5 words", "...", "..."]}}}}
Use only facts and numbers that are stated in the narration; never invent numbers.
For every other line set "motion": null. Always fill "queries" and "ai_prompt" too.
"""


def _txt(v, limit: int) -> str:
    v = " ".join(str(v or "").split())
    return v if len(v) <= limit else v[:limit - 1].rstrip() + "…"


def _clean_motion(m) -> dict | None:
    """Validate a motion spec from the LLM; return a safe spec or None."""
    if not isinstance(m, dict) or m.get("type") not in MOTION_TYPES or not isinstance(m.get("data"), dict):
        return None
    t, d = m["type"], m["data"]
    if t == "stat":
        value = _txt(d.get("value"), 12)
        if not value or not any(ch.isdigit() for ch in value):
            return None
        data = {"value": value, "label": _txt(d.get("label"), 70), "kicker": _txt(d.get("kicker"), 18)}
    elif t == "keyword":
        word = _txt(d.get("word"), 28)
        if not word or len(word.split()) > 3:
            return None
        data = {"word": word, "sub": _txt(d.get("sub"), 60)}
    elif t == "compare":
        sides = {}
        for side in ("left", "right"):
            sd = d.get(side) if isinstance(d.get(side), dict) else {}
            title = _txt(sd.get("title"), 22)
            if not title:
                return None
            sides[side] = {"title": title, "detail": _txt(sd.get("detail"), 50)}
        data = sides
    else:  # steps
        items = [_txt(x, 40) for x in (d.get("items") or []) if str(x or "").strip()][:4]
        if len(items) < 2:
            return None
        data = {"title": _txt(d.get("title"), 24), "items": items}
    return {"type": t, "data": {k: v for k, v in data.items() if v}}


def _limit_motion(plans: list[dict]) -> list[dict]:
    """Enforce: not the first shot, never two in a row, at most MOTION_SHARE of shots."""
    most = max(1, round(len(plans) * MOTION_SHARE))
    count, prev = 0, False
    for i, p in enumerate(plans):
        if p.get("motion") and (i == 0 or prev or count >= most):
            p["motion"] = None
        prev = bool(p.get("motion"))
        count += prev
    return plans


def _render_motion(beats: list[str], plans: list[dict], out_dir: Path) -> dict:
    """Render every planned motion scene in one browser session. Returns {beat_index: mp4 path}."""
    import subprocess
    jobs, index = [], {}
    for i, (beat, p) in enumerate(zip(beats, plans)):
        if p.get("motion"):
            out = out_dir / f"motion_{i:02d}.mp4"
            # long enough for the line (about 2.4 words/s) plus room; trimmed to the exact length later
            seconds = min(9.0, len(beat.split()) / 2.4 + 2.0)
            jobs.append({"scene": {**p["motion"], "brand": {"name": BRAND_NAME}}, "duration": round(seconds, 2),
                         "out": str(out)})
            index[str(out)] = i
    if not jobs:
        return {}
    jobs_file = out_dir / "motion_jobs.json"
    jobs_file.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    log(f"Rendering {len(jobs)} animated scenes...")
    try:
        r = subprocess.run(["node", "render.mjs", "--batch", str(jobs_file)], cwd=MOTION_KIT_DIR,
                           capture_output=True, text=True, timeout=900)
        results = json.loads((r.stdout.strip().splitlines() or ["[]"])[-1])
    except Exception as e:
        log(f"Animated scenes failed ({e}) -- using stock footage instead")
        return {}
    done = {index[x["out"]]: Path(x["out"]) for x in results if x.get("ok") and Path(x["out"]).exists()}
    if len(done) < len(jobs):
        log(f"{len(jobs) - len(done)} animated scene(s) failed -- those shots use stock footage")
    return done


def _plan_shots(beats: list[str], topic_hint: str) -> list[dict]:
    """Ask the LLM for 3 search queries (specific -> general) and an AI-image prompt per beat."""
    numbered = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(beats))
    motion_on = _motion_enabled()
    motion_rules = _motion_prompt(len(beats)) if motion_on else ""
    motion_key = ', "motion": null' if motion_on else ""
    prompt = f"""You are a video editor choosing stock footage for a YouTube Short.
Topic: {topic_hint}

For EACH numbered narration line, plan one shot that a viewer would instantly
connect to what is being said. Return:
- "queries": 3 stock-footage searches of 1-3 simple words, from most specific
  to most general. Use common nouns that stock sites tag (e.g. "wifi router",
  "router", "living room"), never abstract ideas, brand names or people's names.
  The 3rd query must be very general but still clearly on topic.
- "ai_prompt": one sentence describing a realistic photo that would show this
  line exactly (used only if no stock footage fits).
Keep shots varied, and make line 1 the most eye-catching.
{motion_rules}
Narration:
{numbered}

Reply with ONLY a JSON array of {len(beats)} objects like
{{"queries": ["...", "...", "..."], "ai_prompt": "..."{motion_key}}}"""
    try:
        if get_gemini_key():
            # Flash Lite (500 free requests/day) -- keeps regular Flash (20/day) for scripts
            from .gemini_model import generate
            raw = generate([{"text": prompt}], max_tokens=4000, temperature=0.4, kind="vision")
        else:
            from .llm import call_llm
            raw = call_llm(prompt, max_tokens=4000)
        match = re.search(r"\[.*\]", raw, re.S)
        plans = json.loads(match.group(0)) if match else []
        ok = (len(plans) == len(beats) and all(
            isinstance(p, dict) and isinstance(p.get("queries"), list) and p["queries"] for p in plans))
        if ok:
            out = [{"queries": [str(q).strip() for q in p["queries"] if str(q).strip()][:3],
                    "ai_prompt": str(p.get("ai_prompt") or p["queries"][0]),
                    "motion": _clean_motion(p.get("motion"))} for p in plans]
            return _limit_motion(out)
        log(f"Shot plan had {len(plans)} items for {len(beats)} beats -- using keywords")
    except Exception as e:
        log(f"Could not plan shots with the LLM ({e}) -- using keywords")
    return [{"queries": [_keyword_query(b)], "ai_prompt": b, "motion": None} for b in beats]


# ---------------------------------------------------------------- vision check

VISION_PAUSE = float(os.environ.get("VISION_PAUSE", "4.5"))  # Flash Lite free tier: 15 requests/min
_vision_disabled = False


def _pick_with_vision(line: str, query: str, candidates: list[dict]) -> dict | None:
    """Show Gemini the thumbnails and let it pick the one that fits the line (or none).

    Returns the chosen candidate, None if none fit, or raises if vision is unavailable.
    """
    global _vision_disabled
    if _vision_disabled or not get_gemini_key():
        raise RuntimeError("vision unavailable")
    import base64
    from .gemini_model import generate
    parts = [{"text": (
        f'Narration line in a YouTube Short: "{line}"\n'
        f'We searched stock footage for: "{query}"\n'
        f"Below are candidate shots, numbered from 1 in order.\n"
        "Pick the ONE a viewer would instantly connect to this narration line. Reject shots that are "
        "unrelated, generic filler (beaches, sunsets, candles, cosmetics, random people), green screen, "
        "or contain big text, buttons or logos. Reply with just the number, or 0 if none clearly fits.")}]
    shown = []
    for c in candidates:
        try:
            r = requests.get(c["thumb"], timeout=20)
            r.raise_for_status()
            mime = r.headers.get("content-type", "image/jpeg").split(";")[0]
            parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(r.content).decode()}})
            shown.append(c)
        except Exception:
            continue
    if not shown:
        return None
    try:
        text = generate(parts, max_tokens=20, temperature=0, kind="vision")
    except Exception as e:
        if any(code in str(e) for code in ("API 400", "API 403", "API 404", "quota used up")):
            _vision_disabled = True
            log(f"Gemini vision unavailable -- falling back to tag matching ({str(e)[:160]})")
        raise
    import time
    time.sleep(VISION_PAUSE)
    m = re.search(r"\d+", text)
    choice = int(m.group(0)) if m else 0
    return shown[choice - 1] if 1 <= choice <= len(shown) else None


# ---------------------------------------------------------------- sourcing

def _download(url: str, dest: Path):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)


PIXABAY = "https://pixabay.com/api/"


def _pixabay_get(path: str, params: dict, key: str) -> dict:
    params = {**params, "key": key, "safesearch": "true", "q": params["q"][:100]}
    r = requests.get(PIXABAY + path, params=params, timeout=30)
    if r.status_code == 429:
        import time
        time.sleep(30)
        r = requests.get(PIXABAY + path, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _pixabay_videos(query: str, key: str) -> list[dict]:
    out = []
    for hit in _pixabay_get("videos/", {"q": query, "per_page": 20, "video_type": "film"}, key).get("hits", []):
        if hit.get("duration", 0) < 3:
            continue
        versions = [v for v in hit.get("videos", {}).values() if v.get("url") and v.get("height")]
        vertical = [v for v in versions if v["height"] > v["width"] and v["height"] >= 1280]
        pool = vertical or [v for v in versions if v["height"] >= 720]
        if not pool:
            continue
        best = (min(pool, key=lambda v: abs(v["height"] - 1920)) if vertical
                else min(pool, key=lambda v: abs(v["height"] - 1080)))
        thumb = next((v.get("thumbnail") for v in hit["videos"].values() if v.get("thumbnail")), None)
        out.append({"id": f"pv{hit['id']}", "url": best["url"], "thumb": thumb,
                    "tags": hit.get("tags", ""), "vertical": bool(vertical)})
    return out


def _pixabay_photos(query: str, key: str) -> list[dict]:
    hits = _pixabay_get("", {"q": query, "per_page": 20, "image_type": "photo"}, key).get("hits", [])
    return [{"id": f"pp{h['id']}", "url": h.get("largeImageURL") or h["webformatURL"],
             "thumb": h.get("webformatURL") or h.get("previewURL"), "tags": h.get("tags", ""),
             "vertical": h.get("imageHeight", 0) > h.get("imageWidth", 1)} for h in hits]


def _pexels_get(path: str, params: dict, key: str) -> dict:
    r = requests.get(f"{PEXELS}{path}", params=params, headers={"Authorization": key}, timeout=30)
    r.raise_for_status()
    return r.json()


def _pexels_videos(query: str, key: str) -> list[dict]:
    out = []
    for v in _pexels_get("/videos/search", {"query": query, "per_page": 20}, key).get("videos", []):
        files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("height")]
        vertical = [f for f in files if f["height"] > f["width"] and f["height"] >= 1280]
        pool = vertical or [f for f in files if f["height"] >= 720]
        if v.get("duration", 0) < 3 or not pool:
            continue
        best = (min(pool, key=lambda f: abs(f["height"] - 1920)) if vertical
                else min(pool, key=lambda f: abs(f["height"] - 1080)))
        out.append({"id": f"xv{v['id']}", "url": best["link"], "thumb": v.get("image"),
                    "tags": v.get("url", "").replace("-", " "), "vertical": bool(vertical)})
    return out


def _pexels_photos(query: str, key: str) -> list[dict]:
    photos = _pexels_get("/v1/search", {"query": query, "per_page": 20}, key).get("photos", [])
    return [{"id": f"xp{p['id']}", "url": p["src"].get("portrait") or p["src"]["large2x"],
             "thumb": p["src"].get("medium"), "tags": (p.get("alt") or "") + " " + p.get("url", ""),
             "vertical": p.get("height", 0) > p.get("width", 1)} for p in photos]


def _stock_source():
    pixabay = os.environ.get("PIXABAY_API_KEY", "").strip()
    pexels = os.environ.get("PEXELS_API_KEY", "").strip()
    if pixabay:
        return "Pixabay", pixabay, _pixabay_videos, _pixabay_photos
    if pexels:
        return "Pexels", pexels, _pexels_videos, _pexels_photos
    return None


BLOCKED_TAGS = ("green screen", "greenscreen", "chroma", "subscribe", "like button", "youtube",
                "logo", "lower third", "overlay", "intro", "outro", "countdown", "banner",
                "watermark", "text animation", "title", "bell", "alpha channel", "transition")


def _is_overlay_asset(tags: str) -> bool:
    t = tags.lower()
    return any(b in t for b in BLOCKED_TAGS)


def _rank(candidates: list[dict], query: str, line: str, used: set) -> list[dict]:
    """Rank by how well the stock tags match the query and the line; vertical only breaks ties."""
    q, ln = _words(query), _words(line)
    scored = []
    for c in candidates:
        if c["id"] in used or _is_overlay_asset(c["tags"]):
            continue
        tags = _words(c["tags"])
        score = 3 * len(q & tags) + len(ln & tags) + (0.5 if c["vertical"] else 0)
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    return [c for score, c in scored if score >= 1]


def _choose(line: str, queries: list[str], search, key: str, used: set) -> dict | None:
    """Search every query, pool the results, and make ONE Gemini check for this shot."""
    pooled: dict = {}
    for query in queries:
        try:
            results = search(query, key)
        except Exception as e:
            log(f"  search failed for '{query}' ({e})")
            continue
        for rank_pos, c in enumerate(_rank(results, query, line, used)[:6]):
            score = 3 * len(_words(query) & _words(c["tags"])) + len(_words(line) & _words(c["tags"]))
            prev = pooled.get(c["id"])
            if not prev or score > prev["score"]:
                pooled[c["id"]] = {**c, "query": query, "score": score - 0.1 * rank_pos}
    if not pooled:
        return None
    ranked = sorted(pooled.values(), key=lambda c: -c["score"])[:8]
    with_thumbs = [c for c in ranked if c.get("thumb")]
    try:
        if not with_thumbs:
            raise RuntimeError("no thumbnails")
        pick = _pick_with_vision(line, " / ".join(queries), with_thumbs)
        if not pick:
            log("  Gemini rejected every candidate")
        return pick
    except Exception:
        # no vision check available: accept only a strong tag match
        best = ranked[0]
        need = min(2, len(_words(best["query"]))) or 1
        return best if len(_words(best["query"]) & _words(best["tags"])) >= need else None


def _crop_to_portrait(path: Path):
    img = Image.open(path).convert("RGB")
    scale = max(W / img.width, H / img.height)
    img = img.resize((int(img.width * scale) + 1, int(img.height * scale) + 1), Image.LANCZOS)
    left, top = (img.width - W) // 2, (img.height - H) // 2
    img.crop((left, top, left + W, top + H)).save(path)


def generate_visuals(prompts: list, out_dir: Path, script: str | None = None, title: str = "", draft: dict | None = None, **_) -> list[Path]:
    """Drop-in replacement for broll.generate_broll."""
    story_mode_on = os.environ.get("STORY_MODE", "").strip() in ("1", "true", "True")
    if story_mode_on and script:
        try:
            try:
                from .story_mode import plan_story
            except (ImportError, ValueError):
                from story_mode import plan_story
            scenes = plan_story(script, title=title, out_dir=out_dir, draft=draft)
            story_plan = {"title": title, "scenes": scenes}
            (Path(out_dir) / "story_plan.json").write_text(
                json.dumps(story_plan, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"Story mode: planned {len(scenes)} scenes.")
            placeholder = Path(out_dir) / "story_frame.png"
            if not placeholder.exists():
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                                "-i", "color=c=black:s=1080x1920:d=1", "-frames:v", "1", str(placeholder)], check=False)
            return [placeholder]
        except Exception as e:
            log(f"Story mode failed ({e}) -- using standard assembly")

    source = _stock_source()
    if not source or not script:
        log("No PIXABAY_API_KEY (or PEXELS_API_KEY) set -- using the original 3-image visuals")
        return _broll.generate_broll(prompts, out_dir)
    source_name, key, search_videos, search_photos = source

    beats = _split_script(script)
    shot_plans = _plan_shots(beats, prompts[0] if prompts else "")
    log(f"Planning {len(beats)} shots from {source_name}")
    motion_clips = _render_motion(beats, shot_plans, out_dir) if _motion_enabled() else {}
    used: set = set()
    plan = []

    for i, (beat, sp) in enumerate(zip(beats, shot_plans)):
        entry = {"text": beat, "weight": max(1, len(beat.split()))}
        if i in motion_clips:
            m = sp["motion"]
            label = m["data"].get("value") or m["data"].get("word") or m["data"].get("title") \
                or m["data"].get("left", {}).get("title", "")
            entry.update(path=str(motion_clips[i]), kind="motion", query=f"{m['type']}: {label}")
            log(f"Shot {i + 1}/{len(beats)} [motion] {entry['query']}")
            plan.append(entry)
            continue
        try:
            pick = _choose(beat, sp["queries"], search_videos, key, used)
            if pick:
                clip = out_dir / f"shot_{i:02d}.mp4"
                _download(pick["url"], clip)
                entry.update(path=str(clip), kind="video", query=pick["query"])
            else:
                pick = _choose(beat, sp["queries"][:2], search_photos, key, used)
                if pick:
                    photo = out_dir / f"shot_{i:02d}.jpg"
                    _download(pick["url"], photo)
                    _crop_to_portrait(photo)
                    entry.update(path=str(photo), kind="image", query=pick["query"])
            if pick:
                used.add(pick["id"])
        except Exception as e:
            log(f"Shot {i + 1}: {source_name} failed ({e})")
            entry.pop("path", None)

        if "path" not in entry:
            ai = out_dir / f"shot_{i:02d}_ai.png"
            try:
                _broll._generate_image_gemini(
                    f"{sp['ai_prompt']}. Realistic cinematic photo, vertical 9:16, no text", ai,
                    get_gemini_key())
                _crop_to_portrait(ai)
                entry.update(path=str(ai), kind="image", query="AI: " + sp["ai_prompt"])
            except Exception as e:
                log(f"Shot {i + 1}: AI image failed ({e}) -- plain frame")
                entry.update(path=str(_broll._fallback_frame(i, out_dir)), kind="image", query="none")

        log(f"Shot {i + 1}/{len(beats)} [{entry['kind']}] {entry['query']}")
        plan.append(entry)

    plan[0]["hook"] = title or beats[0]
    (out_dir / PLAN_FILE).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return [Path(p["path"]) for p in plan]


# ---------------------------------------------------------------- assembly

GRADE = "eq=contrast=1.06:saturation=1.15:brightness=-0.03"
PUSH_IN = float(os.environ.get("PUSH_IN", "0.08"))  # slow zoom on every clip keeps the frame alive

# Big, bold Shorts-style captions + a hook title for the first seconds
CAPTION_FONT = os.environ.get("CAPTION_FONT", "Montserrat")
CAPTION_SIZE = int(os.environ.get("CAPTION_SIZE", "100"))
HOOK_SECONDS = float(os.environ.get("HOOK_SECONDS", "2.8"))
WATERMARK_TEXT = os.environ.get("WATERMARK_TEXT", "@unseenfacts").strip()


def _restyle_captions(ass_path: str, hook: str, out_dir: Path) -> str:
    """Rewrite the caption style (bigger, outlined, mid-lower) and add a hook title."""
    lines = Path(ass_path).read_text(encoding="utf-8").splitlines()
    style = (f"Style: Default,{CAPTION_FONT},{CAPTION_SIZE},&H00FFFFFF,&H000000FF,&H00000000,"
             f"&H96000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,520,1")
    hook_style = (f"Style: Hook,{CAPTION_FONT},{int(CAPTION_SIZE * 1.15)},&H0000E5FF,&H000000FF,"
                  f"&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,8,4,8,70,70,260,1")
    watermark_style = (f"Style: Watermark,{CAPTION_FONT},38,&H60FFFFFF,&H000000FF,&H90000000,"
                       f"&H00000000,-1,0,0,0,100,100,2,0,1,2,0,8,60,60,150,1")
    out = []
    for line in lines:
        if line.startswith("Style: Default"):
            out += [style, hook_style]
            if WATERMARK_TEXT:
                out.append(watermark_style)
        else:
            out.append(line)
    if hook:
        safe = hook.replace("{", "(").replace("}", ")").replace("\\", " ")
        end = f"0:00:{HOOK_SECONDS:05.2f}"
        out.append(f"Dialogue: 1,0:00:00.00,{end},Hook,,0,0,0,,{{\\fad(0,250)}}{safe.upper()}")
    if WATERMARK_TEXT:
        wm_safe = WATERMARK_TEXT.replace("{", "(").replace("}", ")").replace("\\", " ")
        out.append(f"Dialogue: 0,0:00:00.00,9:59:59.00,Watermark,,0,0,0,,{wm_safe}")
    new_path = Path(out_dir) / "captions_styled.ass"
    new_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return str(new_path)


def _render_video_shot(src: Path, dest: Path, seconds: float):
    try:
        clip_len = get_audio_duration(src)
    except Exception:
        clip_len = 0
    start = random.uniform(0, max(0.0, clip_len - seconds - 0.3)) if clip_len > seconds + 1 else 0
    run_cmd([
        "ffmpeg", "-stream_loop", "-1", "-ss", f"{start:.2f}", "-i", str(src),
        "-t", f"{seconds:.3f}",
        "-vf", (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                f"scale=w='trunc({W}*(1+{PUSH_IN}*t/{seconds:.3f})/2)*2':h=-2:eval=frame,"
                f"crop={W}:{H},setsar=1,fps=30,{GRADE}"),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        str(dest), "-y", "-loglevel", "error",
    ])


def _render_motion_shot(src: Path, dest: Path, seconds: float):
    run_cmd([
        "ffmpeg", "-i", str(src), "-t", f"{seconds:.3f}",
        "-vf", f"tpad=stop_mode=clone:stop_duration=10,scale={W}:{H},setsar=1,fps=30",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(dest), "-y", "-loglevel", "error",
    ])


def _render_image_shot(src: Path, dest: Path, seconds: float, index: int):
    effects = ["zoom_in", "pan_right", "zoom_out"]
    _broll.animate_frame(src, dest, seconds, effects[index % len(effects)])


def _build_duck_filter(voiceover_path, vol_speech=None, vol_gap=None):
    """Build a duck filter with fine-grained sentence gap detection."""
    vol_speech = vol_speech if vol_speech is not None else float(os.environ.get("DUCK_SPEECH", "0.22"))
    vol_gap = vol_gap if vol_gap is not None else float(os.environ.get("DUCK_GAP", "0.55"))
    try:
        from .captions import _whisper_word_timestamps
        words = _whisper_word_timestamps(Path(voiceover_path))
        if words:
            regions = []
            r_start = words[0]["start"]
            r_end = words[0]["end"]
            for w in words[1:]:
                if w["start"] - r_end < 0.18:
                    r_end = w["end"]
                else:
                    regions.append((r_start, r_end))
                    r_start = w["start"]
                    r_end = w["end"]
            regions.append((r_start, r_end))

            buffer = 0.05
            cond_str = "+".join(f"between(t,{max(0.0, s - buffer):.2f},{e + buffer:.2f})" for s, e in regions)
            if cond_str:
                return f"volume='if({cond_str}, {vol_speech}, {vol_gap})':eval=frame"
    except Exception as e:
        log(f"Warning: fine-grained duck filter fallback ({e})")
    return None


def assemble_video(frames, voiceover, out_dir, job_id, lang="en",
                   ass_path=None, music_path=None, duck_filter=None):
    """Cut between the planned shots in time with the voiceover."""
    story_mode_on = os.environ.get("STORY_MODE", "").strip() in ("1", "true", "True")
    story_plan_file = Path(out_dir) / "story_plan.json"
    if story_mode_on and story_plan_file.exists():
        try:
            try:
                from .story_mode import render_story
            except (ImportError, ValueError):
                from story_mode import render_story
            out_path = render_story(
                story_plan_file=story_plan_file,
                voiceover=voiceover,
                out_dir=out_dir,
                job_id=job_id,
                lang=lang,
                music_path=music_path,
                duck_filter=duck_filter,
                timeout=720  # 12 minutes
            )
            return out_path
        except Exception as e:
            log(f"Story mode failed ({e}) -- using standard assembly")
            if not (Path(out_dir) / PLAN_FILE).exists():
                try:
                    log("Building fallback stock visuals for standard assembly...")
                    pdata = json.loads(story_plan_file.read_text(encoding="utf-8"))
                    fallback_title = pdata.get("title", "")
                    fallback_script = " ".join(s.get("text", "") for s in pdata.get("scenes", []))
                    os.environ["STORY_MODE"] = "0"
                    generate_visuals([], out_dir, script=fallback_script, title=fallback_title)
                except Exception as fb_err:
                    log(f"Fallback visual generation warning: {fb_err}")
                finally:
                    os.environ["STORY_MODE"] = "1"

    plan_path = Path(out_dir) / PLAN_FILE
    if not plan_path.exists():
        return _original_assemble(frames=frames, voiceover=voiceover, out_dir=out_dir,
                                  job_id=job_id, lang=lang, ass_path=ass_path,
                                  music_path=music_path, duck_filter=duck_filter)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    duration = get_audio_duration(voiceover)
    total = sum(p["weight"] for p in plan)
    log(f"Assembling {len(plan)} shots over {duration:.1f}s...")

    segments = []
    for i, shot in enumerate(plan):
        seconds = duration * shot["weight"] / total + (0.3 if i == len(plan) - 1 else 0)
        seg = Path(out_dir) / f"seg_{i:02d}.mp4"
        src = Path(shot["path"])
        try:
            if shot["kind"] == "motion":
                _render_motion_shot(src, seg, seconds)
            elif shot["kind"] == "video":
                _render_video_shot(src, seg, seconds)
            else:
                _render_image_shot(src, seg, seconds, i)
        except Exception as e:
            log(f"Shot {i + 1} render failed ({e}) -- plain frame")
            _render_image_shot(_broll._fallback_frame(i, Path(out_dir)), seg, seconds, i)
        segments.append(seg)

    concat_file = Path(out_dir) / "concat.txt"
    concat_file.write_text("\n".join("file '{}'".format(str(s).replace("'", "'\\''")) for s in segments))
    merged = Path(out_dir) / "merged_video.mp4"
    run_cmd(["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_file),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
             str(merged), "-y", "-loglevel", "error"])

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MEDIA_DIR / f"verticals_{job_id}_{lang}.mp4"

    vf = None
    if ass_path and Path(ass_path).exists():
        try:
            ass_path = _restyle_captions(ass_path, plan[0].get("hook", ""), Path(out_dir))
        except Exception as e:
            log(f"Could not restyle captions ({e}) -- using the original style")
        if _ffmpeg_has_libass():
            esc = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            vf = f"ass={esc}"
        else:
            log("WARNING: this ffmpeg has no libass -- captions will not be burned in.")

    cmd = ["ffmpeg", "-i", str(merged), "-i", str(voiceover)]
    if music_path and Path(music_path).exists():
        fine_duck = _build_duck_filter(voiceover)
        active_duck = fine_duck or duck_filter
        music = f"[2:a]aloop=loop=-1:size=2e+09,atrim=0:{duration}"
        if active_duck:
            music += f",{active_duck}"
        music += "[music];[1:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        cmd += ["-stream_loop", "-1", "-i", str(music_path), "-filter_complex", music]
        audio_map = ["-map", "0:v", "-map", "[aout]"]
    else:
        audio_map = ["-map", "0:v", "-map", "1:a"]
    if vf:
        cmd += ["-vf", vf]
    cmd += audio_map + ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", "-shortest", str(out_path), "-y", "-loglevel", "error"]
    run_cmd(cmd)
    log(f"Video assembled: {out_path}")
    return out_path
