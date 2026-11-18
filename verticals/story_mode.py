"""Story mode: Designed motion-graphics scenes in the Unseen Facts brand.

Generates and renders a full-motion HyperFrames composition:
- plan_story: 1 Gemini Flash Lite call to split script into scenes and select scene types
- word alignment: Whisper timestamps map scenes to exact spoken timings
- hero images: one stylized 3D render per hero scene (Pollinations, 0-Pollen model, key), same
  subject description + seed for the whole video, checked by Gemini vision
- photo treatment: black-and-gold duotone for photo scenes via Pixabay + Gemini check
- render_story: runs HyperFrames and muxes voiceover + ducked music
"""
import base64
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Local / pipeline relative imports
try:
    from .gemini_model import generate
    from .config import get_gemini_key, MEDIA_DIR
    from .log import log
except (ImportError, ValueError):
    try:
        from gemini_model import generate
        from verticals.config import get_gemini_key, MEDIA_DIR
        from verticals.log import log
    except Exception:
        def log(msg: str):
            print(f"  [story] {msg}", file=sys.stderr)
        def get_gemini_key():
            return os.environ.get("GEMINI_API_KEY", "")
        MEDIA_DIR = Path.home() / ".verticals" / "media"


VALID_SCENE_TYPES = {
    "big_number", "icon", "silhouette", "ui_card", "headlines",
    "documents", "compare", "checklist", "timeline", "network",
    "map_pin", "photo", "hero",
    "map_route", "map_highlight", "scale_compare", "ranking",
    "line_chart", "scoreboard", "progress_ring", "annotate",
    "archive", "rating", "pixel_title", "poll"
}

# Every hero image = subject + moment + this style, so the whole video looks like one set.
HERO_STYLE = ("stylized 3D render, centered, pure black background, dramatic golden rim light, "
              "soft glow, high detail, no text, no letters, no watermark")
MAX_HERO_IMAGES = 6
HERO_W, HERO_H = 1000, 1100           # the size the hero scene shows the image at
# Hero images: Pollinations' authenticated endpoint, key from POLLINATIONS_KEY, and ONLY
# this 0-Pollen model (checked against the live price list every run; never a paid one).
POLLINATIONS_MODEL = "community/scriptsnsenses-sys/flux-2-dev-free"
POLLINATIONS_IMAGE_URL = "https://gen.pollinations.ai/image/"
POLLINATIONS_MODELS_URL = "https://gen.pollinations.ai/image/models"
POLLINATIONS_GAP = 8.0                # the model allows 8 requests per minute per user
# Words that would push the image model towards photographs of real events.
_REALISM = re.compile(r"\b(photo-?realistic|hyper-?realistic|realistic|photograph(?:ic|y|s)?|photos?|"
                      r"real[- ]life|dslr|8k|4k|cinematic|documentary|footage|news)\b", re.I)

UI_CARD_KINDS = {"app", "wallet", "terminal", "chat", "notification"}


def find_categories_json() -> Path | None:
    """Locate categories.json across common workspace paths."""
    candidates = [
        os.environ.get("CATEGORIES_JSON", ""),
        Path(__file__).resolve().parent / "unseen-facts-category-styles" / "data" / "categories.json",
        Path(__file__).resolve().parent.parent / "unseen-facts-category-styles" / "data" / "categories.json",
        Path(__file__).resolve().parent.parent / "my-shorts-bot" / "unseen-facts-category-styles" / "data" / "categories.json",
        Path.cwd() / "my-shorts-bot" / "unseen-facts-category-styles" / "data" / "categories.json",
        Path.cwd() / "unseen-facts-category-styles" / "data" / "categories.json",
        Path.cwd().parent / "unseen-facts-category-styles" / "data" / "categories.json",
        Path.cwd() / "data" / "categories.json",
    ]
    for c in candidates:
        if c:
            p = Path(c).resolve()
            if p.is_file():
                return p
    return None


def load_categories_data() -> dict:
    """Load the raw categories.json dictionary."""
    p = find_categories_json()
    if p and p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"Warning: could not parse categories.json ({e})")
    return {}


def get_category_profile(category_name: str | None = None) -> tuple[dict, str]:
    """Load category profile from categories.json (case-insensitive).
    
    Falls back to global defaults when missing or unknown.
    Returns (profile_dict, canonical_category_name).
    """
    data = load_categories_data()
    global_cfg = data.get("global", {})
    categories = data.get("categories", {})

    target = (category_name or os.environ.get("TOPIC_CATEGORY", "") or os.environ.get("TOPIC_NICHE", "")).strip()
    if target:
        for cat_name, prof in categories.items():
            if cat_name.lower() == target.lower():
                return prof, cat_name

    # Fallback to global defaults
    fallback_prof = {
        "look": {
            "accent": global_cfg.get("brand", {}).get("caption_highlight", "#FFD666"),
            "texture": "",
        },
        "preferred_scenes": ["hero", "big_number", "icon", "compare", "checklist"],
        "hero_style": HERO_STYLE,
        "hook_patterns": [],
        "formats": [
            {
                "id": "standard_fact",
                "name": "Standard Fact",
                "structure": "hook -> surprising fact -> explanation -> why it matters",
            }
        ],
        "length_s": global_cfg.get("target_length_s", [30, 45]),
        "seconds_per_visual": float(global_cfg.get("seconds_per_visual_max", 3.0)),
        "music_mood": "lofi-ambient",
        "ending": "loop",
    }
    return fallback_prof, target or "General"


def get_last_format_for_category(category_name: str) -> str | None:
    """Read history/videos.jsonl backwards to find the last format used by this category."""
    candidates = [
        Path(__file__).resolve().parent / "history" / "videos.jsonl",
        Path(__file__).resolve().parent.parent / "history" / "videos.jsonl",
        Path.cwd() / "history" / "videos.jsonl",
        Path.cwd().parent / "history" / "videos.jsonl",
    ]
    target_log = None
    for c in candidates:
        if c.is_file():
            target_log = c
            break

    if not target_log:
        return None

    cat_lower = category_name.strip().lower()
    try:
        lines = target_log.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if str(rec.get("category", "")).strip().lower() == cat_lower:
                fmt = rec.get("format") or rec.get("story_format")
                if fmt:
                    return str(fmt).strip()
    except Exception as e:
        log(f"Warning reading history/videos.jsonl: {e}")
    return None


def pick_category_format(profile: dict, category_name: str) -> dict:
    """Pick a format for the category, avoiding the last used one if another fits."""
    formats = profile.get("formats", [])
    if not formats:
        return {"id": "standard_fact", "name": "Standard Fact", "structure": "hook -> fact -> why it matters"}
    if len(formats) == 1:
        return formats[0]

    last_fmt = get_last_format_for_category(category_name)
    candidates = [f for f in formats if f.get("id") != last_fmt]
    if candidates:
        chosen = candidates[0]
        log(f"Picked format '{chosen.get('id')}' for {category_name} (avoided last used '{last_fmt}')")
        return chosen
    log(f"Re-using format '{formats[0].get('id')}' for {category_name} (only one available)")
    return formats[0]


def _update_draft_file_with_format(out_dir: Path | None, format_id: str):
    """Store story_format in the corresponding draft JSON file on disk."""
    drafts_dir = Path.home() / ".verticals" / "drafts"
    if not drafts_dir.exists():
        return
    # 1. Match job_id if out_dir corresponds to job_id
    if out_dir:
        job_draft = drafts_dir / f"{Path(out_dir).name}.json"
        if job_draft.is_file():
            try:
                d = json.loads(job_draft.read_text(encoding="utf-8"))
                d["story_format"] = format_id
                job_draft.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
                return
            except Exception:
                pass
    # 2. Update newest draft in ~/.verticals/drafts if recently modified
    try:
        drafts = sorted(drafts_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if drafts:
            newest = drafts[-1]
            if (time.time() - newest.stat().st_mtime) < 1800:
                d = json.loads(newest.read_text(encoding="utf-8"))
                d["story_format"] = format_id
                newest.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log(f"Warning updating draft file with format: {e}")


def get_story_template_dir() -> Path:
    """Find the story-template directory."""
    candidates = [
        os.environ.get("STORY_TEMPLATE_DIR", "").strip(),
        Path.cwd() / "story-template",
        Path(__file__).resolve().parent / "story-template",
        Path(__file__).resolve().parent.parent / "story-template",
        Path.cwd().parent / "story-template",
        Path("/Users/sumitmehra/Documents/development/practise_work/shorts/my-shorts-bot/story-template"),
    ]
    for c in candidates:
        if c:
            p = Path(c).resolve()
            if (p / "index.html").exists() and (p / "package.json").exists():
                return p
    raise FileNotFoundError("Could not find story-template directory with index.html and package.json")


def load_valid_icons() -> set[str]:
    """Load the list of supported Lucide icon names."""
    try:
        tpl = get_story_template_dir()
        icons_path = tpl / "icons.json"
        if icons_path.exists():
            return set(json.loads(icons_path.read_text(encoding="utf-8")))
    except Exception as e:
        log(f"Warning: could not load icons.json ({e})")
    return {"sparkles", "store", "trending-up", "trending-down", "shield", "globe",
            "zap", "activity", "alert-circle", "check", "x", "package", "cpu"}


def _clean_text(val, limit: int = 100) -> str:
    s = " ".join(str(val or "").split())
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "…"


def _validate_scene(s: dict, valid_icons: set[str]) -> dict:
    """Validate and sanitize a single scene against SCENES.md schema."""
    if not isinstance(s, dict):
        return {"type": "icon", "text": "", "data": {"icons": ["sparkles"], "label": "Unseen Facts"}}

    text = _clean_text(s.get("text", ""), 160)
    stype = str(s.get("type", "")).strip().lower()
    data = s.get("data") if isinstance(s.get("data"), dict) else {}
    stamp = _clean_text(data.get("stamp", ""), 20).upper() if data.get("stamp") else None

    if stype not in VALID_SCENE_TYPES:
        stype = "icon"
        data = {"icons": ["sparkles"], "label": _clean_text(text, 60)}

    clean_data: dict = {}
    if stamp:
        clean_data["stamp"] = stamp

    if stype == "big_number":
        val = str(data.get("value", "")).strip()
        if not val or not any(ch.isdigit() for ch in val):
            stype = "icon"
            clean_data.update(icons=["sparkles"], label=_clean_text(data.get("label") or text, 60))
        else:
            clean_data.update(value=_clean_text(val, 20), label=_clean_text(data.get("label") or text, 80))

    elif stype == "icon":
        raw_icons = data.get("icons", [])
        if isinstance(raw_icons, str):
            raw_icons = [raw_icons]
        icons = [i.strip() for i in raw_icons if isinstance(i, str) and i.strip() in valid_icons][:3]
        if not icons:
            icons = ["sparkles"]
        clean_data.update(icons=icons, label=_clean_text(data.get("label") or text, 80))

    elif stype == "silhouette":
        clean_data.update(label=_clean_text(data.get("label") or text, 80))

    elif stype == "ui_card":
        kind = str(data.get("kind", "notification")).strip().lower()
        if kind not in UI_CARD_KINDS:
            kind = "notification"
        raw_rows = data.get("rows", [])
        if not isinstance(raw_rows, list):
            raw_rows = [str(raw_rows)] if raw_rows else []
        clean_rows = []
        for r in raw_rows[:5]:
            if isinstance(r, list) and len(r) >= 2:
                clean_rows.append([_clean_text(r[0], 30), _clean_text(r[1], 40)])
            elif isinstance(r, dict):
                clean_rows.append({"label": _clean_text(r.get("label", ""), 30), "value": _clean_text(r.get("value", ""), 40)})
            else:
                clean_rows.append(_clean_text(str(r), 60))
        if not clean_rows:
            clean_rows = [_clean_text(text, 60)]
        clean_data.update(kind=kind, title=_clean_text(data.get("title") or "Update", 40), rows=clean_rows)

    elif stype == "headlines":
        raw_items = data.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = [str(raw_items)] if raw_items else []
        items = [_clean_text(x, 60) for x in raw_items if str(x).strip()][:3]
        if len(items) < 2:
            items.append(_clean_text(text, 60))
        clean_data.update(items=items[:3])

    elif stype == "documents":
        try:
            cnt = int(data.get("count", 3))
        except (ValueError, TypeError):
            cnt = 3
        clean_data.update(count=max(1, min(cnt, 20)), label=_clean_text(data.get("label") or "pages", 40))

    elif stype == "compare":
        left = data.get("left") if isinstance(data.get("left"), dict) else {}
        right = data.get("right") if isinstance(data.get("right"), dict) else {}
        clean_data.update(
            left={"title": _clean_text(left.get("title") or "Before", 30), "detail": _clean_text(left.get("detail", ""), 50)},
            right={"title": _clean_text(right.get("title") or "After", 30), "detail": _clean_text(right.get("detail", ""), 50)}
        )

    elif stype == "checklist":
        raw_items = data.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = [str(raw_items)] if raw_items else []
        items = [_clean_text(x, 50) for x in raw_items if str(x).strip()][:5]
        if not items:
            items = [_clean_text(text, 50)]
        mark = "check" if str(data.get("mark", "x")).lower() == "check" else "x"
        clean_data.update(items=items, mark=mark)

    elif stype == "timeline":
        raw_points = data.get("points", [])
        if not isinstance(raw_points, list):
            raw_points = []
        points = []
        for p in raw_points[:5]:
            if isinstance(p, dict):
                points.append({"year": _clean_text(p.get("year", ""), 10), "label": _clean_text(p.get("label", ""), 40)})
        if len(points) < 2:
            stype = "icon"
            clean_data = {"icons": ["clock"], "label": _clean_text(text, 60)}
        else:
            clean_data.update(points=points)

    elif stype == "network":
        clean_data.update(label=_clean_text(data.get("label") or text, 80))

    elif stype == "map_pin":
        clean_data.update(place=_clean_text(data.get("place") or "Location", 40),
                          label=_clean_text(data.get("label") or text, 80))

    elif stype == "photo":
        clean_data.update(path=str(data.get("path", "")),
                          label=_clean_text(data.get("label") or text, 80))

    elif stype == "hero":
        raw_callouts = data.get("callouts", [])
        if isinstance(raw_callouts, str):
            raw_callouts = [raw_callouts]
        callouts = [_clean_text(c, 26) for c in (raw_callouts if isinstance(raw_callouts, list) else [])
                    if str(c or "").strip()][:3]
        raw_icons = data.get("icons", [])
        if isinstance(raw_icons, str):
            raw_icons = [raw_icons]
        icons = [i.strip() for i in raw_icons if isinstance(i, str) and i.strip() in valid_icons][:3]
        clean_data.update(moment=_clean_text(data.get("moment", ""), 140),
                          label=_clean_text(data.get("label", ""), 40),
                          callouts=callouts, icons=icons or ["sparkles"])

    elif stype == "map_route":
        clean_data.update(
            **{"from": _clean_text(data.get("from") or "USA", 30),
               "to": _clean_text(data.get("to") or "FRA", 30),
               "label": _clean_text(data.get("label") or text, 80)}
        )

    elif stype == "map_highlight":
        raw_countries = data.get("countries", ["USA"])
        if isinstance(raw_countries, str):
            raw_countries = [raw_countries]
        countries = [_clean_text(c, 10).upper() for c in (raw_countries if isinstance(raw_countries, list) else [])[:3]] or ["USA"]
        clean_data.update(countries=countries, label=_clean_text(data.get("label") or text, 80))

    elif stype == "scale_compare":
        raw_items = data.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []
        clean_items = []
        for it in raw_items[:4]:
            if isinstance(it, dict):
                try:
                    val = float(it.get("value", 1))
                except (ValueError, TypeError):
                    val = 1.0
                clean_items.append({
                    "label": _clean_text(it.get("label", ""), 30),
                    "value": val,
                    "unit": _clean_text(it.get("unit", ""), 15)
                })
        if len(clean_items) < 2:
            clean_items = [
                {"label": "Standard", "value": 1, "unit": "x"},
                {"label": "Scale", "value": 10, "unit": "x"}
            ]
        clean_data.update(items=clean_items)

    elif stype == "ranking":
        raw_items = data.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []
        clean_items = []
        for it in raw_items[:6]:
            if isinstance(it, dict):
                try:
                    val = float(it.get("value", 1))
                except (ValueError, TypeError):
                    val = 1.0
                clean_items.append({
                    "label": _clean_text(it.get("label", ""), 30),
                    "value": val
                })
        if len(clean_items) < 2:
            clean_items = [
                {"label": "Top Rank", "value": 100},
                {"label": "Second Rank", "value": 75}
            ]
        clean_data.update(
            title=_clean_text(data.get("title") or "Top Ranking", 40),
            unit=_clean_text(data.get("unit", ""), 15),
            items=clean_items
        )

    elif stype == "line_chart":
        raw_pts = data.get("points", [])
        if not isinstance(raw_pts, list):
            raw_pts = []
        clean_pts = []
        for i, p in enumerate(raw_pts[:8]):
            if isinstance(p, dict):
                try:
                    x = float(p.get("x", i))
                    y = float(p.get("y", 10))
                except (ValueError, TypeError):
                    x, y = float(i), 10.0
                clean_pts.append({"x": x, "y": y})
        if len(clean_pts) < 2:
            clean_pts = [{"x": 1, "y": 10}, {"x": 2, "y": 50}]
        clean_data.update(
            label=_clean_text(data.get("label") or text, 40),
            unit=_clean_text(data.get("unit", ""), 15),
            points=clean_pts
        )

    elif stype == "scoreboard":
        clean_data.update(
            left=_clean_text(data.get("left") or "TEAM A", 25),
            right=_clean_text(data.get("right") or "TEAM B", 25),
            left_value=str(data.get("left_value", 0)),
            right_value=str(data.get("right_value", 0)),
            label=_clean_text(data.get("label") or text, 40)
        )

    elif stype == "progress_ring":
        try:
            val = int(float(data.get("value", 50)))
        except (ValueError, TypeError):
            val = 50
        clean_data.update(
            value=max(0, min(100, val)),
            label=_clean_text(data.get("label") or text, 40)
        )

    elif stype == "annotate":
        raw_callouts = data.get("callouts", [])
        if not isinstance(raw_callouts, list):
            raw_callouts = [str(raw_callouts)] if raw_callouts else []
        clean_callouts = []
        for c in raw_callouts[:3]:
            if isinstance(c, dict):
                t = _clean_text(c.get("text", ""), 30)
            else:
                t = _clean_text(str(c), 30)
            if t:
                clean_callouts.append({"text": t})
        if not clean_callouts:
            clean_callouts = [{"text": _clean_text(text, 30)}]
        clean_data.update(
            image=str(data.get("image", "")),
            callouts=clean_callouts
        )

    elif stype == "archive":
        clean_data.update(
            image=str(data.get("image", "")),
            caption=_clean_text(data.get("caption") or text, 80),
            year=_clean_text(str(data.get("year", "")), 10)
        )

    elif stype == "rating":
        try:
            score = float(data.get("score", 8.5))
        except (ValueError, TypeError):
            score = 8.5
        try:
            out_of = int(data.get("out_of", 10))
        except (ValueError, TypeError):
            out_of = 10
        clean_data.update(
            score=score,
            out_of=out_of,
            label=_clean_text(data.get("label") or text, 40),
            verdict=_clean_text(data.get("verdict", "VERDICT"), 20)
        )

    elif stype == "pixel_title":
        clean_data.update(
            title=_clean_text(data.get("title") or text, 30),
            subtitle=_clean_text(data.get("subtitle", ""), 40)
        )

    elif stype == "poll":
        clean_data.update(
            question=_clean_text(data.get("question") or text, 60),
            a=_clean_text(data.get("a", "Option A"), 25),
            b=_clean_text(data.get("b", "Option B"), 25)
        )

    return {
        "text": text,
        "type": stype,
        "data": clean_data,
        "photo_query": _clean_text(s.get("photo_query") or data.get("photo_query") or text, 40)
    }


def _enforce_variety(scenes: list[dict], valid_icons: set[str]) -> list[dict]:
    """Ensure max 25% photos and never 3 identical scene types in a row."""
    if not scenes:
        return scenes

    # 1. Limit photos to max 25% of scenes
    max_photos = max(1, len(scenes) // 4)
    photo_count = 0
    for s in scenes:
        if s["type"] == "photo":
            photo_count += 1
            if photo_count > max_photos:
                s["type"] = "icon"
                s["data"] = {"icons": ["sparkles"], "label": s["data"].get("label") or s["text"]}

    # 2. At most MAX_HERO_IMAGES hero scenes (each one is an AI image)
    hero_count = 0
    for s in scenes:
        if s["type"] == "hero":
            hero_count += 1
            if hero_count > MAX_HERO_IMAGES:
                _hero_to_icon(s)

    # 3. Never same type 3 times in a row
    for i in range(2, len(scenes)):
        if scenes[i]["type"] == scenes[i-1]["type"] == scenes[i-2]["type"]:
            prev_type = scenes[i]["type"]
            if prev_type == "hero":
                _hero_to_icon(scenes[i])
                continue
            alt_type = "silhouette" if prev_type == "icon" else "icon"
            scenes[i]["type"] = alt_type
            if alt_type == "icon":
                scenes[i]["data"] = {"icons": ["sparkles"], "label": scenes[i]["data"].get("label") or scenes[i]["text"]}
            else:
                scenes[i]["data"] = {"label": scenes[i]["data"].get("label") or scenes[i]["text"]}

    return scenes


def _hero_to_icon(s: dict):
    """Turn a hero scene into its icon fallback (keeps the label, uses the planner's icons)."""
    d = s.get("data") or {}
    s["type"] = "icon"
    s["data"] = {"icons": d.get("icons") or ["sparkles"],
                 "label": d.get("label") or (d.get("callouts") or [""])[0] or ""}


_STOP = set("""a an the and or but of to in on at for with by from as is are was were be been being it its
this that these those there their they them he she his her you your we our i my me not no so very just
than then also into over about after before more most what which who how why when where while do does did
has have had can could will would should may might one""".split())


def _content_words(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", str(text).lower()):
        if w in _STOP or len(w) < 2:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def _repeats_captions(label: str, spoken: str) -> bool:
    """True when every content word of `label` is already being said (= shown in captions)."""
    words = _content_words(label)
    return bool(words) and words <= _content_words(spoken)


_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_NUMBER_WORDS.update(thirty="30", forty="40", fifty="50", sixty="60", seventy="70", eighty="80",
                     ninety="90", hundred="100", thousand="1000", million="1000000", billion="1000000000",
                     single="1", twin="2", twins="2", double="2", dual="2", triple="3")


def _numbers_in(text: str) -> set[str]:
    """Every number in `text`, as digits ("two" -> "2", "1,200" -> "1200", "24/7" -> {"24", "7"})."""
    t = str(text).lower().replace(",", "")
    nums = set(re.findall(r"\d+(?:\.\d+)?", t))
    nums |= {_NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", t) if w in _NUMBER_WORDS}
    return nums


def _invents_numbers(label: str, script_numbers: set[str]) -> bool:
    """True when `label` states a number the script never says (e.g. "24/7 monitoring")."""
    return bool(_numbers_in(label) - script_numbers)


def _drop_redundant_labels(scenes: list[dict], script: str = ""):
    """Scene labels must ADD information but never invent it.

    Drops labels/callouts/stamps that only repeat the narration, and any that state a
    number the script does not contain.
    """
    script_numbers = _numbers_in(script)
    for s in scenes:
        d, spoken = s.get("data") or {}, s.get("text", "")
        if s["type"] in ("hero", "icon", "silhouette", "network", "photo", "big_number", "map_pin") \
                and d.get("label") and (_repeats_captions(d["label"], spoken)
                                        or _invents_numbers(d["label"], script_numbers)):
            d["label"] = ""
        if d.get("stamp") and _invents_numbers(d["stamp"], script_numbers):
            d.pop("stamp")
        if s["type"] == "hero":
            d["callouts"] = [c for c in d.get("callouts", [])
                             if not _repeats_captions(c, spoken) and not _invents_numbers(c, script_numbers)]


def plan_story(script: str, title: str = "", out_dir: Path | None = None,
               category: str | None = None, draft: dict | None = None) -> list[dict]:
    """Single Gemini Flash Lite call to plan the entire Short scene by scene.

    Also generates the hero images, and writes story_meta.json (subject, series,
    seed, format, category, image check results) next to the plan in out_dir.
    """
    if not script or not script.strip():
        raise ValueError("Cannot plan story without a script")

    profile, cat_name = get_category_profile(category)
    chosen_format = pick_category_format(profile, cat_name)
    format_id = chosen_format.get("id", "standard_fact")
    format_name = chosen_format.get("name", "Standard")
    format_structure = chosen_format.get("structure", "")

    if isinstance(draft, dict):
        draft["story_format"] = format_id

    _update_draft_file_with_format(out_dir, format_id)

    preferred_scenes = profile.get("preferred_scenes", ["hero", "big_number", "icon", "compare", "checklist"])
    length_s = profile.get("length_s", [30, 45])
    seconds_per_visual = float(profile.get("seconds_per_visual", 2.5))
    ending = str(profile.get("ending", "loop")).strip().lower()
    accent = profile.get("look", {}).get("accent", "#FFD666")
    texture = profile.get("look", {}).get("texture", "")

    valid_icons = load_valid_icons()
    popular_icons = ["sparkles", "store", "trending-up", "trending-down", "shield", "globe",
                     "zap", "activity", "alert-circle", "cpu", "laptop", "smartphone",
                     "wallet", "database", "lock", "footprints", "gamepad-2", "music",
                     "ticket", "package", "shopping-cart", "building-2", "camera"]

    prompt = f"""You are the motion-graphics director for "Unseen Facts", a high-end YouTube Shorts channel with a black + gold brand.
Design an animated motion-graphics story composition for this Short in the {cat_name} category.

TITLE: {title or 'Unseen Facts'}
CATEGORY: {cat_name}
FORMAT: {format_name} ({format_id})
FORMAT STRUCTURE: {format_structure}
TARGET LENGTH: {length_s[0]}-{length_s[1]} seconds
VISUAL PACING: A new scene every {seconds_per_visual} seconds of speech (about 5-7 words per scene).
ENDING STYLE: {"Loop ending: The final scene flows back towards the opening premise so a replay feels seamless" if ending == "loop" else "Question ending: End on a specific question or a 'poll' scene ({question, a, b}) that invites viewer comments"}

PREFERRED SCENE TYPES for {cat_name} (plan scenes MAINLY from this list):
{', '.join(preferred_scenes)}

SCRIPT:
\"\"\"{script.strip()}\"\"\"

STEP 1 - SUBJECT. Write ONE "subject" description: what the main subject of this video LOOKS like, as a visual
description an illustrator could draw (species/object, colour, size, distinctive features), 8-25 words.
Example: "a tiny olive-green baby sea turtle hatchling with two fully formed heads on one shell".
It is reused word for word for every hero image, so describe only the subject itself (no setting, no action).

STEP 2 - SCENES. Split the SCRIPT into consecutive scenes of 5-8 words each (about {seconds_per_visual} seconds of speech).
The visual must change every {seconds_per_visual} seconds: split long sentences into several scenes (a new scene at a comma
or clause), never more than 9 words in one scene.
Every single word of the narration must be included across the scenes in order with NO omissions and NO gaps.

Scene Types (prioritize PREFERRED SCENE TYPES above):
- hero: {{"moment": "what the subject is doing/showing in THIS scene", "label": "optional kicker", "callouts": ["up to 3", "short chips"], "icons": ["fallback lucide icon"]}}
  -> A large image of the main subject. Use it for every line ABOUT the subject (what it is, looks like, does).
- big_number: {{"value": "40%", "label": "short label"}} -> Years, prices, percentages, counts. (MUST contain digits).
- icon: {{"icons": ["store"], "label": "Retail stores"}} -> Abstract ideas, institutions, tools. Up to 3 Lucide names.
- silhouette: {{"label": "The founder"}} -> A human person (NEVER real faces).
- ui_card: {{"kind": "app|wallet|terminal|chat|notification", "title": "Title", "rows": ["Row 1", "Row 2"]}} -> Code, balances, apps, alerts.
- compare: {{"left": {{"title": "Before", "detail": "detail"}}, "right": {{"title": "After", "detail": "detail"}}}} -> A vs B, before/after.
- checklist: {{"items": ["Step 1", "Step 2"], "mark": "x|check"}} -> Lists, rules, "no X, no Y".
- headlines: {{"items": ["Headline 1", "Headline 2"]}} -> News coverage.
- documents: {{"count": 7, "label": "pages"}} -> "N pages", reports, filings.
- timeline: {{"points": [{{"year": "2011", "label": "Launch"}}, {{"year": "2024", "label": "Growth"}}]}} -> Progression across years.
- network: {{"label": "Global connection"}} -> Networks, global systems.
- map_pin: {{"place": "City/Country", "label": "Short label"}} -> A geographic place.
- map_route: {{"from": "USA", "to": "FRA", "label": "Flight path"}} -> Glowing arc from A to B across world map.
- map_highlight: {{"countries": ["JPN", "KOR"], "label": "East Asia"}} -> World map highlighting 1-3 countries.
- scale_compare: {{"items": [{{"label": "Moon", "value": 1, "unit": "x"}}, {{"label": "Earth", "value": 4, "unit": "x"}}]}} -> Relative comparison of 2-4 items.
- ranking: {{"title": "Top Speeds", "unit": "mph", "items": [{{"label": "Car A", "value": 240}}, {{"label": "Car B", "value": 210}}]}} -> Animated bar ranking.
- line_chart: {{"label": "Cost over time", "unit": "$", "points": [{{"x": 2020, "y": 10}}, {{"x": 2024, "y": 95}}]}} -> Trend line.
- scoreboard: {{"left": "REAL", "right": "BARCA", "left_value": 3, "right_value": 1, "label": "Final Score"}} -> Split-flap scoreboard.
- progress_ring: {{"value": 85, "label": "Completion"}} -> Circular gauge filling 0-100%.
- annotate: {{"callouts": [{{"text": "Heat shield"}}, {{"text": "Sensor"}}]}} -> Technical annotations on hero subject.
- archive: {{"caption": "Original Patent", "year": "1923"}} -> Aged paper print archive with stamped year.
- rating: {{"score": 9.2, "out_of": 10, "label": "Performance", "verdict": "OUTSTANDING"}} -> Visual score rating card.
- pixel_title: {{"title": "LEVEL 1", "subtitle": "PRESS START"}} -> Retro arcade pixel title.
- poll: {{"question": "Which would you choose?", "a": "Option A", "b": "Option B"}} -> Interactive poll / question ending.

Rules:
1. Scene 1 MUST be a hero scene showing the subject clearly (full view).
2. About HALF of all scenes are hero scenes (max 6). Use non-hero scenes ONLY for abstract ideas, numbers, and comparisons - never for the subject itself.
3. Labels, callouts and titles must ADD information the viewer is not hearing at that moment (a name, a number,
   a place, a fact from the script). NEVER repeat the words spoken in that scene - the captions already show them.
   Leave "label" empty rather than repeating the narration.
4. ONLY use numbers and facts stated in the script; never invent anything. Callouts and labels too: no numbers,
   schedules, durations or claims the script does not say (e.g. never "24/7 monitoring" unless the script says 24/7).
5. Never use the same scene type 3 times in a row (alternate hero with other types).
6. Optional: add "stamp": "WORD" in data for a dramatic verdict (e.g. CONFIRMED, RARE, FAILED).
7. Hero images: ONE simple main subject per image (no second creature/object), and no hands, people,
   crowds, text or logos in the moment. If the subject is hard to draw (unusual anatomy such as extra heads
   or limbs, complex machines), show that difficult feature in at most 2 key hero scenes; give the other
   hero scenes simpler related moments (e.g. the shell from above, its habitat, a single eye, a silhouette
   on black) that do not depend on the difficult feature.
8. "series": true ONLY if the script explicitly says this video is part of a series (e.g. "part 1"); otherwise false.

Respond with ONLY raw JSON:
{{
  "subject": "visual description of the main subject",
  "series": false,
  "scenes": [
    {{"text": "exact words spoken in this scene", "type": "hero | big_number | icon | ...", "data": {{ ... }}}}
  ]
}}
"""
    log("Planning story scenes with Gemini Flash Lite...")
    raw = generate([{"text": prompt}], max_tokens=6000, temperature=0.2, kind="vision")
    parsed = _parse_plan(raw)
    subject = _clean_text(parsed.get("subject") or title or "the main subject", 160)
    series = parsed.get("series") is True
    raw_scenes = parsed.get("scenes") or []
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("Parsed scenes list is empty")

    validated = [_validate_scene(s, valid_icons) for s in raw_scenes]
    # Rule: scene 1 is always a hero image of the subject.
    if validated[0]["type"] != "hero":
        first = validated[0]
        first["data"] = {"moment": "full view of the subject", "label": "", "callouts": [],
                         "icons": first["data"].get("icons") or ["sparkles"], **({"stamp": first["data"]["stamp"]} if first["data"].get("stamp") else {})}
        first["type"] = "hero"
    scenes = _enforce_variety(validated, valid_icons)
    _drop_redundant_labels(scenes, script)

    heroes = sum(s["type"] == "hero" for s in scenes)
    log(f"Planned {len(scenes)} scenes, {heroes} hero. Subject: {subject} [category: {cat_name}, format: {format_id}]")

    meta = {
        "subject": subject,
        "series": series,
        "seed": _video_seed(title, subject),
        "category": cat_name,
        "story_format": format_id,
        "accent": accent,
        "texture": texture,
        "images": []
    }
    if heroes:
        hero_style = profile.get("hero_style", HERO_STYLE)
        data = load_categories_data()
        image_rules = data.get("global", {}).get("image_rules", [])
        _resolve_hero_scenes(scenes, subject, meta, out_dir, hero_style=hero_style, image_rules=image_rules)

    if any(s["type"] == "photo" for s in scenes):
        _resolve_photo_scenes(scenes, out_dir)

    if out_dir:
        (Path(out_dir) / "story_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Planned {len(scenes)} story scenes successfully.")
    return scenes


def _parse_plan(raw: str) -> dict:
    """Accept {"subject", "series", "scenes"} or (older shape) a bare scene array."""
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        a, b = text.find(opener), text.rfind(closer)
        if a < 0 or b <= a:
            continue
        try:
            obj = json.loads(text[a:b + 1])
        except ValueError:
            continue
        if isinstance(obj, list):
            return {"scenes": obj}
        if isinstance(obj, dict) and isinstance(obj.get("scenes"), list):
            return obj
    raise ValueError(f"Gemini did not return a scene plan: {text[:200]}")


# ── hero images ───────────────────────────────────────────────────────────────

def _video_seed(title: str, subject: str) -> int:
    return int(hashlib.sha1(f"{title}|{subject}".encode("utf-8")).hexdigest()[:8], 16) % 1_000_000_000


def _hero_prompt(subject: str, moment: str, hero_style: str = "", image_rules: list[str] | str = "") -> str:
    """subject + moment + hero_style + image_rules, scrubbed of anything asking for a photograph.

    Pollinations reads at most 400 characters, so the style and rules always fit in full.
    """
    def scrub(t):
        return re.sub(r"\s{2,}", " ", _REALISM.sub("", str(t or ""))).strip(" ,.;")

    style = scrub(hero_style) or HERO_STYLE
    if isinstance(image_rules, list):
        rules_text = ", ".join(scrub(r) for r in image_rules if r)
    else:
        rules_text = scrub(image_rules)
    if not rules_text:
        rules_text = "stylized, never photorealistic, no faces, no logos"

    fixed_suffix = f"{style}, {rules_text}"
    budget = 400 - len(fixed_suffix) - 4
    if budget < 80:
        fixed_suffix = style[:150]
        budget = 400 - len(fixed_suffix) - 4

    subj = scrub(subject)[:min(150, max(40, budget // 2))]
    mom_budget = max(0, budget - len(subj) - 2)
    mom = scrub(moment)[:mom_budget]

    parts = [subj, mom, fixed_suffix]
    return ", ".join(p for p in parts if p)


class HeroImagesUnavailable(RuntimeError):
    """No more hero images this run (no key, model not free, auth/limit error): use icons."""


_last_image_at = [0.0]
_model_checked: dict = {}


def _model_is_free(model: str) -> bool:
    """True only if Pollinations' public model list shows `model` with a price of 0 Pollen.

    Checked before the first image of every run, so a model that becomes paid is
    never called (story mode then falls back to icons; it never picks another model).
    """
    if model in _model_checked:
        return _model_checked[model]
    import requests
    free = False
    try:
        r = requests.get(POLLINATIONS_MODELS_URL, timeout=20)
        r.raise_for_status()
        for m in r.json():
            if model == m.get("name") or model in (m.get("aliases") or []):
                prices = {k: v for k, v in (m.get("pricing") or {}).items() if k != "currency"}
                free = (not m.get("paid_only")) and all(float(v or 0) == 0 for v in prices.values()) \
                    and not any(float(a.get("price") or 0) for a in m.get("pricing_adjustments") or [])
                if not free:
                    log(f"Pollinations model {model} is not free (pricing {prices}) -- not using it")
                break
        else:
            log(f"Pollinations model {model} is not in the model list -- not using it")
    except Exception as e:
        log(f"Could not read the Pollinations model list ({type(e).__name__}) -- not generating images")
    _model_checked[model] = free
    return free


def _pollinations_image(prompt: str, out: Path, seed: int):
    """One image from the authenticated Pollinations endpoint, free model only.

    The key comes from POLLINATIONS_KEY and goes only in the Authorization header:
    never in the prompt, URL or logs. Raises HeroImagesUnavailable when no more
    images can be made this run; other errors (timeouts, bad image) raise RuntimeError.
    """
    import requests
    import urllib.parse
    key = os.environ.get("POLLINATIONS_KEY", "").strip()
    if not key:
        raise HeroImagesUnavailable("POLLINATIONS_KEY is not set")
    if not _model_is_free(POLLINATIONS_MODEL):
        raise HeroImagesUnavailable(f"model {POLLINATIONS_MODEL} is not free")
    if key in prompt:
        raise HeroImagesUnavailable("refusing to send a prompt that contains the key")
    wait = _last_image_at[0] + POLLINATIONS_GAP - time.time()
    if wait > 0:
        time.sleep(wait)
    url = (POLLINATIONS_IMAGE_URL + urllib.parse.quote(prompt[:400], safe="") + "?" + urllib.parse.urlencode({
        "model": POLLINATIONS_MODEL, "width": HERO_W, "height": HERO_H, "seed": seed,
        "nologo": "true", "enhance": "false", "safe": "true"}))
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=120)
    except Exception as e:
        raise RuntimeError(f"request failed ({type(e).__name__})") from None
    finally:
        _last_image_at[0] = time.time()
    if r.status_code in (401, 402, 403, 429):
        raise HeroImagesUnavailable(f"Pollinations answered HTTP {r.status_code} (key invalid, no access or rate-limited)")
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
        raise RuntimeError(f"Pollinations answered HTTP {r.status_code}")
    out.write_bytes(r.content)


_check_model_cache: list = []


def _check_model() -> str:
    """The vision model every hero check of this run uses: GEMINI_VISION_MODEL, else the top Flash Lite."""
    if not _check_model_cache:
        env = os.environ.get("GEMINI_VISION_MODEL", "").strip()
        if env:
            _check_model_cache.append(env)
        else:
            try:
                from .gemini_model import _ordered_models  # type: ignore
            except (ImportError, ValueError):
                from gemini_model import _ordered_models  # type: ignore
            _check_model_cache.append(_ordered_models("vision")[0])
        log(f"Hero image checks use {_check_model_cache[0]}")
    return _check_model_cache[0]


def _image_shows(path: Path, description: str) -> bool | None:
    """Ask Gemini Flash Lite whether the image shows `description`. None = check unavailable."""
    data = path.read_bytes()
    mime = "image/png" if data[:4] == b"\x89PNG" else "image/webp" if data[8:12] == b"WEBP" else "image/jpeg"
    question = (f"Does this image clearly show {description}, with correct anatomy and no text, "
                "no doubled outlines, ghosting, smeared or blurry edges, and no extra or missing body parts? "
                "Reply YES or NO.")
    parts = [{"text": question}, {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}]
    # Ask the SAME model for every image of the run. gemini_model drops a model for the
    # whole run after one timeout, and the next Flash Lite in line judges much more
    # strictly, which used to reject every later image. Pin it, retry transient errors.
    model = _check_model()
    saved = os.environ.get("GEMINI_VISION_MODEL")
    os.environ["GEMINI_VISION_MODEL"] = model
    try:
        reply = None
        for attempt in range(3):
            try:
                reply = generate(parts, max_tokens=10, temperature=0.0, kind="vision")
                break
            except Exception as e:
                if attempt == 2:
                    log(f"Hero image check unavailable on {model} ({type(e).__name__}) -- keeping the image")
                    return None
                time.sleep(4 * (attempt + 1))
    finally:
        if saved is None:
            os.environ.pop("GEMINI_VISION_MODEL", None)
        else:
            os.environ["GEMINI_VISION_MODEL"] = saved
    m = re.search(r"\b(YES|NO)\b", str(reply).upper())
    return bool(m and m.group(1) == "YES")


HERO_ATTEMPTS = 3                        # first try + 2 retries, each retry with a new seed
REUSE_FRAMINGS = ("closeup", "wide", "pan")


def _pick_reuse_source(i: int, hero_order: list[int], approved: list[int]) -> int:
    """Which approved image a failed hero scene `i` reuses.

    Never the hero scene right before or after it (the viewer would see the same
    image twice in a row) unless no other image was approved; among the rest, the
    nearest one, preferring an earlier scene on a tie.
    """
    pos = hero_order.index(i)
    neighbours = {hero_order[k] for k in (pos - 1, pos + 1) if 0 <= k < len(hero_order)}
    pool = [j for j in approved if j not in neighbours] or approved
    return min(pool, key=lambda j: (abs(i - j), j > i))


def _resolve_hero_scenes(scenes: list[dict], subject: str, meta: dict, out_dir: Path | None,
                         hero_style: str = "", image_rules: list[str] | str = ""):
    """Generate + check one image per hero scene (up to HERO_ATTEMPTS tries).

    A scene whose image fails every try reuses an approved image from this video
    with a different framing (close-up, wide, pan; never the same framing twice in
    a row), taken from a hero scene that is not right next to it when possible. Only when no image was approved at all does it become an icon scene.
    """
    tpl = get_story_template_dir()
    media_dir = tpl / "assets" / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    for old in media_dir.glob("hero_*"):  # images from earlier runs
        old.unlink(missing_ok=True)
    stamp = int(time.time())
    seed = meta["seed"]
    unavailable = False
    failed: list[int] = []
    for i, s in enumerate(scenes):
        if s["type"] != "hero":
            continue
        if unavailable:
            failed.append(i)
            continue
        moment = s["data"].get("moment") or "full view of the subject"
        prompt = _hero_prompt(subject, moment, hero_style=hero_style, image_rules=image_rules)
        out = media_dir / f"hero_{stamp}_{i:02d}.jpg"
        record = {"scene": i, "moment": moment, "prompt": prompt, "result": "failed", "seeds": []}
        for attempt in range(HERO_ATTEMPTS):
            try_seed = seed + 7919 * attempt  # same seed for the whole video; new ones only to retry
            record["seeds"].append(try_seed)
            try:
                _pollinations_image(prompt, out, try_seed)
            except HeroImagesUnavailable as e:
                log(f"Hero images unavailable ({e}) -- reusing approved images or icons")
                unavailable = True
                break
            except Exception as e:
                log(f"Hero image {i} (try {attempt + 1}/{HERO_ATTEMPTS}) failed to generate: {e}")
                continue
            ok = _image_shows(out, f"{subject}, {moment}")
            if ok is not False:
                record["result"] = (f"pass_try{attempt + 1}" if ok else "unchecked")
                break
            log(f"Hero image {i} (try {attempt + 1}/{HERO_ATTEMPTS}) rejected by the vision check")
            if out_dir:  # keep rejects for review
                shutil.copy(out, Path(out_dir) / f"rejected_{out.stem}_try{attempt + 1}.jpg")
        meta["images"].append(record)
        if record["result"] == "failed":
            out.unlink(missing_ok=True)
            failed.append(i)
            continue
        s["data"]["image"] = f"assets/media/{out.name}"
        if out_dir:
            shutil.copy(out, Path(out_dir) / out.name)

    approved = [i for i, s in enumerate(scenes) if s["type"] == "hero" and s["data"].get("image")]
    hero_order = [i for i, s in enumerate(scenes) if s["type"] == "hero"]
    for i in failed:
        if not approved:
            _hero_to_icon(scenes[i])
            continue
        src = _pick_reuse_source(i, hero_order, approved)
        scenes[i]["data"]["image"] = scenes[src]["data"]["image"]
        scenes[i]["data"]["reused_from"] = src

    # framings, in order: new images use the default slow push-in; a reused one takes
    # the next framing that differs from the hero scene right before it
    prev, k = None, 0
    for s in scenes:
        if s["type"] != "hero":
            continue
        if "reused_from" in s["data"]:
            for _ in range(len(REUSE_FRAMINGS)):
                f = REUSE_FRAMINGS[k % len(REUSE_FRAMINGS)]
                k += 1
                if f != prev:
                    break
            s["data"]["framing"] = f
        else:
            s["data"]["framing"] = "push"
        prev = s["data"]["framing"]

    new = len(approved)
    reused = sum(1 for s in scenes if s["type"] == "hero" and "reused_from" in s["data"])
    icons = len(failed) - reused
    first = sum(r["result"] == "pass_try1" for r in meta["images"])
    meta["hero_summary"] = {"new_images": new, "reused": reused, "icons": icons, "passed_first_try": first}
    log(f"Hero scenes: {new} new images ({first} passed the check on the first try), "
        f"{reused} reused with a new framing, {icons} icons")


def _resolve_photo_scenes(scenes: list[dict], out_dir: Path | None):
    """Download and treat photos for any photo scenes using Pixabay + Gemini check."""
    tpl = get_story_template_dir()
    media_dir = tpl / "assets" / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    treat_script = tpl / "tools" / "treat_photo.sh"

    # Attempt to import stock sourcing helpers from stock_visuals
    source = None
    try:
        from . import stock_visuals
        source = stock_visuals._stock_source()
    except Exception:
        try:
            import stock_visuals
            source = stock_visuals._stock_source()
        except Exception:
            pass

    used_photos: set = set()
    for i, s in enumerate(scenes):
        if s["type"] != "photo":
            continue
        query = s.get("photo_query") or s["data"].get("label") or s["text"]
        out_jpg = media_dir / f"photo_scene_{i:02d}_{int(time.time())}.jpg"
        rel_path = f"assets/media/{out_jpg.name}"

        success = False
        if source and source[1]:  # has API key
            source_name, key, _, search_photos = source
            try:
                candidates = search_photos(query, key)
                chosen = None
                if candidates:
                    try:
                        from .stock_visuals import _pick_with_vision
                        chosen = _pick_with_vision(s["text"], query, candidates[:4])
                    except Exception:
                        chosen = candidates[0]
                if chosen:
                    raw_tmp = (out_dir or Path("/tmp")) / f"raw_photo_{i}.jpg"
                    stock_visuals._download(chosen["url"], raw_tmp)
                    if treat_script.exists():
                        subprocess.run(["bash", str(treat_script), str(raw_tmp), str(out_jpg)],
                                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        _ffmpeg_duotone(raw_tmp, out_jpg)
                    if out_jpg.exists() and out_jpg.stat().st_size > 0:
                        s["data"]["path"] = rel_path
                        success = True
            except Exception as e:
                log(f"Photo scene {i} sourcing failed ({e})")

        if not success:
            # Fallback: check if test_hall.jpg exists or convert scene to safe icon
            if (media_dir / "test_hall.jpg").exists():
                s["data"]["path"] = "assets/media/test_hall.jpg"
            else:
                s["type"] = "icon"
                s["data"] = {"icons": ["sparkles"], "label": s["data"].get("label") or s["text"]}


def _ffmpeg_duotone(src: Path, dest: Path):
    """Fallback duotone filter if treat_photo.sh is missing."""
    vf = ("scale=1080:1110:force_original_aspect_ratio=increase,crop=1080:1110,"
          "format=gray,eq=contrast=1.25:brightness=-0.06:gamma=0.9,format=rgb24,"
          "lutrgb=r='val*1.0':g='val*0.84':b='val*0.40'")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-frames:v", "1",
                    "-vf", vf, "-q:v", "3", str(dest)], check=True)


def get_audio_duration(audio_path: Path) -> float:
    """Get accurate duration of an audio file in seconds via ffprobe."""
    try:
        r = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
        ], capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return 35.0


def _glue_word_pieces(words: list[dict]) -> list[dict]:
    """Whisper splits some written words ("two" + "-headed", "it" + "'s", "40" + "%"); join them back."""
    out: list[dict] = []
    for w in words:
        prev = out[-1] if out else None
        if prev and (re.match(r"^[-'’%,.!?;:)\]]", w["text"]) or re.search(r"[-(\[]$", prev["text"])):
            prev["text"] += w["text"]
            prev["end"] = max(prev["end"], w["end"])
        else:
            out.append(dict(w))
    return out


def align_word_timings(scenes: list[dict], voiceover_path: Path, script: str) -> tuple[float, list[dict], list[dict]]:
    """Align scene start/end timings to spoken words using Whisper word timestamps."""
    duration = get_audio_duration(voiceover_path)

    # Get whisper word timestamps
    whisper_words: list[dict] = []
    try:
        from verticals.captions import _whisper_word_timestamps
        whisper_words = _whisper_word_timestamps(Path(voiceover_path))
    except Exception as e:
        log(f"Whisper word timestamps unavailable ({e}) -- using estimated timings")

    clean_words = []
    for w in whisper_words:
        txt = str(w.get("word") or w.get("text") or "").strip()
        if txt:
            clean_words.append({
                "text": txt,
                "start": round(float(w["start"]), 2),
                "end": round(float(w["end"]), 2),
            })

    clean_words = _glue_word_pieces(clean_words)

    # If Whisper produced no words, generate proportional synthetic words
    if not clean_words:
        script_words = script.split()
        if script_words:
            step = duration / len(script_words)
            for idx, word in enumerate(script_words):
                clean_words.append({
                    "text": word,
                    "start": round(idx * step, 2),
                    "end": round((idx + 1) * step, 2)
                })

    # Map scenes to word timestamps
    aligned_scenes: list[dict] = []
    total_scenes = len(scenes)

    if clean_words and total_scenes > 0:
        # Match each scene's text to word positions
        word_idx = 0
        scene_starts = [0.0]

        for s_idx in range(1, total_scenes):
            scene_text = scenes[s_idx]["text"].lower()
            tokens = [re.sub(r"^\W+|\W+$", "", t) for t in scene_text.split() if t]
            found_idx = None
            if tokens:
                first_token = tokens[0]
                # Look ahead in clean_words
                for j in range(word_idx + 1, min(len(clean_words), word_idx + 25)):
                    cw = re.sub(r"^\W+|\W+$", "", clean_words[j]["text"].lower())
                    if cw == first_token:
                        found_idx = j
                        break
            if found_idx is not None and found_idx > word_idx:
                word_idx = found_idx
                t_start = clean_words[found_idx]["start"]
            else:
                # Interpolate proportionally
                t_start = round(duration * (s_idx / total_scenes), 2)
            # Ensure strictly increasing
            prev_start = scene_starts[-1]
            if t_start <= prev_start + 0.5:
                t_start = round(prev_start + 1.2, 2)
            scene_starts.append(t_start)

        # Build final scene objects with non-overlapping [start, end]
        for i in range(total_scenes):
            s = scenes[i]
            s_start = scene_starts[i]
            s_end = scene_starts[i + 1] if i + 1 < total_scenes else round(duration, 2)
            if s_end <= s_start:
                s_end = round(s_start + 1.2, 2)
            aligned_scenes.append({
                "type": s["type"],
                "start": s_start,
                "end": s_end,
                "text": s["text"],
                "data": s["data"]
            })
    else:
        # Fallback even distribution
        step = duration / max(1, total_scenes)
        for i, s in enumerate(scenes):
            aligned_scenes.append({
                "type": s["type"],
                "start": round(i * step, 2),
                "end": round((i + 1) * step if i + 1 < total_scenes else duration, 2),
                "text": s["text"],
                "data": s["data"]
            })

    aligned_scenes = _merge_short_scenes(aligned_scenes)
    return duration, clean_words, aligned_scenes


MIN_SCENE_S = 1.6        # under ~1.2 s the entrance animation cannot finish
FIRST_SCENE_END_S = 3.4  # the hook title is gone at 2.8 s: show the opening hero full size for 0.6 s+


def _merge_short_scenes(scenes: list[dict]) -> list[dict]:
    """Merge scenes too short to read into a neighbour, and keep scene 1 past the hook.

    The merged scene keeps the visual of the hero if either one is a hero (the
    subject matters most), otherwise the longer one's; narration text is joined.
    """
    out = [dict(s) for s in scenes]

    def merge(i, j):  # i, j adjacent; result replaces both
        a, b = (out[i], out[j]) if i < j else (out[j], out[i])
        if a["type"] == "hero" or b["type"] == "hero":
            keep = a if a["type"] == "hero" else b
        else:
            keep = a if (a["end"] - a["start"]) >= (b["end"] - b["start"]) else b
        m = dict(keep, start=a["start"], end=b["end"], text=f'{a["text"]} {b["text"]}'.strip())
        lo = min(i, j)
        out[lo:lo + 2] = [m]

    while len(out) > 1 and out[0]["end"] < FIRST_SCENE_END_S:
        merge(0, 1)
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, sc in enumerate(out):
            if sc["end"] - sc["start"] >= MIN_SCENE_S:
                continue
            if i == 0:
                j = 1
            elif i == len(out) - 1:
                j = i - 1
            else:  # into the shorter neighbour, so scene lengths stay even
                prev, nxt = out[i - 1], out[i + 1]
                j = i - 1 if (prev["end"] - prev["start"]) <= (nxt["end"] - nxt["start"]) else i + 1
            merge(i, j)
            changed = True
            break
    # merging can make two reused hero scenes adjacent with the same framing: re-pick
    prev = None
    for sc in out:
        if sc["type"] != "hero":
            continue
        d = sc.get("data") or {}
        if d.get("framing") == prev and "reused_from" in d:
            d["framing"] = next(f for f in REUSE_FRAMINGS if f != prev)
        prev = d.get("framing")
    return out


def build_story_vars(scenes: list[dict], voiceover_path: Path, title: str, script: str = "",
                     series: bool = False, accent: str = "", texture: str = "", category: str = "") -> dict:
    """Build the full vars.json data structure for HyperFrames."""
    duration, words, aligned_scenes = align_word_timings(scenes, voiceover_path, script)

    # Choose hook highlight word (longest word in title, or first capitalized word)
    title_words = [w for w in re.findall(r"[A-Za-z0-9]+", title) if len(w) > 3]
    hook_highlight = max(title_words, key=len).upper() if title_words else "UNSEEN"

    watermark = os.environ.get("WATERMARK_TEXT", "@reelunseenfacts").strip() or "@reelunseenfacts"

    skip_captions = os.environ.get("CAPTIONS") == "0"

    vars_dict = {
        "duration": round(duration, 2),
        "title": title or "Unseen Facts",
        "hook_highlight": hook_highlight,
        "watermark": watermark,
        # Only promise a Part 2 when the plan explicitly marks the video as part of a series.
        "end_card": {"text": "FOLLOW FOR PART 2" if series else "FOLLOW FOR MORE"},
        "accent": accent or "#FFD666",
        "texture": texture or "",
        "category": category or "General",
        "words": [] if skip_captions else words,
        "scenes": aligned_scenes
    }
    return vars_dict


def render_story(story_plan_file: Path, voiceover: Path, out_dir: Path, job_id: str,
                 lang: str = "en", music_path: Path | None = None, duck_filter: str | None = None,
                 timeout: int = 720) -> Path:
    """Render HyperFrames composition to MP4 and mux voiceover + music."""
    tpl = get_story_template_dir()
    plan_data = json.loads(Path(story_plan_file).read_text(encoding="utf-8"))
    scenes = plan_data if isinstance(plan_data, list) else plan_data.get("scenes", [])
    title = plan_data.get("title", "") if isinstance(plan_data, dict) else ""
    script = " ".join(s.get("text", "") for s in scenes)

    meta_file = Path(story_plan_file).parent / "story_meta.json"
    meta = {}
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except ValueError:
            meta = {}
    series = (plan_data.get("series") if isinstance(plan_data, dict) else None) is True or meta.get("series") is True
    accent = meta.get("accent") or (plan_data.get("accent") if isinstance(plan_data, dict) else "")
    texture = meta.get("texture") or (plan_data.get("texture") if isinstance(plan_data, dict) else "")
    category = meta.get("category") or (plan_data.get("category") if isinstance(plan_data, dict) else "")

    if not accent or not texture:
        prof, cat_name = get_category_profile(category)
        accent = accent or prof.get("look", {}).get("accent", "#FFD666")
        texture = texture or prof.get("look", {}).get("texture", "")
        category = category or cat_name

    vars_data = build_story_vars(scenes, Path(voiceover), title=title, script=script, series=series,
                                 accent=accent, texture=texture, category=category)
    duration = vars_data["duration"]

    # Write vars.json in story-template
    tpl_vars = tpl / "vars.json"
    tpl_vars.write_text(json.dumps(vars_data, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also keep a copy in work_dir for inspection
    (Path(out_dir) / "vars.json").write_text(tpl_vars.read_text(encoding="utf-8"), encoding="utf-8")

    raw_mp4 = Path(out_dir) / f"hyperframes_raw_{job_id}.mp4"
    log(f"Rendering HyperFrames composition ({duration:.1f}s, target {raw_mp4.name})...")
    t0 = time.time()

    cmd = [
        "npx", "hyperframes", "render", ".",
        "--variables-file", "vars.json",
        "--end", str(duration),
        "-o", str(raw_mp4)
    ]
    env = {**os.environ, "HYPERFRAMES_NO_TELEMETRY": "1"}
    r = subprocess.run(cmd, cwd=tpl, env=env, timeout=timeout, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"HyperFrames render failed (code {r.returncode}):\n{r.stderr[:400]}")

    render_time = time.time() - t0
    log(f"HyperFrames rendered in {render_time:.1f}s.")

    if not raw_mp4.exists() or raw_mp4.stat().st_size == 0:
        raise RuntimeError(f"Rendered file {raw_mp4} is missing or empty")

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MEDIA_DIR / f"verticals_{job_id}_{lang}.mp4"

    # Mux voiceover and background music with ducking
    log(f"Muxing audio with ffmpeg into {out_path.name}...")
    mux_cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(raw_mp4), "-i", str(voiceover)]
    if music_path and Path(music_path).exists():
        from . import stock_visuals
        fine_duck = getattr(stock_visuals, "_build_duck_filter", lambda v: None)(voiceover)
        active_duck = fine_duck or duck_filter
        music_filter = f"[2:a]aloop=loop=-1:size=2e+09,atrim=0:{duration}"
        if active_duck:
            music_filter += f",{active_duck}"
        music_filter += "[music];[1:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        mux_cmd += ["-stream_loop", "-1", "-i", str(music_path), "-filter_complex", music_filter,
                    "-map", "0:v", "-map", "[aout]"]
    else:
        mux_cmd += ["-map", "0:v", "-map", "1:a"]

    mux_cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out_path)]
    res = subprocess.run(mux_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Retry with re-encoding if copy fails
        mux_cmd[-6] = "libx264"
        mux_cmd.insert(-5, "-preset")
        mux_cmd.insert(-5, "fast")
        mux_cmd.insert(-5, "-crf")
        mux_cmd.insert(-5, "20")
        subprocess.run(mux_cmd, check=True)

    log(f"Final video assembled: {out_path} ({render_time:.1f}s render)")
    return out_path
