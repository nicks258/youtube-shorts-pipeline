"""Prepare background music for the run based on the category's music mood.

Copies music/<music_mood>/*.mp3 into pipeline/music/ (flattened).
If music/<music_mood> is missing or empty, falls back to copying every .mp3
from all subfolders of music/ directly into pipeline/music/ (flattened).
"""
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def find_categories_json() -> pathlib.Path | None:
    candidates = [
        os.environ.get("CATEGORIES_JSON", ""),
        ROOT / "unseen-facts-category-styles" / "data" / "categories.json",
        ROOT.parent / "unseen-facts-category-styles" / "data" / "categories.json",
        pathlib.Path.cwd() / "unseen-facts-category-styles" / "data" / "categories.json",
        pathlib.Path.cwd() / "data" / "categories.json",
    ]
    for c in candidates:
        if c:
            p = pathlib.Path(c).resolve()
            if p.is_file():
                return p
    return None


def get_music_mood_for_category(category: str | None) -> str:
    if not category:
        return ""
    cat_path = find_categories_json()
    if not cat_path:
        return ""
    try:
        data = json.loads(cat_path.read_text(encoding="utf-8"))
        categories = data.get("categories", {})
        for name, prof in categories.items():
            if name.lower() == category.strip().lower():
                return prof.get("music_mood", "")
    except Exception as e:
        print(f"  [music] Warning loading categories.json: {e}", file=sys.stderr)
    return ""


def find_music_source_dir() -> pathlib.Path:
    candidates = [
        ROOT / "music",
        ROOT.parent / "music",
        pathlib.Path.cwd() / "music",
        pathlib.Path.cwd().parent / "music",
    ]
    for c in candidates:
        p = pathlib.Path(c).resolve()
        if p.is_dir():
            # Verify it contains at least one subfolder or mp3
            if list(p.glob("**/*.mp3")):
                return p
    return ROOT / "music"


def prepare_music(category: str | None = None, pipeline_dir: str | pathlib.Path = "pipeline") -> list[pathlib.Path]:
    """Prepare background music for the run in pipeline/music/."""
    cat = (category or os.environ.get("TOPIC_CATEGORY", "") or os.environ.get("TOPIC_NICHE", "")).strip()
    mood = get_music_mood_for_category(cat)

    src_root = find_music_source_dir()
    dest_dir = (pathlib.Path(pipeline_dir) / "music").resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing MP3s directly in dest_dir to avoid mixing across runs
    for existing in dest_dir.glob("*.mp3"):
        try:
            existing.unlink()
        except Exception:
            pass

    # 1. Try category mood directory
    mood_dir = (src_root / mood) if mood else None
    tracks: list[pathlib.Path] = []
    if mood_dir and mood_dir.is_dir():
        tracks = sorted(mood_dir.glob("*.mp3"))

    copied: list[pathlib.Path] = []

    if tracks:
        print(f"  [music] Found {len(tracks)} tracks for mood '{mood}' ({cat or 'unspecified'})")
        for t in tracks:
            target = dest_dir / t.name
            shutil.copy2(t, target)
            copied.append(target)
    else:
        # Fallback: copy every .mp3 from all subfolders directly into dest_dir (flattened)
        print(f"  [music] Mood '{mood}' missing or empty for category '{cat}' -- running fallback")
        all_tracks = []
        for p in sorted(src_root.rglob("*.mp3")):
            # Avoid copying from dest_dir if dest_dir is inside src_root
            if dest_dir in p.parents or p.parent == dest_dir:
                continue
            all_tracks.append(p)

        if not all_tracks:
            print(f"  [music] WARNING: No MP3 tracks found anywhere in {src_root}!", file=sys.stderr)
            return []

        print(f"  [music] Fallback: copying {len(all_tracks)} tracks from all subfolders flattened into {dest_dir}")
        for t in all_tracks:
            target = dest_dir / t.name
            shutil.copy2(t, target)
            copied.append(target)

    print(f"  [music] Ready: {len(copied)} tracks in {dest_dir}")
    return copied


def main():
    cat = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TOPIC_CATEGORY", "")
    target = sys.argv[2] if len(sys.argv) > 2 else "pipeline"
    prepare_music(cat, target)


if __name__ == "__main__":
    main()
