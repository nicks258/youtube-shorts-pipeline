"""Hashtag validator and helper for Unseen Facts Shorts."""

from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any

BRAND_TAG = "#UnseenFacts"

BANNED_GENERIC_TAGS = {
    "viral",
    "fyp",
    "trending",
    "foryou",
    "foryoupage",
    "shorts",
    "short",
    "reels",
    "tiktok",
    "explore",
    "explorepage",
    "viralvideo",
    "viralshorts",
}


def find_categories_json() -> pathlib.Path | None:
    """Find data/categories.json from known project locations."""
    env_path = os.environ.get("CATEGORIES_JSON")
    if env_path and pathlib.Path(env_path).is_file():
        return pathlib.Path(env_path)

    here = pathlib.Path(__file__).resolve().parent
    candidates = [
        here / "skills" / "unseen-facts-category-styles" / "data" / "categories.json",
        here / "unseen-facts-category-styles" / "data" / "categories.json",
        here.parent / "my-shorts-bot" / "skills" / "unseen-facts-category-styles" / "data" / "categories.json",
        here.parent / "my-shorts-bot" / "unseen-facts-category-styles" / "data" / "categories.json",
        here.parent / "skills" / "unseen-facts-category-styles" / "data" / "categories.json",
        pathlib.Path.cwd() / "my-shorts-bot" / "skills" / "unseen-facts-category-styles" / "data" / "categories.json",
        pathlib.Path.cwd() / "skills" / "unseen-facts-category-styles" / "data" / "categories.json",
        pathlib.Path.cwd() / "data" / "categories.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_categories_data() -> dict[str, Any]:
    """Load the full categories.json structure."""
    path = find_categories_json()
    if path and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_category_hashtags(category: str | None) -> list[str]:
    """Return category hashtags list from categories.json, or [] if unknown."""
    if not category:
        return []
    data = load_categories_data()
    categories = data.get("categories", {})
    cat_lower = category.strip().lower()
    for name, profile in categories.items():
        if name.lower() == cat_lower:
            tags = profile.get("hashtags", [])
            return [str(t) for t in tags if str(t).strip()]
    return []


def clean_hashtag(tag: str) -> str | None:
    """Validate and clean a single hashtag.

    Rules:
      - Starts with #
      - Exactly one word, alphanumeric (no spaces, punctuation, dashes)
      - Never generic (#viral, #fyp, #trending, #foryou, etc.)
    """
    if not tag or not isinstance(tag, str):
        return None
    t = tag.strip()
    if not t:
        return None
    if not t.startswith("#"):
        t = "#" + t

    # Remove internal spaces or punctuation after #
    raw_word = t[1:].strip()
    clean_word = re.sub(r"[^\w]", "", raw_word)
    if not clean_word:
        return None

    # Check banned generic tags
    if clean_word.lower() in BANNED_GENERIC_TAGS:
        return None

    # Format in CamelCase if all lowercase, otherwise preserve CamelCase
    if clean_word.islower():
        clean_word = clean_word.capitalize()

    return f"#{clean_word}"


def validate_hashtags(
    raw_hashtags: list[Any] | str | None,
    category: str | None = None,
) -> list[str]:
    """Validate and format a list of hashtags.

    Rules:
      - 3-5 hashtags in total, capped at 5
      - Drop invalid hashtags, spaces, punctuation
      - Drop duplicates and banned generic tags
      - Ensure #UnseenFacts is ALWAYS last
      - If raw_hashtags is empty or unusable, fall back to category hashtags + #UnseenFacts
      - Fall back to [#UnseenFacts] if category isn't listed
    """
    cat_tags = get_category_hashtags(category)
    brand = BRAND_TAG

    items: list[str] = []
    if isinstance(raw_hashtags, str):
        # Could be comma or space separated
        items = re.split(r"[,\s]+", raw_hashtags)
    elif isinstance(raw_hashtags, list):
        items = [str(x) for x in raw_hashtags]

    validated: list[str] = []
    seen: set[str] = set()

    for item in items:
        cleaned = clean_hashtag(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key == brand.lower():
            continue  # Brand tag will be added at the end
        if key not in seen:
            seen.add(key)
            validated.append(cleaned)

    # If nothing usable was returned, fall back to category's hashtags
    if not validated:
        for ct in cat_tags:
            cl = clean_hashtag(ct)
            if cl and cl.lower() != brand.lower() and cl.lower() not in seen:
                seen.add(cl.lower())
                validated.append(cl)

    # If we have space (< 4) and category hashtags are available, ensure at least 1 category tag
    has_category_tag = any(ct.lower() in seen for ct in cat_tags)
    if not has_category_tag and cat_tags and len(validated) < 4:
        for ct in cat_tags:
            cl = clean_hashtag(ct)
            if cl and cl.lower() != brand.lower() and cl.lower() not in seen:
                seen.add(cl.lower())
                validated.append(cl)
                break

    # Cap non-brand tags to 4 so brand tag makes total <= 5
    validated = validated[:4]

    # Always ensure brand tag is last
    validated.append(brand)

    # Cap at 5
    return validated[:5]


def apply_hashtags_to_draft(draft: dict[str, Any], category: str | None = None) -> list[str]:
    """Validate hashtags, save to draft['hashtags'], and append to youtube_description."""
    cat = category or draft.get("category") or draft.get("niche")
    raw_tags = draft.get("hashtags")
    validated = validate_hashtags(raw_tags, category=cat)
    draft["hashtags"] = validated

    # Don't put hashtags in the title: strip any accidental hashtag from title
    title = draft.get("youtube_title")
    if title and isinstance(title, str):
        clean_title = re.sub(r"\s*#\w+", "", title).strip()
        if clean_title:
            draft["youtube_title"] = clean_title

    # Append validated hashtags to the END of youtube_description on their own line
    desc = str(draft.get("youtube_description") or "").rstrip()

    # Strip existing trailing hashtag line if present
    lines = desc.split("\n")
    while lines and lines[-1].strip().startswith("#"):
        lines.pop()

    base_desc = "\n".join(lines).rstrip()
    tag_line = " ".join(validated)

    new_desc = f"{base_desc}\n\n{tag_line}" if base_desc else tag_line

    # Keep description under 4800 characters
    if len(new_desc) > 4800:
        max_body = 4800 - len(tag_line) - 2
        new_desc = base_desc[:max_body].rstrip() + "\n\n" + tag_line

    draft["youtube_description"] = new_desc
    return validated
