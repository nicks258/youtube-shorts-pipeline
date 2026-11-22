"""Real photo sourcing for named real-world subjects.

Sources:
1. Openverse API (keyless, api.openverse.org/v1/images)
   - Filtered to cc0 and pdm ONLY (no attribution obligations, no share-alike)
2. Wikimedia Commons API (via extmetadata)
   - Filtered to cc0 and pdm ONLY (SA and ND fail closed)

Safety & Screening:
- Every result screened: skip anything whose title/creator matches rights-holders
  (studios, publishers, game companies, sports leagues, news agencies).
- Prefer institutional providers (NASA, Smithsonian, National Park Service, USGS,
  government agencies, museums).
- Burn a small credit line on screen for each image anyway.
"""

import json
import os
import re
import shutil
import urllib.parse
import sys
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from .log import log
except Exception:
    def log(msg: str):
        print(f"  [real_photos] {msg}", file=sys.stderr)

USER_AGENT = "shorts-bot/1.0 (contact: info@unseenfacts.com)"

# Rights-holder blocklist: skip regardless of declared license string
RIGHTS_HOLDERS = {
    # Studios, networks & streaming
    "disney", "paramount", "warner", "wb", "sony pictures", "universal pictures",
    "columbia pictures", "20th century", "lucasfilm", "marvel", "dc comics",
    "netflix", "hbo", "amazon studios", "apple tv", "cbs", "nbc", "abc", "fox",
    "lionsgate", "mgm", "peacock", "hulu", "paramount+", "pixar", "dreamworks",
    # Game developers & publishers
    "rockstar", "take-two", "take two", "nintendo", "playstation", "xbox",
    "microsoft gaming", "electronic arts", "ea sports", "activision", "blizzard",
    "ubisoft", "epic games", "valve", "bethesda", "square enix", "capcom",
    "konami", "bandai", "sega", "riot games",
    # Sports leagues & organizations
    "nfl", "nba", "mlb", "nhl", "fifa", "uefa", "premier league", "la liga",
    "f1", "formula 1", "formula one", "wwe", "ufc", "olympic", "ioc", "ncaa",
    # News & commercial stock agencies
    "getty", "getty images", "shutterstock", "istock", "alamy", "ap photo",
    "associated press", "reuters", "afp", "agence france-presse", "bloomberg",
    "new york times", "nyt", "washington post", "wall street journal", "wsj",
    "cnn", "bbc", "ign", "gamespot", "variety", "hollywood reporter", "deadline"
}

# Institutional providers to prioritize
INSTITUTIONAL_KEYWORDS = [
    "nasa", "smithsonian", "national park", "nps", "usgs", "noaa", "usda", "usfs",
    "fish and wildlife", "fws", "library of congress", "national archives", "nara",
    "santa monica mountains", "caltrans", "department of transportation", "dot",
    "defense imagery", "defense visual", "government", "gov", "u.s. government",
    "metropolitan museum", "rijksmuseum", "cleveland museum", "art institute",
    "wellcome collection", "biodiversity heritage library", "british museum"
]


def clean_html(text: str) -> str:
    """Strip HTML tags and unescape common entities."""
    if not text:
        return ""
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_rights_holder(text: str) -> bool:
    """Check if title or creator matches commercial rights-holder."""
    low = text.lower()
    for rh in RIGHTS_HOLDERS:
        if re.search(rf"\b{re.escape(rh)}\b", low):
            return True
    return False


def institutional_score(text: str) -> int:
    """Score provider/creator text for institutional trustworthiness."""
    low = text.lower()
    score = 0
    for kw in INSTITUTIONAL_KEYWORDS:
        if kw in low:
            score += 2
    return score


def search_openverse(query: str, limit: int = 20) -> list[dict]:
    """Search Openverse for CC0 and Public Domain images."""
    url = f"https://api.openverse.org/v1/images/?q={urllib.parse.quote(query)}&license=cc0,pdm&page_size={limit}"
    candidates = []
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        for hit in data.get("results", []):
            lic = str(hit.get("license", "")).lower()
            if lic not in ("cc0", "pdm"):
                continue

            title = clean_html(hit.get("title") or "")
            creator = clean_html(hit.get("creator") or "Unknown")
            provider = clean_html(hit.get("provider") or hit.get("source") or "")

            # Screen against rights-holders
            if is_rights_holder(f"{title} {creator} {provider}"):
                continue

            img_url = hit.get("url")
            if not img_url:
                continue

            score = institutional_score(f"{provider} {creator} {title}")
            lic_display = "Public Domain" if lic == "pdm" else "CC0"
            credit = f"Photo: {creator} / {provider.title() or 'Openverse'} ({lic_display})"

            candidates.append({
                "source": "openverse",
                "title": title,
                "creator": creator,
                "provider": provider,
                "license": lic_display,
                "url": img_url,
                "credit": credit,
                "score": score
            })
    except Exception as e:
        log(f"Openverse search failed for '{query}': {e}")
    return candidates


def search_wikimedia_commons(query: str, limit: int = 20) -> list[dict]:
    """Search Wikimedia Commons for CC0 / Public Domain images."""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",  # File: namespace
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "format": "json"
    }
    candidates = []
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            ii = page.get("imageinfo", [{}])[0]
            img_url = ii.get("url")
            mime = str(ii.get("mime", "")).lower()
            if not img_url or ("image" not in mime and not img_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                continue

            meta = ii.get("extmetadata", {})
            lic_name = str(meta.get("LicenseShortName", {}).get("value") or meta.get("License", {}).get("value") or "").strip()
            lic_low = lic_name.lower()

            # Strict check: CC0 and PDM ONLY. SA and ND fail closed.
            if any(term in lic_low for term in ("sa", "share", "nd", "derivatives", "nc", "noncommercial")):
                continue
            if "by" in lic_low and "cc0" not in lic_low:
                continue

            is_valid_pd = (
                "public domain" in lic_low
                or "pd" in lic_low
                or "cc0" in lic_low
                or "cc-zero" in lic_low
                or "pdm" in lic_low
                or "usgov" in lic_low
                or "no restrictions" in lic_low
            )
            if not is_valid_pd:
                continue

            raw_title = page.get("title", "").replace("File:", "")
            title = clean_html(raw_title)
            artist = clean_html(meta.get("Artist", {}).get("value") or meta.get("Credit", {}).get("value") or "Wikimedia Commons")
            if not artist or artist.lower() in ("unknown", "null"):
                artist = "Wikimedia Commons"

            # Screen against rights-holders
            if is_rights_holder(f"{title} {artist}"):
                continue

            score = institutional_score(f"{artist} {title}") + 1  # baseline commons boost
            lic_display = "Public Domain" if "pd" in lic_low or "public domain" in lic_low else "CC0"
            credit = f"Photo: {artist} / Commons ({lic_display})"

            candidates.append({
                "source": "commons",
                "title": title,
                "creator": artist,
                "provider": "Wikimedia Commons",
                "license": lic_display,
                "url": img_url,
                "credit": credit,
                "score": score
            })
    except Exception as e:
        log(f"Wikimedia Commons search failed for '{query}': {e}")
    return candidates


def search_real_photo_candidates(query: str, limit: int = 15) -> list[dict]:
    """Search Openverse and Wikimedia Commons, filtered and prioritized."""
    results = search_openverse(query, limit=limit) + search_wikimedia_commons(query, limit=limit)
    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for r in results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

    # Sort descending by institutional score
    unique.sort(key=lambda x: x["score"], reverse=True)
    return unique


def burn_credit_line(image_path: Path, credit_text: str):
    """Burn a small semi-transparent credit box on screen onto the image."""
    if not credit_text or not image_path.is_file():
        return
    try:
        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Truncate if too long
        display_text = credit_text if len(credit_text) <= 65 else credit_text[:62] + "..."

        # Choose font size proportional to image width
        font_size = max(16, int(img.width * 0.022))
        try:
            font = ImageFont.truetype("Arial", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except Exception:
                font = ImageFont.load_default()

        # Text bounding box
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Position at bottom-left corner with margins
        pad_x, pad_y = 12, 6
        box_x0 = 24
        box_y0 = img.height - text_h - (pad_y * 2) - 28
        box_x1 = box_x0 + text_w + (pad_x * 2)
        box_y1 = box_y0 + text_h + (pad_y * 2)

        # Draw semi-transparent dark pill box
        draw.rounded_rectangle([box_x0, box_y0, box_x1, box_y1], radius=8, fill=(10, 10, 10, 180))
        # Draw white / gold tinted text
        draw.text((box_x0 + pad_x, box_y0 + pad_y), display_text, fill=(240, 240, 240, 230), font=font)

        out = Image.alpha_composite(img, overlay).convert("RGB")
        out.save(image_path, "JPEG", quality=92)
    except Exception as e:
        log(f"Warning: could not burn credit line onto {image_path.name}: {e}")


def fetch_real_subject_photo(query: str, out_path: Path) -> Optional[dict]:
    """Find, download, and credit a verified CC0/PDM photo for a query."""
    candidates = search_real_photo_candidates(query)
    for cand in candidates:
        try:
            r = requests.get(cand["url"], headers={"User-Agent": USER_AGENT}, timeout=25, stream=True)
            if r.status_code == 200:
                raw_bytes = r.content
                if len(raw_bytes) < 10000:
                    continue  # too small / corrupted
                out_path.write_bytes(raw_bytes)
                # Verify PIL can open it
                with Image.open(out_path) as im:
                    im.verify()

                # Re-open and convert to RGB JPEG
                with Image.open(out_path) as im:
                    rgb = im.convert("RGB")
                    rgb.save(out_path, "JPEG", quality=94)

                # Burn on-screen credit
                burn_credit_line(out_path, cand["credit"])
                cand["path"] = str(out_path)
                log(f"Found real photo from {cand['source']}: '{cand['title']}' ({cand['license']}) by {cand['creator']}")
                return cand
        except Exception as e:
            log(f"Download candidate {cand['url'][:50]} failed: {e}")
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            continue
    return None
