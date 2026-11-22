"""Real photo sourcing for named real-world subjects.

Priority Order:
1. Wikimedia Commons CATEGORY search for exact subject name
2. Flickr API with license filter (licenses 4,7,8,9,10,11; CC BY-SA and CC BY-ND excluded)
   - Prefer official agency accounts (state DOTs, NPS, USFWS, USGS, NASA)
3. Openverse API (filtered to cc0, pdm, by)
4. Wikimedia Commons keyword search (filtered to cc0, pdm, by; SA and ND fail closed)
5. Pixabay footage (handled in story planner fallback)
6. AI hero image (visibly stylized when real subject is not found)

Safety & Screening:
- Every result screened: skip anything whose title/creator matches rights-holders
  (studios, publishers, game companies, sports leagues, news agencies) regardless of license.
- Prefer institutional providers (NASA, Smithsonian, NPS, USGS, government agencies, museums).
- Burn a small credit line on screen for each image.
"""

import json
import os
import re
import sys
import urllib.parse
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
FLICKR_DEFAULT_KEY = "783d7d4a8ffdb1bf82c3526ffe2ca04a"

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
            score += 3
    return score


def search_wikimedia_commons_category(subject: str, limit: int = 20) -> list[dict]:
    """Search Wikimedia Commons category members for exact subject name."""
    if not subject or len(subject.strip()) < 3:
        return []
    base_url = "https://commons.wikimedia.org/w/api.php"
    candidates = []
    try:
        # Search for exact or matching category
        cat_params = {
            "action": "query",
            "list": "search",
            "srsearch": subject.strip(),
            "srnamespace": "14",  # Category:
            "format": "json"
        }
        r = requests.get(base_url, params=cat_params, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return []
        hits = r.json().get("query", {}).get("search", [])
        if not hits:
            return []

        # Check first matching category
        cat_title = hits[0].get("title", "")
        if not cat_title.startswith("Category:"):
            return []

        mem_params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": cat_title,
            "gcmnamespace": "6",  # File:
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mime",
            "gcmlimit": str(limit),
            "format": "json"
        }
        r2 = requests.get(base_url, params=mem_params, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r2.status_code != 200:
            return []
        pages = r2.json().get("query", {}).get("pages", {})

        for pid, page in pages.items():
            ii = (page.get("imageinfo") or [{}])[0]
            img_url = ii.get("url")
            mime = str(ii.get("mime", "")).lower()
            if not img_url or ("image" not in mime and not img_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                continue

            meta = ii.get("extmetadata", {})
            lic_name = str(meta.get("LicenseShortName", {}).get("value") or meta.get("License", {}).get("value") or "").strip()
            lic_low = lic_name.lower()

            # Fail closed on SA, ND, and NC
            if any(term in lic_low for term in ("sa", "share", "nd", "derivatives", "nc", "noncommercial")):
                continue

            # Must be CC0, Public Domain, or CC BY
            is_valid = (
                "public domain" in lic_low
                or "pd" in lic_low
                or "cc0" in lic_low
                or "pdm" in lic_low
                or "usgov" in lic_low
                or "no restrictions" in lic_low
                or "cc by" in lic_low
                or "by" in lic_low
            )
            if not is_valid:
                continue

            raw_title = page.get("title", "").replace("File:", "")
            title = clean_html(raw_title)
            artist = clean_html(meta.get("Artist", {}).get("value") or meta.get("Credit", {}).get("value") or "Wikimedia Commons")
            if not artist or artist.lower() in ("unknown", "null"):
                artist = "Wikimedia Commons"

            if is_rights_holder(f"{title} {artist}"):
                continue

            score = institutional_score(f"{artist} {title}") + 5  # Category match high priority
            lic_display = "Public Domain" if any(p in lic_low for p in ("pd", "public domain", "usgov")) else lic_name
            credit = f"Photo: {artist} / Commons ({lic_display})"

            candidates.append({
                "source": "commons_category",
                "title": title,
                "creator": artist,
                "provider": f"Wikimedia Commons ({cat_title})",
                "license": lic_display,
                "url": img_url,
                "credit": credit,
                "score": score
            })
    except Exception as e:
        log(f"Commons category search failed for '{subject}': {e}")
    return candidates


def search_flickr(query: str, limit: int = 20) -> list[dict]:
    """Search Flickr API with license filter (CC BY, Public Domain, US Gov works; SA and ND excluded)."""
    api_key = os.environ.get("FLICKR_API_KEY", "").strip() or FLICKR_DEFAULT_KEY
    url = "https://api.flickr.com/services/rest/"
    # Flickr license parameter: 4=CC BY, 7=No known restrictions, 8=US Gov, 9=CC0, 10=Public Domain Mark, 11=CC BY 4.0
    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "text": query,
        "license": "4,7,8,9,10,11",
        "extras": "license,owner_name,description,tags,url_l,url_o,url_m",
        "per_page": str(limit),
        "format": "json",
        "nojsoncallback": "1"
    }
    candidates = []
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        photos = data.get("photos", {}).get("photo", [])

        # License mapping for display
        lic_map = {
            "4": "CC BY 2.0",
            "7": "Public Domain",
            "8": "U.S. Government Work",
            "9": "CC0",
            "10": "Public Domain Mark",
            "11": "CC BY 4.0"
        }

        for p in photos:
            lic_id = str(p.get("license", ""))
            # Strictly reject licenses not in our allowed whitelist (excludes 5=CC BY-SA, 6=CC BY-ND, etc.)
            if lic_id not in lic_map:
                continue

            img_url = p.get("url_l") or p.get("url_o") or p.get("url_m")
            if not img_url:
                continue

            title = clean_html(p.get("title") or "Flickr Photo")
            owner = clean_html(p.get("ownername") or "Flickr")

            if is_rights_holder(f"{title} {owner}"):
                continue

            score = institutional_score(f"{owner} {title}") + 2
            lic_display = lic_map.get(lic_id, "Free License")
            credit = f"Photo: {owner} / Flickr ({lic_display})"

            candidates.append({
                "source": "flickr",
                "title": title,
                "creator": owner,
                "provider": "Flickr",
                "license": lic_display,
                "url": img_url,
                "credit": credit,
                "score": score
            })
    except Exception as e:
        log(f"Flickr search failed for '{query}': {e}")
    return candidates


def search_openverse(query: str, limit: int = 20) -> list[dict]:
    """Search Openverse for CC0, Public Domain, and CC BY images."""
    url = f"https://api.openverse.org/v1/images/?q={urllib.parse.quote(query)}&license=cc0,pdm,by&page_size={limit}"
    candidates = []
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        for hit in data.get("results", []):
            lic = str(hit.get("license", "")).lower()
            # Must be cc0, pdm, or by (no sa, nd, nc)
            if any(term in lic for term in ("sa", "nd", "nc")):
                continue
            if lic not in ("cc0", "pdm", "by"):
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
            lic_display = "Public Domain" if lic == "pdm" else ("CC0" if lic == "cc0" else "CC BY")
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
    """Search Wikimedia Commons for CC0, Public Domain, and CC BY images (SA and ND fail closed)."""
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
            ii = (page.get("imageinfo") or [{}])[0]
            img_url = ii.get("url")
            mime = str(ii.get("mime", "")).lower()
            if not img_url or ("image" not in mime and not img_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                continue

            meta = ii.get("extmetadata", {})
            lic_name = str(meta.get("LicenseShortName", {}).get("value") or meta.get("License", {}).get("value") or "").strip()
            lic_low = lic_name.lower()

            # Strict check: SA and ND fail closed
            if any(term in lic_low for term in ("sa", "share", "nd", "derivatives", "nc", "noncommercial")):
                continue

            is_valid = (
                "public domain" in lic_low
                or "pd" in lic_low
                or "cc0" in lic_low
                or "cc-zero" in lic_low
                or "pdm" in lic_low
                or "usgov" in lic_low
                or "no restrictions" in lic_low
                or "cc by" in lic_low
                or "by" in lic_low
            )
            if not is_valid:
                continue

            raw_title = page.get("title", "").replace("File:", "")
            title = clean_html(raw_title)
            artist = clean_html(meta.get("Artist", {}).get("value") or meta.get("Credit", {}).get("value") or "Wikimedia Commons")
            if not artist or artist.lower() in ("unknown", "null"):
                artist = "Wikimedia Commons"

            # Screen against rights-holders
            if is_rights_holder(f"{title} {artist}"):
                continue

            # Strict relevance: if searching for California/Los Angeles subjects, reject unrelated states
            low_text = f"{title} {artist}".lower()
            low_query = query.lower()
            if any(term in low_query for term in ("california", "los angeles", "agoura", "101")) and any(other in low_text for other in ("oregon", "tillamook", "washington", "portland")):
                continue

            score = institutional_score(f"{artist} {title}") + 1  # baseline commons boost
            lic_display = "Public Domain" if any(p in lic_low for p in ("pd", "public domain", "usgov")) else lic_name
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


def search_real_photo_candidates(query: str, subject_name: str = "", limit: int = 15) -> list[dict]:
    """Search sources in priority order:
    1. Commons Category (exact subject)
    2. Flickr API (CC BY, PDM, US Gov; SA/ND excluded)
    3. Openverse (cc0, pdm, by)
    4. Wikimedia Commons (keyword search)
    """
    results = []

    # 1. Wikimedia Commons Category Search
    if subject_name:
        results.extend(search_wikimedia_commons_category(subject_name, limit=limit))

    # 2. Flickr API
    results.extend(search_flickr(query, limit=limit))

    # 3. Openverse
    results.extend(search_openverse(query, limit=limit))

    # 4. Wikimedia Commons Keyword Search
    results.extend(search_wikimedia_commons(query, limit=limit))

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for r in results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

    # Sort descending by institutional score while maintaining source priority
    unique.sort(key=lambda x: x["score"], reverse=True)
    return unique


def burn_credit_line(image_path: Path, credit_text: str):
    """Burn a small semi-transparent credit box in the corner onto the image."""
    if not credit_text or not image_path.is_file():
        return
    try:
        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Truncate if too long
        display_text = credit_text if len(credit_text) <= 65 else credit_text[:62] + "..."

        # Small font size proportional to image width
        font_size = max(14, int(img.width * 0.020))
        try:
            font = ImageFont.truetype("Arial", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except Exception:
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad_x, pad_y = 12, 6
        box_x0 = 24
        box_y0 = img.height - text_h - (pad_y * 2) - 28
        box_x1 = box_x0 + text_w + (pad_x * 2)
        box_y1 = box_y0 + text_h + (pad_y * 2)

        draw.rounded_rectangle([box_x0, box_y0, box_x1, box_y1], radius=6, fill=(10, 10, 10, 180))
        draw.text((box_x0 + pad_x, box_y0 + pad_y), display_text, fill=(240, 240, 240, 230), font=font)

        out = Image.alpha_composite(img, overlay).convert("RGB")
        out.save(image_path, "JPEG", quality=92)
    except Exception as e:
        log(f"Warning: could not burn credit line onto {image_path.name}: {e}")


def fetch_real_subject_photo(query: str, out_path: Path, subject_name: str = "") -> Optional[dict]:
    """Find, download, and credit a verified licensed photo for a query in priority order."""
    candidates = search_real_photo_candidates(query, subject_name=subject_name)
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

                # Re-open, check dimensions for low-res flag, and convert to RGB JPEG
                with Image.open(out_path) as im:
                    cand["width"], cand["height"] = im.size
                    cand["is_low_res"] = im.width < 720 or im.height < 720
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
