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

USER_AGENT = "ShortsBot/1.0 (https://github.com/nicks258/my-shorts-bot; info@unseenfacts.com) Python-requests/2.31.0"
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
FLICKR_DEFAULT_KEY = "783d7d4a8ffdb1bf82c3526ffe2ca04a"

def _download_image_bytes(url: str, timeout: int = 15) -> Optional[bytes]:
    """Smart download trying Wikimedia-compliant UA or browser UA depending on host."""
    uas = [USER_AGENT, BROWSER_UA]
    if any(h in url for h in ("flickr.com", "inaturalist", "si.edu", "openaccess")):
        uas = [BROWSER_UA, USER_AGENT]
    for ua in uas:
        try:
            r = requests.get(url, headers={"User-Agent": ua}, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            pass
    return None

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
    """Search Wikimedia Commons category members for exact subject name or Wikidata P373 category directly."""
    if not subject or len(subject.strip()) < 3:
        return []
    base_url = "https://commons.wikimedia.org/w/api.php"
    candidates = []
    try:
        raw_s = subject.strip()
        cat_title = raw_s if raw_s.startswith("Category:") else f"Category:{raw_s}"

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
        pages = {}
        if r2.status_code == 200:
            pages = r2.json().get("query", {}).get("pages", {})

        # If direct category had no files, search for exact category match
        if not pages:
            cat_params = {
                "action": "query",
                "list": "search",
                "srsearch": raw_s.replace("Category:", ""),
                "srnamespace": "14",  # Category:
                "format": "json"
            }
            r = requests.get(base_url, params=cat_params, headers={"User-Agent": USER_AGENT}, timeout=15)
            if r.status_code == 200:
                hits = r.json().get("query", {}).get("search", [])
                for hit in hits:
                    ht = hit.get("title", "")
                    if ht.startswith("Category:") and raw_s.lower().replace("category:", "") in ht.lower():
                        cat_title = ht
                        mem_params["gcmtitle"] = cat_title
                        r_sub = requests.get(base_url, params=mem_params, headers={"User-Agent": USER_AGENT}, timeout=15)
                        if r_sub.status_code == 200:
                            sub_pages = r_sub.json().get("query", {}).get("pages", {})
                            if sub_pages:
                                pages = sub_pages
                                break

        for pid, page in pages.items():
            ii = (page.get("imageinfo") or [{}])[0]
            img_url = ii.get("url")
            mime = str(ii.get("mime", "")).lower()
            if not img_url or "svg" in mime or img_url.lower().endswith(".svg") or ("image" not in mime and not img_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
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
            if not img_url or "svg" in mime or img_url.lower().endswith(".svg") or ("image" not in mime and not img_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
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


# ---------------------------------------------------------------- specialist sources

def search_nasa(query: str, entity_data: Optional[dict] = None, limit: int = 10) -> list[dict]:
    """Search NASA Images API (images-api.nasa.gov).
    Public domain (US Federal Government Work).
    Skips items with third-party copyright in metadata, logos/insignias.
    """
    search_q = query
    if entity_data and entity_data.get("name"):
        search_q = entity_data["name"]

    candidates = []
    url = "https://images-api.nasa.gov/search"
    params = {"q": search_q, "media_type": "image"}
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        items = data.get("collection", {}).get("items", [])
        for it in items[:limit]:
            d = it.get("data", [{}])[0]
            links = it.get("links", [{}])
            if not links:
                continue
            thumb_url = links[0].get("href", "")
            title = d.get("title", "")
            desc = d.get("description", "")
            center = d.get("center", "NASA")

            combined_text = f"{title} {desc} {center}".lower()
            if any(term in combined_text for term in ("copyright", "(c)", "all rights reserved", "courtesy of")):
                continue
            if is_rights_holder(combined_text):
                continue
            if any(logo in combined_text for logo in ("meatball logo", "worm logo", "nasa insignia")):
                continue

            orig_url = thumb_url.replace("~thumb.jpg", "~large.jpg").replace("~medium.jpg", "~large.jpg")
            credit = f"Photo: NASA / {center} (Public Domain)"

            candidates.append({
                "source": "nasa",
                "title": title,
                "creator": f"NASA / {center}",
                "provider": "NASA Image and Video Library",
                "license": "Public Domain",
                "url": orig_url,
                "thumb_url": thumb_url,
                "credit": credit,
                "score": 5,
            })
    except Exception as e:
        log(f"NASA search failed for '{search_q}': {e}")
    return candidates


def search_inaturalist(query: str, entity_data: Optional[dict] = None, limit: int = 10) -> list[dict]:
    """Search iNaturalist API (api.inaturalist.org) for research-grade observations.
    Filters strictly to CC0 and CC-BY only (rejects CC-BY-NC, CC-BY-SA, All Rights Reserved).
    Uses scientific name when available.
    """
    sci_name = entity_data.get("scientific_name") if entity_data else None
    params = {
        "quality_grade": "research",
        "photos": "true",
        "per_page": limit * 2,
    }
    if sci_name:
        params["taxon_name"] = sci_name
    else:
        params["q"] = query

    candidates = []
    url = "https://api.inaturalist.org/v1/observations"
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        results = data.get("results", [])
        for obs in results:
            taxon = obs.get("taxon", {})
            tax_name = taxon.get("name") or query
            common_name = taxon.get("preferred_common_name")
            display_title = f"{tax_name} ({common_name})" if common_name else tax_name

            for photo in obs.get("photos", []):
                lic_code = (photo.get("license_code") or "").lower()
                if lic_code not in ("cc0", "cc-by"):
                    continue

                attr = photo.get("attribution") or obs.get("user", {}).get("name") or obs.get("user", {}).get("login") or "iNaturalist contributor"
                if is_rights_holder(f"{display_title} {attr}"):
                    continue

                raw_url = photo.get("url", "")
                if not raw_url:
                    continue
                large_url = raw_url.replace("square.jpg", "large.jpg").replace("small.jpg", "large.jpg")
                lic_display = "CC0" if lic_code == "cc0" else "CC BY"
                credit = f"Photo: {attr} / iNaturalist ({lic_display})"

                candidates.append({
                    "source": "inaturalist",
                    "title": display_title,
                    "creator": attr,
                    "provider": "iNaturalist",
                    "license": lic_display,
                    "url": large_url,
                    "thumb_url": raw_url,
                    "credit": credit,
                    "score": 5,
                })
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break
    except Exception as e:
        log(f"iNaturalist search failed for '{query}': {e}")
    return candidates


def search_smithsonian(query: str, entity_data: Optional[dict] = None, limit: int = 10) -> list[dict]:
    """Search Smithsonian Open Access API (api.si.edu/openaccess/api/v1.0).
    Filters strictly to the CC0 Open Access subset.
    """
    api_key = os.environ.get("SMITHSONIAN_API_KEY", os.environ.get("DATA_GOV_API_KEY", "DEMO_KEY"))
    search_q = query
    if entity_data and entity_data.get("name"):
        search_q = entity_data["name"]

    candidates = []
    url = "https://api.si.edu/openaccess/api/v1.0/search"
    params = {
        "api_key": api_key,
        "q": search_q,
        "online_media_type": "Images",
        "rows": limit * 2,
    }
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        rows = data.get("response", {}).get("rows", [])
        for r_item in rows:
            content = r_item.get("content", {})
            usage_flags = content.get("indexedStructured", {}).get("usage_flag", [])
            if "CC0" not in usage_flags:
                continue

            title = r_item.get("title", "")
            if is_rights_holder(title):
                continue

            descriptive = content.get("descriptiveNonRepeating", {})
            unit_code = descriptive.get("unit_code", "Smithsonian")
            media = descriptive.get("online_media", {}).get("media", [])
            if not media:
                continue

            img_url = media[0].get("content")
            thumb_url = media[0].get("thumbnail") or img_url
            if not img_url:
                continue

            credit = f"Photo: Smithsonian Institution / {unit_code} (CC0)"
            candidates.append({
                "source": "smithsonian",
                "title": title,
                "creator": f"Smithsonian Institution / {unit_code}",
                "provider": "Smithsonian Open Access",
                "license": "CC0",
                "url": img_url,
                "thumb_url": thumb_url,
                "credit": credit,
                "score": 5,
            })
            if len(candidates) >= limit:
                break
    except Exception as e:
        log(f"Smithsonian search failed for '{search_q}': {e}")
    return candidates


def search_library_of_congress(query: str, entity_data: Optional[dict] = None, limit: int = 10) -> list[dict]:
    """Library of Congress loc.gov JSON API.
    Note: loc.gov is gated by Cloudflare Turnstile anti-bot challenge (HTTP 403).
    Per Step 0 verification, automated HTTP queries fail closed.
    """
    return []


def verify_relevance_gate_decisions(
    b64_jpeg: str,
    entity_name: str,
    short_description: str,
    narration_text: str = "",
) -> tuple[str, float, float]:
    """Check candidate relevance using Google Decisions API (gemini-decision-1).

    Uses resolution='low' to keep cost down (~395 input tokens, $0.00004 USD).
    Asks about the SCENE's subject using that scene's narration line.
    Explicit true/false criteria:
      true: the actual named subject
      false: an unrelated object, logo, screenshot, or a different subject that merely shares a name
    Returns: (decision_status, probability, confidence)
      decision_status: 'ACCEPT' (>= 0.8), 'REJECT' (< 0.5), or 'BORDERLINE' (0.5 to 0.8).
    """
    try:
        from .reviewer import _get_genai_client, _log_review_entry
    except Exception:
        try:
            from reviewer import _get_genai_client, _log_review_entry
        except Exception:
            from verticals.reviewer import _get_genai_client, _log_review_entry

    client = _get_genai_client()

    desc_clause = f" ({short_description})" if short_description else ""
    narr_clause = f', shown during narration: "{narration_text}"' if narration_text else ""
    question_instruction = f"Does this image clearly show the scene subject: {entity_name}{desc_clause}{narr_clause}?"

    resp = client.decisions.create(
        model="gemini-decision-1",
        input=[{
            "type": "image",
            "data": b64_jpeg,
            "mime_type": "image/jpeg",
            "resolution": "low",
        }],
        questions={
            "image_relevance": {
                "type": "noul",
                "instructions": question_instruction,
                "criteria": {
                    "true_description": f"the actual named subject ({entity_name})",
                    "false_description": "an unrelated object, logo, screenshot, or a different subject that merely shares a name",
                },
            }
        },
    )

    ans = resp.answers.get("image_relevance")
    prob = float(ans.noul) if ans else 0.0
    conf = float(ans.confidence) if ans else 0.0

    if prob >= 0.8:
        status = "ACCEPT"
    elif prob < 0.5:
        status = "REJECT"
    else:
        status = "BORDERLINE"

    try:
        import datetime
        inp_tokens = resp.usage.total_input_tokens if resp.usage and resp.usage.total_input_tokens else 395
        cost_usd = round(inp_tokens * (0.10 / 1_000_000), 6)
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stage": "image_relevance",
            "provider": "google-decisions",
            "model": "gemini-decision-1",
            "entity": entity_name,
            "description": short_description,
            "probability": prob,
            "confidence": conf,
            "status": status,
            "pass": prob >= 0.5,
            "prompt_tokens": inp_tokens,
            "completion_tokens": 0,
            "total_tokens": inp_tokens,
            "cost_usd": cost_usd,
            "cost_cny": round(cost_usd * 7.14, 6),
        }
        _log_review_entry(entry)
    except Exception as e:
        log(f"Failed to log image relevance entry: {e}")

    return status, prob, conf


def verify_relevance_gate(
    img_url_or_path: str,
    entity_name: str,
    short_description: str,
    narration_text: str = "",
    candidate_title: str = "",
    candidate_caption: str = "",
) -> tuple[bool, str]:
    """Check candidate relevance using Google Decisions API (gemini-decision-1)
    with Ling-3.2-flash / Gemini vision fallback.
    Accept >= 0.8, reject < 0.5, borderline (0.5 to 0.8).
    """
    if not entity_name:
        return True, "YES (no entity specified)"
    if str(img_url_or_path).lower().endswith(".svg"):
        return False, "NO: SVG not supported"

    # Pre-vision reject: corporate logos, software, screenshots, or graphics
    cand_text = f"{candidate_title} {candidate_caption}".lower()
    if any(junk in cand_text for junk in ("logo", "screenshot", "diagram", "graphic", "software", "microsystems", "google earth")):
        if not any(k in entity_name.lower() for k in ("software", "company", "logo", "graphic")):
            return False, "NO: Unrelated logo, screenshot, or graphic"

    if entity_name.lower() in ("sun", "solar radiation") and any(bad in cand_text for bad in ("microsystems", "stanford", "coffee", "cup", "mug")):
        return False, "NO: Different subject merely sharing a name"

    try:
        import io, base64
        if img_url_or_path.startswith("http://") or img_url_or_path.startswith("https://"):
            img_bytes = _download_image_bytes(img_url_or_path, timeout=15)
            if not img_bytes:
                return False, "NO: Download failed"
        else:
            img_bytes = Path(img_url_or_path).read_bytes()

        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        long_side = max(im.size)
        if long_side > 512:
            scale = 512.0 / long_side
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 1. Try Google Decisions API first
        try:
            status, prob, conf = verify_relevance_gate_decisions(
                b64,
                entity_name,
                short_description,
                narration_text=narration_text,
            )
            verdict = f"{status} (prob={prob:.4f}, conf={conf:.4f})"
            return (status != "REJECT"), verdict
        except Exception as e:
            log(f"Decisions image relevance check error ({e}). Falling back to Ling-3.2-flash/Gemini vision.")

        # 2. Fallback to Ling-3.2-flash or Gemini vision

        desc_clause = f" ({short_description})" if short_description else ""
        narr_clause = f", as described in: \"{narration_text}\"" if narration_text else ""
        prompt = (
            f"Does this image show {entity_name}{desc_clause}{narr_clause}?\n\n"
            "Answer YES or NO first, followed by a brief 5-10 word reason.\n"
            "Format: YES: <reason> or NO: <reason>\n"
            "Answer NO if it shows an unrelated object, logo, screenshot, map/graphic, or a different subject that merely shares a name."
        )

        try:
            from .reviewer import _call_antling_chat, _call_gemini_chat, _get_antling_key, ANTLING_VISION_MODEL
        except Exception:
            try:
                from reviewer import _call_antling_chat, _call_gemini_chat, _get_antling_key, ANTLING_VISION_MODEL
            except Exception:
                from verticals.reviewer import _call_antling_chat, _call_gemini_chat, _get_antling_key, ANTLING_VISION_MODEL

        if _get_antling_key():
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}
            ]
            resp, _ = _call_antling_chat(messages, model=ANTLING_VISION_MODEL, max_tokens=35)
        else:
            parts = [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
            ]
            resp, _ = _call_gemini_chat(parts, model_kind="vision", max_tokens=35)

        cleaned = (resp or "").strip()
        upper = cleaned.upper()
        if upper.startswith("YES"):
            return True, cleaned
        return False, cleaned if cleaned else "NO: Does not show subject"
    except Exception as e:
        log(f"Relevance gate check error: {e}")
        return False, f"NO: Error ({e})"


def search_real_photo_candidates(
    query: str,
    subject_name: str = "",
    category: str = "",
    entity_data: Optional[dict] = None,
    limit: int = 15,
) -> list[dict]:
    """Search sources according to specialist category priority:
    - Space/Science -> NASA, Smithsonian, Commons Category, Flickr, Openverse, Commons Keyword
    - Animals -> iNaturalist, Smithsonian, Commons Category, Flickr, Openverse, Commons Keyword
    - History -> Library of Congress, Smithsonian, Commons Category, Flickr, Commons Keyword
    - Everything else -> Commons Category, Flickr, Openverse, Commons Keyword
    """
    category_low = (category or "").lower()
    sources_to_query = []

    cat_target = (entity_data.get("commons_category") if entity_data else None) or subject_name

    if any(k in category_low for k in ("space", "science", "astronomy")):
        sources_to_query = [
            lambda: search_nasa(query, entity_data, limit=limit),
            lambda: search_smithsonian(query, entity_data, limit=limit),
            lambda: search_wikimedia_commons_category(cat_target, limit=limit) if cat_target else [],
            lambda: search_flickr(query, limit=limit),
            lambda: search_openverse(query, limit=limit),
            lambda: search_wikimedia_commons(query, limit=limit),
        ]
    elif any(k in category_low for k in ("animal", "wildlife", "nature", "biology")):
        sources_to_query = [
            lambda: search_inaturalist(query, entity_data, limit=limit),
            lambda: search_smithsonian(query, entity_data, limit=limit),
            lambda: search_wikimedia_commons_category(cat_target, limit=limit) if cat_target else [],
            lambda: search_flickr(query, limit=limit),
            lambda: search_openverse(query, limit=limit),
            lambda: search_wikimedia_commons(query, limit=limit),
        ]
    elif any(k in category_low for k in ("history", "archaeology", "ancient")):
        sources_to_query = [
            lambda: search_library_of_congress(query, entity_data, limit=limit),
            lambda: search_smithsonian(query, entity_data, limit=limit),
            lambda: search_wikimedia_commons_category(cat_target, limit=limit) if cat_target else [],
            lambda: search_flickr(query, limit=limit),
            lambda: search_wikimedia_commons(query, limit=limit),
        ]
    else:
        sources_to_query = [
            lambda: search_wikimedia_commons_category(cat_target, limit=limit) if cat_target else [],
            lambda: search_flickr(query, limit=limit),
            lambda: search_openverse(query, limit=limit),
            lambda: search_wikimedia_commons(query, limit=limit),
        ]

    results = []
    for fetcher in sources_to_query:
        try:
            batch = fetcher()
            if batch:
                results.extend(batch)
        except Exception as e:
            log(f"Candidate search step failed: {e}")

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for r in results:
        if r.get("url") and r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

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


def fetch_real_subject_photo(
    query: str,
    out_path: Path,
    subject_name: str = "",
    category: str = "",
    entity_data: Optional[dict] = None,
    narration_text: str = "",
    rejections_log: Optional[list] = None,
) -> Optional[dict]:
    """Find, download, and credit a verified licensed photo for a query in priority order.
    Checks candidate relevance using Ling-3.2-flash / Gemini vision relevance gate (max 2 candidates).
    """
    candidates = search_real_photo_candidates(
        query,
        subject_name=subject_name,
        category=category,
        entity_data=entity_data,
    )

    entity_name = (entity_data.get("name") if entity_data else None) or subject_name or query
    description = (entity_data.get("description") if entity_data else None) or ""

    relevance_checks = 0
    for cand in candidates:
        cand_title = cand.get("title", "")
        cand_credit = cand.get("credit", "")
        cand_text = f"{cand_title} {cand_credit}".lower()

        # Reject candidates whose resolved caption or title differs from the scene's intended entity
        if entity_data and entity_data.get("name"):
            ent_name = entity_data["name"].lower()
            mismatches = [
                ("microsystems", ["microsystems"]),
                ("google earth", ["google earth"]),
                ("virtual map", ["virtual map"]),
                ("coffee", ["coffee", "cup", "mug"]),
                ("route 66", ["route 66"]),
                ("operating system", ["operating system"]),
            ]
            differed = False
            for marker, keywords in mismatches:
                if any(k in cand_text for k in keywords) and not any(k in ent_name for k in keywords):
                    cand["relevance_verdict"] = "NO: Resolved caption differs from intended entity"
                    differed = True
                    break
            if differed:
                log(f"Candidate '{cand_title}' rejected: caption differs from intended entity '{entity_data['name']}'")
                if rejections_log is not None:
                    rejections_log.append({
                        "candidate": cand_title,
                        "source": cand.get("source"),
                        "entity": entity_data.get("name"),
                        "reason": cand["relevance_verdict"]
                    })
                continue

        # Check relevance gate (max 3 candidates checked per scene to control cost)
        if entity_data:
            relevance_checks += 1
            thumb_target = cand.get("thumb_url") or cand.get("url")
            passes, verdict = verify_relevance_gate(
                thumb_target,
                entity_name,
                description,
                narration_text=narration_text,
                candidate_title=cand_title,
                candidate_caption=cand_credit,
            )
            cand["relevance_verdict"] = verdict

            if verdict.startswith("ACCEPT") or verdict.startswith("YES"):
                pass
            elif verdict.startswith("BORDERLINE"):
                log(f"Candidate '{cand_title}' is borderline ({verdict}). Trying next candidate before falling back.")
                if fallback_candidate is None:
                    fallback_candidate = cand
                if relevance_checks >= 3:
                    break
                continue
            else:
                log(f"Candidate '{cand_title}' rejected by relevance gate: {verdict}")
                if rejections_log is not None:
                    rejections_log.append({
                        "candidate": cand_title,
                        "source": cand.get("source"),
                        "entity": entity_data.get("name"),
                        "reason": verdict
                    })
                if relevance_checks >= 3:
                    break
                continue
        else:
            cand["relevance_verdict"] = "UNCHECKED"

        try:
            raw_bytes = _download_image_bytes(cand["url"], timeout=25)
            if raw_bytes and len(raw_bytes) >= 10000:
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
                cand["is_authentic"] = True
                log(f"Found real photo from {cand['source']}: '{cand['title']}' ({cand['license']}) by {cand['creator']} [Relevance: {cand.get('relevance_verdict')}]")
                return cand
        except Exception as e:
            log(f"Download candidate {cand['url'][:50]} failed: {e}")
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            if relevance_checks >= 3:
                break
            continue

    # Fall back to borderline candidate if no candidate scored >= 0.8
    if fallback_candidate:
        log(f"No candidate met >= 0.8 acceptance threshold. Falling back to borderline candidate '{fallback_candidate.get('title')}' ({fallback_candidate.get('relevance_verdict')}).")
        try:
            raw_bytes = _download_image_bytes(fallback_candidate["url"], timeout=25)
            if raw_bytes and len(raw_bytes) >= 10000:
                out_path.write_bytes(raw_bytes)
                with Image.open(out_path) as im:
                    im.verify()
                with Image.open(out_path) as im:
                    fallback_candidate["width"], fallback_candidate["height"] = im.size
                    fallback_candidate["is_low_res"] = im.width < 720 or im.height < 720
                    rgb = im.convert("RGB")
                    rgb.save(out_path, "JPEG", quality=94)
                burn_credit_line(out_path, fallback_candidate["credit"])
                fallback_candidate["path"] = str(out_path)
                fallback_candidate["is_authentic"] = True
                return fallback_candidate
        except Exception as e:
            log(f"Fallback candidate download failed: {e}")
            if out_path.exists():
                out_path.unlink(missing_ok=True)

    return None
