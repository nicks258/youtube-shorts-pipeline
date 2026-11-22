"""Entity resolution via Wikipedia and Wikidata APIs (keyless).

Resolves named subjects/entities to Wikidata items to extract:
- canonical title, aliases, short description
- Commons category (Wikidata P373) -> query directly
- for species: scientific name (P225) -> query iNaturalist and Commons
- for places/structures: coordinates (P625) and country/state (P17/P131)
- for events: date (P585 / P571) -> period-correct filtering

Caches resolved entities in history/entity_cache.json.
If an entity cannot be resolved, returns None (caller treats as generic and skips Tier 1).
"""

import json
import os
import pathlib
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CACHE_FILE = ROOT / "history" / "entity_cache.json"

USER_AGENT = "ShortsBot/1.0 (https://github.com/nicks258/my-shorts-bot; info@unseenfacts.com) Python-requests/2.31.0"
HEADERS = {"User-Agent": USER_AGENT}


def _http_get_json(url: str, timeout: int = 10) -> Optional[dict]:
    """Helper to GET JSON with strict Wikimedia compliant User-Agent."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None


def load_entity_cache(cache_path: Optional[pathlib.Path] = None) -> dict:
    p = cache_path or DEFAULT_CACHE_FILE
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_entity_cache(cache: dict, cache_path: Optional[pathlib.Path] = None) -> None:
    p = cache_path or DEFAULT_CACHE_FILE
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        pass


def resolve_entity(query: str, cache_path: Optional[pathlib.Path] = None) -> Optional[dict]:
    """Resolve a subject query to Wikidata entity metadata.
    Returns None if the entity cannot be cleanly resolved.
    """
    clean_q = query.strip()
    if not clean_q or len(clean_q) < 3:
        return None

    # Check cache
    cache = load_entity_cache(cache_path)
    cache_key = clean_q.lower()
    if cache_key in cache:
        return cache[cache_key]

    qid = None
    top_title = None

    # Step 1: Query Wikidata wbsearchentities
    search_url = (
        "https://www.wikidata.org/w/api.php?action=wbsearchentities&search="
        + urllib.parse.quote(clean_q)
        + "&language=en&format=json&limit=8"
    )
    data = _http_get_json(search_url)
    if data and data.get("search"):
        cands = data["search"]
        def rank_candidate(c):
            lbl = (c.get("label") or "").lower()
            desc = (c.get("description") or "").lower()
            exact = 1 if lbl == clean_q.lower() else 0
            penalize = 0
            if not any(k in clean_q.lower() for k in ("company", "inc", "corp", "software", "program", "app", "song", "album", "article")):
                junk_terms = (
                    "company", "software", "corporation", "virtual map", "operating system",
                    "family name", "surname", "disambiguation", "scientific article", "article published",
                    "extended play", "debut studio album", "studio album", "musical group",
                    "country music band", "song", "single by", "band", "film", "episode of",
                    "performing arts", "beer brewed", "helicopter"
                )
                if any(k in desc for k in junk_terms):
                    penalize = 2
            # Bonus for physical things: species, place, highway, bridge, mountain, suit, telescope, animal, wild
            physical_bonus = 0
            if any(k in desc for k in ("highway", "species", "bridge", "star", "space suit", "spaceflight", "mountain", "animal", "amphibian", "cougar", "wildlife", "cat", "wild")):
                physical_bonus = 2
            elif not desc:
                penalize += 1
            return (exact, physical_bonus, -penalize)

        cands.sort(key=rank_candidate, reverse=True)
        # If the top candidate is heavily penalized, don't use it
        top_cand = cands[0]
        top_desc = (top_cand.get("description") or "").lower()
        if not any(k in clean_q.lower() for k in ("song", "album", "article", "software")):
            if any(k in top_desc for k in ("scientific article", "extended play", "studio album", "musical group", "software", "virtual map")):
                qid = None
                top_title = None
            else:
                qid = top_cand.get("id")
                top_title = top_cand.get("label")
        else:
            qid = top_cand.get("id")
            top_title = top_cand.get("label")

    # Step 2: Fallback to Wikipedia search if Wikidata search had no direct hits
    if not qid:
        wp_url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
            + urllib.parse.quote(clean_q)
            + "&format=json&utf8=1&srlimit=3"
        )
        wp_data = _http_get_json(wp_url)
        if wp_data:
            hits = wp_data.get("query", {}).get("search", [])
            if hits:
                top_title = hits[0].get("title")
                # Fetch wikibase_item pageprops
                prop_url = (
                    "https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles="
                    + urllib.parse.quote(top_title)
                    + "&format=json"
                )
                prop_data = _http_get_json(prop_url)
                if prop_data:
                    pages = prop_data.get("query", {}).get("pages", {})
                    for p in pages.values():
                        qid = p.get("pageprops", {}).get("wikibase_item")
                        if qid:
                            break

    if not qid:
        # Cache negative result as None to avoid hammering API
        cache[cache_key] = None
        save_entity_cache(cache, cache_path)
        return None

    # Step 3: Fetch entity details & claims from Wikidata
    e_url = (
        f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qid}"
        "&props=claims|descriptions|labels|aliases&languages=en&format=json"
    )
    e_data = _http_get_json(e_url)
    if not e_data or "entities" not in e_data or qid not in e_data["entities"]:
        cache[cache_key] = None
        save_entity_cache(cache, cache_path)
        return None

    ent = e_data["entities"][qid]
    label = ent.get("labels", {}).get("en", {}).get("value") or top_title or clean_q
    desc = ent.get("descriptions", {}).get("en", {}).get("value") or ""
    aliases = [a.get("value") for a in ent.get("aliases", {}).get("en", []) if a.get("value")]
    claims = ent.get("claims", {})

    # Commons category: P373
    commons_cat = None
    if "P373" in claims and claims["P373"]:
        try:
            commons_cat = claims["P373"][0]["mainsnak"]["datavalue"]["value"]
        except Exception:
            pass

    # Scientific name: P225
    sci_name = None
    if "P225" in claims and claims["P225"]:
        try:
            sci_name = claims["P225"][0]["mainsnak"]["datavalue"]["value"]
        except Exception:
            pass

    # Coordinates: P625
    coords = None
    if "P625" in claims and claims["P625"]:
        try:
            val = claims["P625"][0]["mainsnak"]["datavalue"]["value"]
            coords = {
                "latitude": val.get("latitude"),
                "longitude": val.get("longitude"),
            }
        except Exception:
            pass

    # Inception / Event date: P585 (point in time), P571 (inception), P580 (start time)
    date_val = None
    for p_id in ("P585", "P571", "P580"):
        if p_id in claims and claims[p_id]:
            try:
                date_val = claims[p_id][0]["mainsnak"]["datavalue"]["value"].get("time")
                if date_val:
                    # e.g. +1969-07-20T00:00:00Z -> 1969-07-20
                    date_val = date_val.lstrip("+").split("T")[0]
                    break
            except Exception:
                pass

    # Country / State: P17 (country), P131 (administrative unit)
    state_or_country = None
    for p_id in ("P131", "P17"):
        if p_id in claims and claims[p_id]:
            try:
                state_or_country = claims[p_id][0]["mainsnak"]["datavalue"]["value"].get("id")
                if state_or_country:
                    break
            except Exception:
                pass

    result = {
        "qid": qid,
        "name": label,
        "description": desc,
        "aliases": aliases,
        "commons_category": commons_cat,
        "scientific_name": sci_name,
        "coordinates": coords,
        "date": date_val,
        "state_or_country": state_or_country,
    }

    cache[cache_key] = result
    save_entity_cache(cache, cache_path)
    return result
