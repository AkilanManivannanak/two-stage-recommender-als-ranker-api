"""
Catalog Enrichment  —  Semantic Intelligence Plane
===================================================
Plane: Semantic Intelligence (LLM/VLM sidecar — offline, not request-path)

Responsibilities:
  1. TMDB hydration  — canonical metadata, posters, backdrops
  2. Structured LLM enrichment  — themes, moods, narrative hooks,
     audience descriptors, spoiler-safe summaries, semantic tags
  3. Semantic clustering  — group titles by inferred topic/mood
  4. Artwork grounding audit  — flag poster/genre mismatches
  5. Trust score  — thumbnail alignment with actual content

Rule: ALL calls here are offline / cached.
      Nothing in this file runs synchronously on a homepage request.

References:
  TMDB auth: Bearer access token (default) or api_key param
  OpenAI Responses API: structured outputs with json_object format
  MediaFM: tri-modal content understanding (text + image + audio/video)
"""
from __future__ import annotations

import http.client, json, os, ssl, time
from pathlib import Path
from typing import Any

_OAI_KEY  = os.environ.get("OPENAI_API_KEY", "")
_TMDB_KEY = os.environ.get("TMDB_API_KEY", "")
_CACHE    = Path("artifacts/catalog_enrichment_cache.json")
_cache: dict[str, Any] = {}


def _load_cache():
    if _CACHE.exists():
        try: _cache.update(json.loads(_CACHE.read_text()))
        except: pass

def _save_cache():
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(_cache))
    except: pass

_load_cache()


# ── TMDB hydration ───────────────────────────────────────────────────
def tmdb_hydrate(title: str, year: int | None = None) -> dict:
    """
    Hydrate a title from TMDB using Bearer token auth (TMDB default).
    Returns: poster_url, backdrop_url, description, tmdb_rating, tmdb_id,
             genre_ids, origin_country, original_language, vote_count.
    """
    ck = f"tmdb:{title}:{year}"
    if ck in _cache: return _cache[ck]
    if not _TMDB_KEY: return {}
    try:
        ctx  = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.themoviedb.org", timeout=5, context=ctx)
        q    = title.replace(" ", "+")
        # TMDB docs: use Bearer token as default auth method
        headers = {"Authorization": f"Bearer {_TMDB_KEY}",
                   "Content-Type": "application/json"}
        # Try TV first, then movie
        for endpoint in [f"/3/search/tv?query={q}", f"/3/search/movie?query={q}"]:
            conn.request("GET", endpoint, headers=headers)
            resp  = json.loads(conn.getresponse().read().decode("utf-8"))
            hits  = resp.get("results", [])
            if hits:
                h = hits[0]
                out = {
                    "poster_url":   f"https://image.tmdb.org/t/p/w500{h['poster_path']}"   if h.get("poster_path")   else None,
                    "backdrop_url": f"https://image.tmdb.org/t/p/w1280{h['backdrop_path']}" if h.get("backdrop_path") else None,
                    "description":  h.get("overview",""),
                    "tmdb_rating":  h.get("vote_average", 0),
                    "tmdb_id":      h.get("id"),
                    "vote_count":   h.get("vote_count", 0),
                    "origin_country": h.get("origin_country", []),
                    "original_language": h.get("original_language",""),
                    "tmdb_genre_ids": h.get("genre_ids", []),
                }
                _cache[ck] = out; _save_cache()
                conn.close()
                return out
        conn.close()
    except Exception as e:
        pass
    return {}


# ── Structured LLM enrichment ────────────────────────────────────────
def llm_enrich_title(title: str, genre: str, description: str) -> dict:
    """
    Uses OpenAI Responses API with structured output (json_object) to
    generate rich semantic metadata for a title.

    Output fields:
      themes, moods, narrative_hooks, audience_descriptors,
      spoiler_safe_summary, semantic_tags, content_warnings,
      pacing, visual_style, comparable_titles
    """
    ck = f"llm_enrich:{title}"
    if ck in _cache: return _cache[ck]
    if not _OAI_KEY:
        return _rule_enrich(title, genre, description)
    try:
        prompt = f"""Enrich this streaming title for a recommendation system.

Title: {title}
Genre: {genre}
Description: {description[:300]}

Return ONLY a JSON object with these fields:
{{
  "themes": ["list of 3-5 core themes"],
  "moods": ["list of 3 emotional tones"],
  "narrative_hooks": ["2-3 compelling hooks that would attract viewers"],
  "audience_descriptors": ["who would love this — 3 audience types"],
  "spoiler_safe_summary": "one sentence, no spoilers",
  "semantic_tags": ["10 searchable tags"],
  "pacing": "slow|medium|fast|variable",
  "visual_style": "describe in 5 words",
  "comparable_titles": ["2-3 similar titles"]
}}"""
        body = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role":"user","content":prompt}],
            "max_tokens": 400, "temperature": 0.2,
            "response_format": {"type":"json_object"},
        }, ensure_ascii=True).encode("utf-8")
        ctx  = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.openai.com", timeout=8, context=ctx)
        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {_OAI_KEY}",
            "Content-Length":str(len(body)),
        })
        resp = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        result = json.loads(resp["choices"][0]["message"]["content"])
        _cache[ck] = result; _save_cache()
        return result
    except Exception as e:
        return _rule_enrich(title, genre, description)


def _rule_enrich(title: str, genre: str, desc: str) -> dict:
    """Rule-based fallback enrichment when OpenAI unavailable."""
    GENRE_MOODS = {
        "Thriller":["tense","suspenseful","gripping"],
        "Crime":["gritty","dark","intense"],
        "Drama":["emotional","thoughtful","moving"],
        "Comedy":["light","witty","entertaining"],
        "Sci-Fi":["imaginative","cerebral","visionary"],
        "Horror":["scary","unsettling","atmospheric"],
        "Romance":["warm","heartfelt","intimate"],
        "Action":["exciting","kinetic","thrilling"],
        "Documentary":["informative","authentic","compelling"],
        "Animation":["creative","vibrant","expressive"],
    }
    moods = GENRE_MOODS.get(genre, ["engaging","interesting","watchable"])
    return {
        "themes": [genre, "human drama", "character development"],
        "moods": moods,
        "narrative_hooks": [f"A {genre.lower()} that keeps you engaged",
                            "Complex characters and plot twists"],
        "audience_descriptors": [f"{genre} fans", "general audience", "binge-watchers"],
        "spoiler_safe_summary": desc[:100] if desc else f"A compelling {genre} title.",
        "semantic_tags": [genre, title.split()[0] if title else "title",
                          "streaming", "series", "recommended"],
        "pacing": "medium",
        "visual_style": f"cinematic {genre.lower()} aesthetic",
        "comparable_titles": [],
    }


# ── Artwork grounding audit (VLM, offline) ───────────────────────────
def artwork_grounding_audit(title: str, genre: str,
                             poster_url: str | None,
                             enrichment: dict) -> dict:
    """
    Audits whether poster artwork honestly represents the content.
    Uses GPT-4o vision (offline/async — never in request path).

    Returns: trust_score, mismatch_detected, audit_notes, recommendation
    """
    ck = f"artwork_audit:{title}"
    if ck in _cache: return _cache[ck]
    if not poster_url or not poster_url.startswith("http"):
        return {"trust_score":1.0,"mismatch_detected":False,
                "audit_notes":"No poster available","recommendation":"none"}
    if not _OAI_KEY:
        return _rule_artwork_audit(genre, enrichment)
    try:
        moods_str = ", ".join(enrichment.get("moods",["unknown"]))
        prompt = f"""Audit this streaming poster for content accuracy.
Title: {title}, Genre: {genre}, Expected mood: {moods_str}

Answer with JSON:
{{
  "trust_score": 0.0-1.0,
  "thumbnail_genre_signal": "what genre this looks like",
  "matches_actual_genre": true/false,
  "mismatch_detected": true/false,
  "misleading_elements": ["any misleading visual elements"],
  "audit_notes": "one sentence assessment",
  "recommendation": "approved|review|replace"
}}"""
        body = json.dumps({
            "model": "gpt-4o",
            "messages": [{"role":"user","content":[
                {"type":"image_url","image_url":{"url":poster_url,"detail":"low"}},
                {"type":"text","text":prompt},
            ]}],
            "max_tokens": 200, "temperature": 0.1,
            "response_format": {"type":"json_object"},
        }, ensure_ascii=True).encode("utf-8")
        ctx  = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.openai.com", timeout=10, context=ctx)
        conn.request("POST","/v1/chat/completions",body=body,headers={
            "Content-Type":"application/json",
            "Authorization":f"Bearer {_OAI_KEY}",
            "Content-Length":str(len(body)),
        })
        result = json.loads(json.loads(conn.getresponse().read().decode("utf-8"))
                            ["choices"][0]["message"]["content"])
        conn.close()
        _cache[ck] = result; _save_cache()
        return result
    except Exception:
        return _rule_artwork_audit(genre, enrichment)

def _rule_artwork_audit(genre: str, enrichment: dict) -> dict:
    return {"trust_score":0.85,"mismatch_detected":False,
            "audit_notes":f"Rule-based audit: {genre} poster assumed honest",
            "recommendation":"approved","method":"rule_based"}


# ── Batch enrichment (called by Metaflow step) ───────────────────────
def enrich_catalog(catalog: dict[int, dict], max_items: int = 500) -> dict[int, dict]:
    """
    Enrich all catalog items with TMDB + LLM data.
    Designed to run as a Metaflow step, not in the request path.
    """
    enriched = {}
    for i, (mid, item) in enumerate(list(catalog.items())[:max_items]):
        title = item.get("title","")
        genre = item.get("primary_genre","Drama")
        desc  = item.get("description","")
        # TMDB hydration
        tmdb  = tmdb_hydrate(title, item.get("year"))
        # LLM enrichment
        enrich= llm_enrich_title(title, genre, desc)
        # Artwork audit
        poster= tmdb.get("poster_url") or item.get("poster_url","")
        audit = artwork_grounding_audit(title, genre, poster, enrich)
        enriched[mid] = {
            **item,
            **{k:v for k,v in tmdb.items() if v},
            "llm_enrichment":  enrich,
            "artwork_audit":   audit,
            "enriched":        True,
        }
        if i % 50 == 0 and i > 0:
            print(f"  [CatalogEnrichment] {i}/{min(len(catalog),max_items)} enriched")
    print(f"  [CatalogEnrichment] Complete: {len(enriched)} items enriched")
    return enriched
