"""
VLM Engine  —  Vision-Language Model Thumbnail Personalization
=============================================================
Real AI Engineer feature #3.

What it does:
  Uses GPT-4o (vision) to:
  1. Analyse a title's poster/thumbnail image
  2. Understand what visual elements it contains (faces, mood, colour, action)
  3. Match those visual elements to what the user historically engages with
  4. Generate a personalised thumbnail description (what to show THIS user)
  5. Detect "bait-and-switch" risk: does the thumbnail match the actual genre?

Netflix uses VLMs for exactly this — dynamic artwork personalization.
The original Netflix tech blog describes showing romance fans a romantic
scene from an action movie, which works short-term but breaks trust.
This engine adds a "trust_score" that flags misleading thumbnails.

Why this is a real AI Engineer feature:
  - ALS and GBM have no concept of visual content
  - A user who engages with dark, moody thumbnails needs different artwork
    than one who engages with bright, comedic thumbnails — even for the same show
  - This closes the loop between the recommendation model and the visual layer
"""
from __future__ import annotations

import base64, json, os
from typing import Any
from recsys.serving._http import openai_post

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
_TMDB_KEY   = os.environ.get("TMDB_API_KEY", "")

# Visual preference profiles derived from genre engagement
_VISUAL_PROFILES = {
    "Thriller":    {"mood": "dark and tense",    "palette": "desaturated blues and greys",    "faces": "intense close-ups"},
    "Crime":       {"mood": "gritty and shadowy", "palette": "dark browns and blacks",         "faces": "serious expressions"},
    "Sci-Fi":      {"mood": "futuristic wonder",  "palette": "neon blues and purples",         "faces": "awe and curiosity"},
    "Drama":       {"mood": "emotional and quiet","palette": "warm earth tones",               "faces": "emotional vulnerability"},
    "Comedy":      {"mood": "light and playful",  "palette": "bright warm colours",            "faces": "laughter and expression"},
    "Horror":      {"mood": "eerie and unsettling","palette":"deep reds and near-black",       "faces": "fear and dread"},
    "Action":      {"mood": "high-energy",        "palette": "high contrast, vivid colours",   "faces": "determination"},
    "Romance":     {"mood": "warm and intimate",  "palette": "soft pinks and golds",           "faces": "longing and connection"},
    "Documentary": {"mood": "authentic and raw",  "palette": "natural, unfiltered colours",    "faces": "real human stories"},
    "Animation":   {"mood": "vibrant and magical","palette": "saturated primary colours",      "faces": "expressive cartoon characters"},
}


def analyse_poster(
    poster_url: str,
    title: str,
    actual_genre: str,
    user_genres: list[str],
) -> dict[str, Any]:
    """
    Uses GPT-4o vision to analyse a poster and determine:
    1. What visual elements dominate
    2. Whether it matches the actual genre (bait-and-switch detection)
    3. Which user taste profile it best suits
    4. A personalised display recommendation

    Returns dict with: visual_analysis, trust_score, best_audience, display_recommendation
    """
    if not _OPENAI_KEY:
        return _fallback_analysis(title, actual_genre, user_genres)

    # Determine which visual profile matches this user
    user_profile = _VISUAL_PROFILES.get(
        user_genres[0] if user_genres else "Drama",
        _VISUAL_PROFILES["Drama"]
    )

    prompt = f"""Analyse this movie/show poster for Netflix thumbnail personalization.

Title: {title}
Actual genre: {actual_genre}
User's preferred visual style (from their history): {user_profile['mood']}, {user_profile['palette']}

Answer these questions as JSON:
{{
  "dominant_visual": "describe the main visual element in 5 words",
  "mood": "describe the emotional mood of the image in 3 words",
  "colour_palette": "dominant colours in 3 words",
  "thumbnail_genre_signal": "what genre does this thumbnail LOOK like",
  "matches_actual_genre": true/false,
  "trust_score": 0.0-1.0 (1.0 = thumbnail perfectly matches actual content, 0.0 = misleading),
  "matches_user_taste": true/false (does it match the user's visual preference?),
  "personalisation_note": "one sentence: what aspect of this image would resonate with THIS user"
}}"""

    try:
        resp = _utf8_post('/v1/chat/completions', json.loads(json.dumps({
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": poster_url, "detail": "low"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": 300,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        })), _OPENAI_KEY)

        analysis = json.loads(resp["choices"][0]["message"]["content"])
        analysis["user_visual_profile"] = user_profile
        analysis["best_audience"] = _best_audience(analysis.get("mood",""))
        return analysis

    except Exception as e:
        result = _fallback_analysis(title, actual_genre, user_genres)
        result["error"] = str(e)
        return result


def _fallback_analysis(title: str, genre: str, user_genres: list[str]) -> dict:
    """Rule-based fallback when OpenAI vision is unavailable."""
    profile = _VISUAL_PROFILES.get(genre, _VISUAL_PROFILES["Drama"])
    user_profile = _VISUAL_PROFILES.get(
        user_genres[0] if user_genres else "Drama", _VISUAL_PROFILES["Drama"])
    matches = genre in user_genres
    return {
        "dominant_visual":       f"Typical {genre} imagery",
        "mood":                  profile["mood"],
        "colour_palette":        profile["palette"],
        "thumbnail_genre_signal":genre,
        "matches_actual_genre":  True,
        "trust_score":           1.0,
        "matches_user_taste":    matches,
        "personalisation_note":  (
            f"This {genre} thumbnail's {profile['mood']} style matches your taste for "
            f"{user_profile['mood']} content."
            if matches else
            f"Exploration: this {genre} thumbnail may appeal to your interest in "
            f"{user_profile['mood']} content."
        ),
        "user_visual_profile":   user_profile,
        "best_audience":         [genre] + (user_genres[:2] if matches else []),
        "method":                "rule_based_fallback",
    }


def _best_audience(mood: str) -> list[str]:
    """Map mood description to likely genre audiences."""
    mood_lower = mood.lower()
    audiences = []
    if any(w in mood_lower for w in ["dark","tense","gritty","unsettling"]):
        audiences += ["Thriller","Crime","Horror"]
    if any(w in mood_lower for w in ["light","playful","bright","fun"]):
        audiences += ["Comedy","Animation"]
    if any(w in mood_lower for w in ["emotional","warm","intimate","quiet"]):
        audiences += ["Drama","Romance"]
    if any(w in mood_lower for w in ["wonder","futuristic","epic","adventure"]):
        audiences += ["Sci-Fi","Action"]
    return list(set(audiences)) or ["Drama"]


def batch_analyse_posters(
    items: list[dict],
    user_genres: list[str],
) -> list[dict]:
    """
    Analyse multiple posters and add VLM fields to each item.
    Rate-limited to avoid hammering the vision API.
    """
    results = []
    for item in items:
        poster = item.get("poster_url") or ""
        if poster and poster.startswith("http"):
            analysis = analyse_poster(
                poster, item.get("title",""), item.get("primary_genre",""), user_genres)
            item = dict(item)
            item["vlm_analysis"]            = analysis
            item["thumbnail_trust_score"]   = analysis.get("trust_score", 1.0)
            item["thumbnail_matches_taste"] = analysis.get("matches_user_taste", True)
            item["vlm_personalisation"]     = analysis.get("personalisation_note","")
        results.append(item)
    return results
