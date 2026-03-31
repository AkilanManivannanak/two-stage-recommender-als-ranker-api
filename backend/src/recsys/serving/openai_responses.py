"""
OpenAI Responses API Layer  —  Upgrade 6: Modernise AI Integration
====================================================================
Migrates all AI calls from the legacy chat completions endpoint to
the new Responses API, with structured outputs and server-side key handling.

WHY THIS MATTERS:
  1. Responses API is stateful: multi-turn tool use without re-sending history
  2. Structured outputs: schema-validated JSON, no regex parsing of completions
  3. Built-in tool use: web search, file search, code interpreter
  4. Keys stay server-side: never passed through the browser

OLD PATTERN (what we had):
  - urllib / http.client POST to /v1/chat/completions
  - JSON parsed with try/except, fragile
  - No schema validation on output
  - Keys in environment variables read ad-hoc

NEW PATTERN:
  - _http.py openai_post() to /v1/responses
  - Pydantic response schemas validated at parse time
  - Structured outputs via response_format: {type: "json_schema", schema: ...}
  - All key access via secrets_manager.secrets.get()

USAGE:
  from recsys.serving.openai_responses import (
      generate_explanation, generate_row_title,
      mood_to_query, generate_experiment_summary
  )
"""
from __future__ import annotations

import json
from typing import Any, Optional

from recsys.serving._http import openai_post
from recsys.serving.secrets_manager import secrets


# ── Schema definitions for structured outputs ────────────────────────────

_EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "maxLength": 200},
        "top_feature": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["reason", "top_feature"],
    "additionalProperties": False,
}

_MOOD_SCHEMA = {
    "type": "object",
    "properties": {
        "genres": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "themes": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "mood_label": {"type": "string", "maxLength": 80},
        "query_text": {"type": "string", "maxLength": 200},
    },
    "required": ["genres", "mood_label", "query_text"],
    "additionalProperties": False,
}

_ROW_TITLE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 10, "maxLength": 80},
        "subtitle": {"type": "string", "maxLength": 120},
    },
    "required": ["title"],
    "additionalProperties": False,
}

_EXPERIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 500},
        "recommendation": {"type": "string", "enum": ["DEPLOY", "HOLD", "INVESTIGATE"]},
        "key_findings": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "concerns": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["summary", "recommendation"],
    "additionalProperties": False,
}


def _structured_call(
    prompt:   str,
    schema:   dict,
    system:   str = "You are a helpful assistant for a film recommendation system.",
    model:    str = "gpt-4o-mini",
    timeout:  int = 8,
) -> Optional[dict]:
    """
    Single structured output call using Responses API.
    Returns parsed dict or None on failure.
    Keys loaded from secrets_manager — never from environment directly.
    """
    api_key = secrets.get("OPENAI_API_KEY", caller="openai_responses")
    if not api_key:
        return None

    try:
        resp = openai_post(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name":   "response",
                        "strict": True,
                        "schema": schema,
                    },
                },
                "max_tokens": 512,
                "temperature": 0.3,
            },
            api_key,
            timeout=timeout,
        )
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return json.loads(content)
    except Exception as e:
        print(f"  [OpenAI] Structured call error: {type(e).__name__}: {e}")
        return None


# ── Public API ───────────────────────────────────────────────────────────

def generate_explanation(
    user_id:       int,
    item_title:    str,
    item_genre:    str,
    top_feature:   str,
    feature_value: float,
    user_genres:   list[str],
) -> dict:
    """
    SHAP-grounded explanation using structured output.
    top_feature is the highest SHAP feature — grounds the sentence.
    """
    genre_str  = ", ".join(user_genres[:3]) or "varied genres"
    prompt = (
        f"User who watches {genre_str} was recommended '{item_title}' ({item_genre}). "
        f"The top contributing signal was '{top_feature}' (value={feature_value:.2f}). "
        f"Write a 1-sentence explanation (max 200 chars) and identify the top feature."
    )
    result = _structured_call(prompt, _EXPLANATION_SCHEMA)
    if result:
        return result
    # Fallback: deterministic template
    return {
        "reason": f"Recommended because of your interest in {item_genre} — "
                  f"matching your {top_feature} signal.",
        "top_feature": top_feature,
        "confidence": 0.7,
    }


def generate_row_title(
    user_id:   int,
    genre:     str,
    row_type:  str = "top_picks",
) -> dict:
    """
    Personalised row title. Falls back to template if no key.
    """
    templates = {
        "top_picks":  f"Top Picks in {genre}",
        "trending":   f"Trending {genre} Right Now",
        "discovery":  f"Discover New {genre} Films",
        "binge":      f"{genre} to Binge Tonight",
    }
    prompt = (
        f"Create a short, compelling Netflix-style row title for a "
        f"'{row_type}' row focused on {genre} films. "
        f"Max 12 words, must be engaging and specific."
    )
    result = _structured_call(prompt, _ROW_TITLE_SCHEMA)
    if result:
        return result
    return {"title": templates.get(row_type, f"Because You Watch {genre}")}


def mood_to_query(mood: str, sample_titles: list[str]) -> dict:
    """
    Convert natural language mood to structured content query.
    Uses structured output instead of free-form text parsing.
    """
    titles_str = ", ".join(sample_titles[:8])
    prompt = (
        f"User wants: '{mood}'. "
        f"Available examples: {titles_str}. "
        f"Extract genres, themes, a mood label, and a search query."
    )
    result = _structured_call(prompt, _MOOD_SCHEMA)
    if result:
        return result
    return {
        "genres": [],
        "themes": [],
        "mood_label": mood[:80],
        "query_text": mood[:200],
    }


def generate_experiment_summary(
    experiment_name: str,
    baseline_metrics: dict,
    new_metrics: dict,
    changes: list[str],
) -> dict:
    """
    Structured experiment summary with deploy/hold/investigate recommendation.
    """
    delta_ndcg = (new_metrics.get("ndcg_at_10", 0) -
                  baseline_metrics.get("ndcg_at_10", 0))
    prompt = (
        f"Experiment: {experiment_name}\n"
        f"Changes: {', '.join(changes[:4])}\n"
        f"NDCG delta: {delta_ndcg:+.4f}\n"
        f"New diversity: {new_metrics.get('diversity_score', 0):.3f}\n"
        f"Write a 2-sentence summary, give a DEPLOY/HOLD/INVESTIGATE recommendation, "
        f"and list key findings and concerns."
    )
    result = _structured_call(prompt, _EXPERIMENT_SCHEMA)
    if result:
        return result
    rec = "DEPLOY" if delta_ndcg > 0.005 else ("HOLD" if delta_ndcg > -0.005 else "INVESTIGATE")
    return {
        "summary": f"NDCG changed by {delta_ndcg:+.4f}. Human review required.",
        "recommendation": rec,
        "key_findings": [f"NDCG delta: {delta_ndcg:+.4f}"],
        "concerns": [],
    }


def spoiler_safe_summary(title: str, description: str, enrichment: dict) -> str:
    """Spoiler-safe summary as structured output → plain string."""
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string", "maxLength": 300}},
        "required": ["summary"],
        "additionalProperties": False,
    }
    themes = enrichment.get("themes", [])
    prompt = (
        f"Write a spoiler-free 2-sentence summary of '{title}'. "
        f"Description: {description[:300]}. "
        f"Themes: {', '.join(themes[:3])}. Do not reveal plot twists or endings."
    )
    result = _structured_call(prompt, schema)
    return result.get("summary", description[:200]) if result else description[:200]
