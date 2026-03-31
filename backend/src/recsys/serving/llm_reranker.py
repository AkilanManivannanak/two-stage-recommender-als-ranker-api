"""
LLM Re-Ranker  —  Language Model as a Ranking Layer
====================================================
Real AI Engineer feature #2.

What it does:
  Uses GPT-4o-mini as a SECOND-STAGE re-ranker sitting on top of ALS + GBM.
  Instead of just scoring by feature weights, the LLM can:
  1. Understand CONTEXT: "User asked to watch something tonight with family"
  2. Apply REASONING: "They haven't watched a documentary in 6 months — time to suggest one"
  3. Handle NUANCE: "They rated Ozark 5 stars but said they find crime shows 'stressful' — avoid Crime"
  4. Session INTENT: blend current session context into ranking decisions

Architecture position:
  ALS (retrieval, top-500)
    → GBM ranker (top-50)
      → LLM re-ranker (top-20) ← THIS FILE
        → Diversity optimizer (final-20)

Why this is a real Netflix pattern:
  Netflix's foundation model work (2024-2026 tech blog) describes exactly this:
  large models doing final-stage reasoning over pre-filtered candidates.
  The LLM doesn't score 500,000 items — it reasons over 20-50 pre-filtered ones.

Cost:
  ~50 tokens in, ~200 tokens out = ~$0.00004 per request at gpt-4o-mini pricing.
  At 1M requests/day that's $40/day — tractable for a major feature.
"""
from __future__ import annotations

import json, os
from typing import Any
from recsys.serving._http import openai_post

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")


def llm_rerank_with_context(
    user_id: int,
    candidates: list[dict],
    user_history_titles: list[str],
    user_genres: list[str],
    session_context: str = "",
    time_of_day: str = "evening",
    top_k: int = 20,
) -> list[dict]:
    """
    LLM re-ranks candidates using chain-of-thought reasoning.

    Inputs:
      candidates          — pre-filtered by ALS + GBM (20-50 items)
      user_history_titles — titles user has watched (last 10)
      user_genres         — inferred genre preferences
      session_context     — optional free-text context ("watching with kids")
      time_of_day         — influences tone (morning=light, evening=heavy drama ok)

    Returns candidates with added fields:
      llm_rank, llm_score, llm_reasoning
    """
    if not _OPENAI_KEY:
        return _no_llm_fallback(candidates, top_k)

    cand_list = "\n".join(
        f"{i+1}. [{c.get('primary_genre','')}] {c.get('title','')} "
        f"(rated {c.get('avg_rating',3.5):.1f}, {c.get('year','')}) — {c.get('description','')[:80]}"
        for i, c in enumerate(candidates[:20])
    )

    context_block = f"Session context: {session_context}\n" if session_context else ""
    time_note = ("prefer lighter, upbeat content" if time_of_day in ["morning","afternoon"]
                 else "deeper drama and thrillers are suitable")

    prompt = f"""You are a Netflix recommendation AI performing final re-ranking.

User #{user_id} watch history (recent): {", ".join(user_history_titles[:6]) or "new user"}
Inferred taste profile: {", ".join(user_genres[:4])}
{context_block}Time context: {time_of_day} ({time_note})

Pre-filtered candidates:
{cand_list}

Re-rank these {len(candidates[:20])} titles for this specific user right now.
Consider: taste match, variety, appropriate for time of day, session context.

Return ONLY valid JSON:
{{"rankings": [{{"rank": 1, "title": "exact title", "score": 0.95, "reasoning": "15-word explanation"}}, ...]}}

Include all {min(len(candidates),20)} titles. Score from 0.0 to 1.0."""

    try:
        resp = _utf8_post('/v1/chat/completions', json.loads(json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        })), _OPENAI_KEY)

        raw  = resp["choices"][0]["message"]["content"]
        data = json.loads(raw)
        ranked = data.get("rankings", data.get("results", []))

        title_map = {c["title"]: c for c in candidates}
        result = []
        for item in ranked:
            t = item.get("title","")
            c = title_map.get(t)
            if c:
                c = dict(c)
                c["llm_rank"]      = item.get("rank", len(result)+1)
                c["llm_score"]     = float(item.get("score", 0.5))
                c["llm_reasoning"] = item.get("reasoning","")
                result.append(c)

        # Fallback: append any candidates the LLM missed
        seen = {c["title"] for c in result}
        for c in candidates:
            if c["title"] not in seen:
                c = dict(c)
                c["llm_rank"] = len(result)+1; c["llm_score"] = 0.3; c["llm_reasoning"] = ""
                result.append(c)

        return result[:top_k]

    except Exception as e:
        print(f"  [LLM Re-ranker] Error: {e}")
        return _no_llm_fallback(candidates, top_k, error=str(e))


def _no_llm_fallback(candidates: list[dict], top_k: int, error: str = "") -> list[dict]:
    """Return candidates unchanged when LLM is unavailable."""
    for i, c in enumerate(candidates[:top_k]):
        c = dict(c)
        c["llm_rank"]      = i + 1
        c["llm_score"]     = round(c.get("final_score", c.get("ranker_score", 0.5)), 4)
        c["llm_reasoning"] = f"LLM unavailable{': '+error if error else ''}. Using GBM score."
        candidates[i] = c
    return candidates[:top_k]


def generate_row_narrative(
    row_name: str,
    items: list[dict],
    user_genres: list[str],
    user_id: int,
) -> str:
    """
    Uses GPT-4o-mini to write a personalised row title for the homepage.
    E.g. instead of "Because you watched Ozark" it generates
    "High-Stakes Crime Dramas That'll Keep You Up"
    """
    if not _OPENAI_KEY or not items:
        return row_name

    genres_in_row = list(set(i.get("primary_genre","") for i in items[:5]))
    titles_in_row = [i.get("title","") for i in items[:4]]

    prompt = (
        f"Write a compelling Netflix row title (max 6 words) for a row containing: "
        f"{', '.join(titles_in_row)}. User likes: {', '.join(user_genres[:3])}. "
        f"Genres in row: {', '.join(genres_in_row)}. "
        f"Make it feel personalised, not generic. Return ONLY the title text, nothing else."
    )
    try:
        resp = _utf8_post('/v1/chat/completions', json.loads(json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role":"user","content":prompt}],
            "max_tokens": 20,
            "temperature": 0.7,
        })), _OPENAI_KEY)
        return resp["choices"][0]["message"]["content"].strip().strip('"')
    except Exception:
        return row_name
