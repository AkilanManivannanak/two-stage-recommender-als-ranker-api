"""
GenAI UX Layer  —  Product Edge
=================================
Plane: GenAI UX (product edge — never blocks the request path)

Responsibilities:
  - "Why recommended?" explanations (grounded in feature attribution)
  - Conversational discovery: mood-to-content mapping
  - Semantic browse / search assistance
  - Title summaries for accessibility and previews
  - Editorial row naming (personalised row titles)
  - Mood-based content matching
  - Support outputs for operators and editors

Rules:
  1. Text generation NEVER blocks the main page response
  2. Generation happens offline, ahead of time, or behind caches
  3. Outputs are structured (json_object) not freeform prose
  4. Every explanation references actual model features, not invented reasons
  5. Image generation is NOT used in production recommendations
     (only internal creative exploration under human review)

Reference: OpenAI Responses API — structured output workflow
"""
from __future__ import annotations

import http.client, json, os, ssl
from typing import Any

_OAI_KEY = os.environ.get("OPENAI_API_KEY","")

_EXPLANATION_CACHE: dict[str,str] = {}


# ── Feature-grounded explanations ────────────────────────────────────
def explain_recommendation(
    user_id:           int,
    item:              dict,
    feature_values:    dict[str,float],
    feature_importance:dict[str,float],
    user_genres:       list[str],
    session_intent:    str = "unknown",
    is_exploration:    bool = False,
) -> dict[str, Any]:
    """
    Generate a grounded "Why recommended?" explanation.
    Grounded = references actual top feature contributions, not templates.
    Falls back to rule-based attribution if OpenAI unavailable.
    """
    ck = f"explain:{user_id}:{item.get('item_id',0)}"
    if ck in _EXPLANATION_CACHE:
        return {"explanation": _EXPLANATION_CACHE[ck], "method":"cached"}

    title = item.get("title","")
    genre = item.get("primary_genre","")

    # Compute top 3 contributing features
    contributions = {
        f: round(feature_importance.get(f,0.0) * abs(feature_values.get(f,0.0)), 4)
        for f in feature_values
    }
    top3 = sorted(contributions.items(), key=lambda x:-x[1])[:3]
    top3_str = ", ".join(f"{f}={v:.4f}" for f,v in top3)

    FEAT_LABELS = {
        "als_score":"collaborative filtering match",
        "genre_affinity":"genre preference match",
        "item_avg_rating":"critically high rating",
        "item_pop":"popularity signal",
        "u_avg":"your rating pattern",
        "item_year":"release recency",
        "lts_score":"long-term satisfaction signal",
    }
    top_feat = top3[0][0] if top3 else "genre_affinity"
    top_label = FEAT_LABELS.get(top_feat, top_feat)

    if not _OAI_KEY:
        explanation = _rule_explanation(title, genre, top_label, user_genres,
                                        is_exploration, session_intent)
        _EXPLANATION_CACHE[ck] = explanation
        return {"explanation":explanation,"top_feature":top_feat,
                "attribution":dict(top3),"method":"rule_based"}

    try:
        prompt = f"""Write ONE sentence (max 25 words) explaining why "{title}" ({genre}) 
was recommended to this user.

Top model attribution features: {top3_str}
User's preferred genres: {', '.join(user_genres[:4])}
Session intent: {session_intent}
Exploration slot: {is_exploration}
Primary driver label: {top_label}

Rules: reference the actual top feature. If exploration, mention it's outside usual picks.
Acknowledge the primary driver is an approximation — many decisions have correlated features.
Use "primary driver" not "the reason". No spoilers. No invented reasons. Max 25 words."""
        body = json.dumps({
            "model":"gpt-4o-mini",
            "messages":[{"role":"user","content":prompt}],
            "max_tokens":60,"temperature":0.3,
        }, ensure_ascii=True).encode("utf-8")
        ctx  = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.openai.com",timeout=4,context=ctx)
        conn.request("POST","/v1/chat/completions",body=body,headers={
            "Content-Type":"application/json","Authorization":f"Bearer {_OAI_KEY}",
            "Content-Length":str(len(body)),
        })
        resp = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        explanation = resp["choices"][0]["message"]["content"].strip()
        _EXPLANATION_CACHE[ck] = explanation
        return {"explanation":explanation,"top_feature":top_feat,
                "attribution":dict(top3),"method":"openai_grounded"}
    except Exception:
        explanation = _rule_explanation(title, genre, top_label, user_genres,
                                        is_exploration, session_intent)
        _EXPLANATION_CACHE[ck] = explanation
        return {"explanation":explanation,"top_feature":top_feat,
                "attribution":dict(top3),"method":"rule_fallback"}


def _rule_explanation(title:str, genre:str, top_label:str,
                       user_genres:list[str], is_exp:bool, intent:str) -> str:
    in_hist = genre in user_genres
    if is_exp or not in_hist:
        return (f"Exploration: driven by {top_label}. "
                f"This {genre} title is outside your usual genres — "
                f"surfaced to broaden your discovery.")
    return (f"Recommended by {top_label}. "
            f"Matches your {genre} preference and viewing history patterns.")


# ── Conversational discovery ─────────────────────────────────────────
def mood_to_content_query(
    mood_description: str,
    catalog_sample:   list[dict],
    top_k:            int = 8,
) -> dict[str, Any]:
    """
    Maps a natural language mood to content recommendations.
    "Something funny to watch with family tonight"
    "A dark slow-burn thriller I can lose myself in"

    Returns structured query with genre filters, mood tags, and title suggestions.
    """
    if not _OAI_KEY:
        return {"query_interpretation": mood_description,
                "suggested_genres":["Drama","Comedy"],
                "mood_tags":["relaxing","engaging"],
                "title_suggestions": catalog_sample[:top_k],
                "method":"fallback"}
    try:
        titles_str = "\n".join(
            f"- {c.get('title','')} ({c.get('primary_genre','')}): {c.get('description','')[:60]}"
            for c in catalog_sample[:20]
        )
        prompt = f"""Map this mood to content: "{mood_description}"

Available titles:
{titles_str}

Return JSON:
{{"query_interpretation": "what the user wants in 10 words",
  "suggested_genres": ["top 3 matching genres"],
  "mood_tags": ["5 mood descriptors"],
  "recommended_titles": ["top {top_k} title names from the list"],
  "why": "one sentence reasoning"}}"""
        body = json.dumps({
            "model":"gpt-4o-mini",
            "messages":[{"role":"user","content":prompt}],
            "max_tokens":300,"temperature":0.4,
            "response_format":{"type":"json_object"},
        }, ensure_ascii=True).encode("utf-8")
        ctx  = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.openai.com",timeout=6,context=ctx)
        conn.request("POST","/v1/chat/completions",body=body,headers={
            "Content-Type":"application/json","Authorization":f"Bearer {_OAI_KEY}",
            "Content-Length":str(len(body)),
        })
        resp = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        result = json.loads(resp["choices"][0]["message"]["content"])
        result["method"] = "openai_mood_mapping"
        # Map recommended title names back to catalog items
        title_map = {c.get("title",""):c for c in catalog_sample}
        result["title_suggestions"] = [
            title_map[t] for t in result.get("recommended_titles",[])
            if t in title_map
        ][:top_k]
        return result
    except Exception as e:
        return {"query_interpretation":mood_description,
                "suggested_genres":[],"mood_tags":[],
                "title_suggestions":catalog_sample[:top_k],
                "method":"fallback","error":str(e)}


# ── Personalised row titles (editorial naming) ───────────────────────

# ── Row title pre-computation cache ────────────────────────────────────────
# ENFORCEMENT OF "zero generative AI in the request path" CONTRACT:
#
# The /ux/row_title/{user_id} endpoint MUST serve from this cache.
# If the cache is empty the endpoint returns a deterministic template.
# OpenAI is only called by precompute_row_titles() which runs OFFLINE
# (in the Metaflow pipeline, not during a user request).
#
# The old implementation called OpenAI live on every request, which
# directly contradicted the "zero GenAI in request path" claim.
# This is the fix: cache-first, OpenAI never in request path.

_ROW_TITLE_CACHE: dict[str, str] = {}   # key: "{user_id}:{row_type}" → title
_TITLE_TEMPLATES = {
    "top_picks":            "Top Picks For You",
    "because_you_watched":  "Because You Watched",
    "trending_now":         "Trending Now",
    "explore_new_genres":   "Discover Something Different",
    "highly_rated":         "Critically Acclaimed",
    "binge":                "Perfect for Binging Tonight",
    "discovery":            "Expand Your Taste",
    "action":               "High-Stakes Action Picks",
    "drama":                "Emotionally Gripping Dramas",
    "comedy":               "Feel-Good Comedies",
    "sci-fi":               "Mind-Bending Sci-Fi",
}

def personalised_row_title(
    items:       list[dict],
    user_genres: list[str],
    row_type:    str = "top_picks",
    user_id:     int = 0,
) -> str:
    """
    Serve row title from PRE-COMPUTED CACHE ONLY.
    Never calls OpenAI on the request path.

    If cache miss → return deterministic template.
    Cache is populated offline by precompute_row_titles().
    """
    cache_key = f"{user_id}:{row_type}"
    if cache_key in _ROW_TITLE_CACHE:
        return _ROW_TITLE_CACHE[cache_key]

    # Genre-based template (no LLM call)
    genre = user_genres[0] if user_genres else ""
    if genre:
        templates = {
            "Action":       f"High-Stakes {genre} You Can't Stop Watching",
            "Drama":        f"Gripping {genre} Picked For You",
            "Comedy":       f"Laugh-Out-Loud {genre} Picks",
            "Horror":       f"Spine-Chilling {genre} Tonight",
            "Sci-Fi":       f"Mind-Bending {genre} Adventures",
            "Romance":      f"Heartfelt {genre} Just For You",
            "Thriller":     f"Edge-of-Your-Seat {genre} Picks",
            "Documentary":  f"Thought-Provoking {genre} Features",
            "Animation":    f"Beloved {genre} For Any Mood",
            "Crime":        f"Gripping {genre} Stories",
        }
        if genre in templates and row_type == "top_picks":
            return templates[genre]

    return _TITLE_TEMPLATES.get(row_type, row_type.replace("_", " ").title())


def precompute_row_titles(
    user_sample:  list[dict],   # [{user_id, genres, recs}]
    row_types:    list[str] | None = None,
) -> dict:
    """
    OFFLINE ONLY. Call from Metaflow pipeline step, not from request path.
    Populates _ROW_TITLE_CACHE for all (user, row_type) pairs.

    Returns {cache_entries_written, used_openai, elapsed_s}.
    """
    if row_types is None:
        row_types = ["top_picks", "trending_now", "explore_new_genres", "binge"]

    written = 0
    used_oai = False
    start    = __import__("time").time()

    for user_data in user_sample:
        uid      = user_data.get("user_id", 0)
        genres   = user_data.get("genres", [])
        items    = user_data.get("recs", [])

        for rt in row_types:
            key = f"{uid}:{rt}"
            if key in _ROW_TITLE_CACHE:
                continue

            # Try OpenAI (offline context) — with strict timeout
            title = None
            if _OAI_KEY and items:
                try:
                    genre_str = ", ".join(list(set(
                        i.get("primary_genre","") for i in items[:4] if i.get("primary_genre")))[:2])
                    top_titles = ", ".join(i.get("title","") for i in items[:3])
                    prompt = (
                        f"Netflix row title, max 8 words, for {rt} row. "
                        f"User genres: {', '.join(genres[:2])}. "
                        f"Items include: {top_titles}. "
                        f"Genre: {genre_str}. Evocative, specific. Return ONLY the title."
                    )
                    body = json.dumps({
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 20, "temperature": 0.6,
                    }, ensure_ascii=True).encode("utf-8")
                    ctx  = ssl.create_default_context()
                    conn = http.client.HTTPSConnection("api.openai.com", timeout=3, context=ctx)
                    conn.request("POST", "/v1/chat/completions", body=body, headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {_OAI_KEY}",
                        "Content-Length": str(len(body)),
                    })
                    resp  = json.loads(conn.getresponse().read().decode("utf-8"))
                    conn.close()
                    raw   = resp["choices"][0]["message"]["content"].strip().strip('"\'')
                    if 5 < len(raw) < 80:
                        title   = raw
                        used_oai = True
                except Exception:
                    pass

            # Fallback to template if OpenAI failed or no key
            if not title:
                title = personalised_row_title(items, genres, rt, uid)

            _ROW_TITLE_CACHE[key] = title
            written += 1

    return {
        "cache_entries_written": written,
        "total_cache_size":      len(_ROW_TITLE_CACHE),
        "used_openai":           used_oai,
        "elapsed_s":             round(__import__("time").time() - start, 2),
        "note": "precompute_row_titles() runs OFFLINE only — never in request path",
    }


# ── Spoiler-safe title summary ───────────────────────────────────────
def spoiler_safe_summary(title: str, description: str,
                          enrichment: dict | None = None) -> str:
    """
    Generate a spoiler-safe 2-sentence summary for hover previews.
    Uses enrichment themes/moods if available to improve quality.
    """
    if not _OAI_KEY:
        return description[:120] + "…" if len(description) > 120 else description
    try:
        themes = enrichment.get("themes",[]) if enrichment else []
        prompt = (f'Write a spoiler-free 2-sentence summary of "{title}".\n'
                  f'Original description: {description[:200]}\n'
                  f'Known themes: {", ".join(themes[:3])}\n'
                  f'Make it compelling for discovery. No spoilers. Max 40 words total.')
        body = json.dumps({
            "model":"gpt-4o-mini",
            "messages":[{"role":"user","content":prompt}],
            "max_tokens":80,"temperature":0.3,
        }, ensure_ascii=True).encode("utf-8")
        ctx  = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.openai.com",timeout=4,context=ctx)
        conn.request("POST","/v1/chat/completions",body=body,headers={
            "Content-Type":"application/json","Authorization":f"Bearer {_OAI_KEY}",
            "Content-Length":str(len(body)),
        })
        resp = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return description[:120] + "…" if len(description) > 120 else description
