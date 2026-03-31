"""
Voice Tool Executor — CineWave Production v5
=============================================
CORE REDESIGN: Intent-aware genre sampling
- Understands what the user ACTUALLY wants
- For single genre: semantic RAG search
- For multiple genres: separate pool per genre, then interleave
- For mood/vibe: maps mood → genres, then samples each
- For "similar to X": finds X's genres, searches those
- Result: recommendations always match what was asked
"""
from __future__ import annotations
import os
import re

# ── Comprehensive genre keyword detection ─────────────────────────────────────

GENRE_KEYWORDS: dict[str, list[str]] = {
    "action":       ["action", "fight", "combat", "explosive", "battle", "war", "kick", "punch", "shoot"],
    "comedy":       ["comedy", "funny", "hilarious", "laugh", "humor", "comedies", "feel good", "feel-good", "lighthearted", "silly", "fun"],
    "drama":        ["drama", "dramatic", "emotional", "powerful", "moving", "touching", "gripping", "intense story"],
    "horror":       ["horror", "scary", "terrifying", "creepy", "spooky", "haunted", "zombie", "ghost", "monster", "fear"],
    "sci-fi":       ["sci-fi", "science fiction", "scifi", "space", "alien", "futuristic", "dystopian", "cyberpunk", "mind-bending", "mind bending", "robot", "ai", "time travel"],
    "romance":      ["romance", "romantic", "love", "love story", "romcom", "rom-com", "heartfelt", "relationship", "dating", "kiss"],
    "thriller":     ["thriller", "suspense", "suspenseful", "tense", "dark", "psychological", "mystery", "twist", "edge-of-seat", "serial killer", "detective"],
    "documentary":  ["documentary", "true story", "real life", "nature", "history", "docuseries", "based on true", "real events", "historical"],
    "animation":    ["animation", "animated", "cartoon", "anime", "pixar", "disney", "dreamworks"],
    "crime":        ["crime", "heist", "murder", "detective", "police", "gangster", "mafia", "cartel", "mob", "law"],
    "adventure":    ["adventure", "quest", "journey", "explore", "treasure", "epic", "expedition"],
    "fantasy":      ["fantasy", "magic", "magical", "mythical", "dragon", "medieval", "wizard", "witch", "supernatural"],
    "family":       ["family", "kids", "children", "child-friendly", "wholesome", "for all ages"],
    "war":          ["war", "military", "soldier", "army", "battle", "ww2", "world war"],
    "western":      ["western", "cowboy", "wild west", "gunfight", "sheriff"],
    "music":        ["music", "musical", "concert", "band", "singer", "song"],
    "history":      ["history", "historical", "period", "ancient", "biography", "biopic"],
    "mystery":      ["mystery", "whodunit", "clue", "solve", "investigation", "puzzle"],
}

MOOD_TO_GENRES: dict[str, list[str]] = {
    "feel good":       ["comedy", "romance", "family"],
    "feel-good":       ["comedy", "romance", "family"],
    "uplifting":       ["comedy", "romance", "drama"],
    "dark":            ["thriller", "crime", "horror"],
    "intense":         ["thriller", "action", "drama"],
    "relaxing":        ["comedy", "romance", "documentary"],
    "scary":           ["horror", "thriller"],
    "exciting":        ["action", "thriller", "adventure"],
    "emotional":       ["drama", "romance"],
    "mind-bending":    ["sci-fi", "thriller"],
    "heartwarming":    ["romance", "comedy", "drama", "family"],
    "gritty":          ["crime", "thriller", "drama"],
    "fun":             ["comedy", "action", "animation", "adventure"],
    "thought-provoking": ["sci-fi", "drama", "documentary"],
    "romantic":        ["romance"],
    "suspenseful":     ["thriller", "crime", "mystery"],
    "inspiring":       ["drama", "documentary"],
    "sad":             ["drama", "romance"],
    "happy":           ["comedy", "family", "animation"],
    "binge":           ["drama", "thriller", "crime"],
    "chill":           ["comedy", "romance", "documentary"],
    "violent":         ["action", "crime", "war"],
    "philosophical":   ["sci-fi", "drama", "documentary"],
    "nostalgic":       ["drama", "romance", "comedy"],
    "magical":         ["fantasy", "animation", "family"],
}

TITLE_GENRES: dict[str, list[str]] = {
    "stranger things":     ["sci-fi", "horror", "thriller"],
    "dark":                ["sci-fi", "thriller"],
    "breaking bad":        ["crime", "drama", "thriller"],
    "game of thrones":     ["fantasy", "drama", "action"],
    "black mirror":        ["sci-fi", "thriller"],
    "ozark":               ["crime", "thriller", "drama"],
    "narcos":              ["crime", "drama"],
    "money heist":         ["crime", "thriller", "action"],
    "squid game":          ["thriller", "drama", "action"],
    "wednesday":           ["horror", "comedy"],
    "bridgerton":          ["romance", "drama"],
    "the crown":           ["drama", "history"],
    "peaky blinders":      ["crime", "drama"],
    "mindhunter":          ["crime", "thriller"],
    "you":                 ["thriller", "crime"],
    "the witcher":         ["action", "fantasy"],
    "dune":                ["sci-fi", "action"],
    "interstellar":        ["sci-fi", "drama"],
    "inception":           ["sci-fi", "thriller"],
    "parasite":            ["thriller", "drama"],
    "get out":             ["horror", "thriller"],
    "titanic":             ["romance", "drama"],
    "avengers":            ["action", "sci-fi"],
    "the office":          ["comedy"],
    "friends":             ["comedy", "romance"],
    "planet earth":        ["documentary"],
    "blue planet":         ["documentary"],
    "making a murderer":   ["documentary", "crime"],
    "the last of us":      ["drama", "horror"],
    "wednesday":           ["comedy", "horror"],
}


def extract_genres_from_query(spoken_query: str, intent_filters: dict) -> list[str]:
    """
    Deeply understand what genres the user wants.
    Handles: explicit genres, moods/vibes, "like X" patterns, mixed requests.
    Returns ordered list of genres (most important first).
    """
    text = spoken_query.lower().strip()
    genres: list[str] = []
    exclude: list[str] = [g.lower() for g in intent_filters.get("exclude_genres", [])]

    # 1. Start with what the intent parser already extracted
    for g in intent_filters.get("genres", []):
        if g.lower() not in genres and g.lower() not in exclude:
            genres.append(g.lower())

    # 2. Detect explicit genre keywords from user's words
    for genre, keywords in GENRE_KEYWORDS.items():
        if genre not in genres and genre not in exclude:
            for kw in keywords:
                if kw in text:
                    genres.append(genre)
                    break

    # 3. Map moods/vibes to genres
    for mood, mapped in MOOD_TO_GENRES.items():
        if mood in text:
            for g in mapped:
                if g not in genres and g not in exclude:
                    genres.append(g)

    # 4. "Similar to / like <title>" → extract that title's genres
    match = re.search(r"(?:like|similar to|same as|such as)\s+(.+?)(?:\s*$|\s+and|\s+but)", text)
    if match:
        ref = match.group(1).strip().strip("'\"").lower()
        for title_key, title_genres in TITLE_GENRES.items():
            if title_key in ref or ref in title_key:
                for g in title_genres:
                    if g not in genres and g not in exclude:
                        genres.append(g)
                break

    # 5. Deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for g in genres:
        if g not in seen and g not in exclude:
            seen.add(g)
            unique.append(g)

    return unique


def get_genre_pool(genre: str, catalog: dict, exclude: list[str], limit: int = 40, prefer_modern: bool = True) -> list[dict]:
    """
    Get the best movies for a specific genre from the catalog.
    Sorted by avg_rating * popularity, with optional recency boost for modern titles.
    """
    matches = []
    for mid, m in catalog.items():
        mg = (m.get("primary_genre") or "").lower()
        if genre in mg:
            if not any(e in mg for e in exclude):
                rating_score  = (m.get("avg_rating", 5.0) / 10.0) * 0.5
                popular_score = min(m.get("popularity", 0) / 200.0, 1.0) * 0.4
                # Recency boost: movies from 1980+ score 0.1 higher
                year = m.get("year") or 0
                recency_score = 0.1 if (prefer_modern and year >= 1980) else 0.0
                score = rating_score + popular_score + recency_score
                matches.append((score, dict(m)))
    matches.sort(key=lambda x: -x[0])
    result = []
    for score, item in matches[:limit]:
        item["score"] = round(score, 4)
        result.append(item)
    return result


def interleave_genres(genre_pools: dict[str, list[dict]], total: int = 24) -> list[dict]:
    """
    Round-robin interleave items from each genre pool.
    If genres = [romance, thriller, comedy], result = [romance#1, thriller#1, comedy#1,
                                                         romance#2, thriller#2, comedy#2, ...]
    This guarantees REAL genre variety in the final 8.
    """
    result = []
    genre_list = list(genre_pools.keys())
    indices = {g: 0 for g in genre_list}
    seen_ids: set = set()

    while len(result) < total:
        added_this_round = False
        for genre in genre_list:
            pool = genre_pools[genre]
            idx = indices[genre]
            while idx < len(pool):
                item = pool[idx]
                idx += 1
                iid = item.get("item_id")
                if iid and iid not in seen_ids:
                    seen_ids.add(iid)
                    result.append(item)
                    added_this_round = True
                    break
            indices[genre] = idx
        if not added_this_round:
            break  # all pools exhausted

    return result[:total]


async def execute_tool(intent, user_id, context_item_ids=None, raw_transcript: str = ""):
    if context_item_ids is None:
        context_item_ids = []
    intent_type = intent.get("intent", "unknown")
    action      = intent.get("action")
    filters     = intent.get("filters", {})

    if intent_type == "control" or action:
        return await _handle_control(action, intent, user_id, context_item_ids)
    if intent_type == "navigate" and intent.get("target_rank") is not None:
        return await _handle_navigate(intent["target_rank"], user_id)
    if intent_type == "explain":
        return await _handle_explain(user_id, intent.get("reference_item_id"), context_item_ids)
    if intent_type == "compare":
        return await _handle_compare(user_id, context_item_ids)

    spoken_query = intent.get("spoken_query") or raw_transcript or ""
    return await _handle_discover(user_id, filters, context_item_ids, intent, spoken_query)


async def _handle_discover(user_id, filters, context_ids, intent=None, spoken_query: str = ""):
    """
    Intent-aware recommendation engine.

    Strategy:
    - Extract ALL genres the user mentioned
    - If 0 genres: use RAG semantic search on full query
    - If 1 genre: RAG search + strict genre filter
    - If 2+ genres: build separate pool per genre, interleave them
      → guarantees e.g. 3 romance + 3 thriller + 2 comedy
    - Always: RL reranking → AI explanations → return top 8
    """
    try:
        from .app import CATALOG, _build_recs
        from .rag_engine import _embed, _INDEX

        genres  = extract_genres_from_query(spoken_query, filters)
        exclude = [g.lower() for g in filters.get("exclude_genres", [])]
        moods   = filters.get("moods", [])

        print(f"[VoiceTool] query={repr(spoken_query[:60])} → genres={genres}")
        items: list[dict] = []

        if len(genres) == 0:
            # ── No genre signal → pure RAG semantic search ────────────────────
            if spoken_query and _INDEX.vecs is not None:
                try:
                    query_vec = _embed(spoken_query)
                    if query_vec is not None:
                        is_similar_q = 'similar to' in spoken_query.lower() or ' like ' in spoken_query.lower()
                        hits = _INDEX.search(query_vec, top_k=80)
                        for mid, score in hits:
                            if mid in CATALOG:
                                item = dict(CATALOG[mid])
                                # For "similar to X" queries, skip very old movies
                                if is_similar_q and (item.get('year') or 0) < 1970:
                                    continue
                                item["score"] = round(score, 4)
                                if not any(e in (item.get("primary_genre") or "").lower() for e in exclude):
                                    items.append(item)
                        print(f"[VoiceTool] RAG (no genre): {len(items)} hits")
                except Exception as e:
                    print(f"[VoiceTool] RAG error: {e}")

            if not items:
                items = _build_recs(user_id, k=30, session_item_ids=context_ids or None)

        elif len(genres) == 1:
            # ── Single genre → RAG + strict filter ───────────────────────────
            genre = genres[0]
            if spoken_query and _INDEX.vecs is not None:
                try:
                    query_vec = _embed(spoken_query)
                    if query_vec is not None:
                        hits = _INDEX.search(query_vec, top_k=80)
                        for mid, score in hits:
                            if mid in CATALOG:
                                item = dict(CATALOG[mid])
                                ig = (item.get("primary_genre") or "").lower()
                                if genre in ig and not any(e in ig for e in exclude):
                                    item["score"] = round(score, 4)
                                    items.append(item)
                        print(f"[VoiceTool] RAG single-genre '{genre}': {len(items)} hits")
                except Exception as e:
                    print(f"[VoiceTool] RAG error: {e}")

            # Supplement with catalog pool if RAG gives < 8
            # For "similar to X" queries, prefer modern movies
            is_similar = 'similar to' in spoken_query.lower() or ' like ' in spoken_query.lower()
            if len(items) < 8:
                pool = get_genre_pool(genre, CATALOG, exclude, limit=40, prefer_modern=is_similar)
                # For "similar to" queries, filter out pre-1970 movies from supplement
                if is_similar:
                    pool = [p for p in pool if (p.get('year') or 0) >= 1970]
                seen = {i.get("item_id") for i in items}
                for p in pool:
                    if p.get("item_id") not in seen:
                        items.append(p)
                        seen.add(p.get("item_id"))

        else:
            # ── Multiple genres → interleave separate pools ───────────────────
            # Each genre gets its own pool from the catalog
            # Then round-robin interleave: genre1#1, genre2#1, genre3#1, genre1#2...
            genre_pools: dict[str, list[dict]] = {}
            items_per_genre = max(12, 24 // len(genres))

            for genre in genres:
                pool = get_genre_pool(genre, CATALOG, exclude, limit=items_per_genre)
                if pool:
                    genre_pools[genre] = pool
                    print(f"[VoiceTool] Pool '{genre}': {len(pool)} items")

            if genre_pools:
                items = interleave_genres(genre_pools, total=24)
                print(f"[VoiceTool] Interleaved {len(genres)} genres → {len(items)} items")
            else:
                items = _build_recs(user_id, k=30, session_item_ids=context_ids or None)

        # ── Exclude genres ────────────────────────────────────────────────────
        if exclude:
            items = [i for i in items
                     if not any(e in (i.get("primary_genre") or "").lower() for e in exclude)]

        # ── Deduplicate ───────────────────────────────────────────────────────
        seen_ids: set = set()
        deduped: list[dict] = []
        for item in items:
            iid = item.get("item_id")
            if iid and iid not in seen_ids:
                seen_ids.add(iid)
                deduped.append(item)
        items = deduped

        # ── RL reranking ──────────────────────────────────────────────────────
        try:
            from .app import RL_AGENT
            items = RL_AGENT.rerank(items[:30], user_activity={"avg_rating": 3.8, "n_ratings": 20})
        except Exception:
            pass

        top8 = items[:8]

        # ── AI explanations for ALL 8 ─────────────────────────────────────────
        top8 = await _attach_explanations(user_id, top8, spoken_query, genres)

        return {
            "tool":            "recommend_titles",
            "items":           top8,
            "filters_applied": bool(genres or exclude or moods),
            "count":           len(top8),
            "query":           spoken_query,
            "genres_used":     genres,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"tool": "recommend_titles", "items": [], "error": str(e)}


async def _attach_explanations(user_id: int, items: list[dict],
                                query: str, genres: list[str]) -> list[dict]:
    """Generate a personalised explanation for every item."""
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    results: list[dict] = []
    for item in items:
        existing = item.get("llm_reasoning") or item.get("rag_reason") or item.get("reason") or ""
        if len(existing) > 20:
            item["reason"] = existing
            results.append(item)
            continue
        reason = ""
        if openai_key:
            try:
                reason = await _openai_explain_item(item, query, genres, openai_key)
            except Exception:
                pass
        if not reason:
            reason = _rule_reason(item, query, genres)
        item["reason"] = reason
        results.append(item)
    return results


async def _openai_explain_item(item: dict, query: str, genres: list[str], api_key: str) -> str:
    import http.client, ssl, json
    title  = item.get("title", "this title")
    genre  = item.get("primary_genre", "")
    year   = item.get("year", "")
    prompt = (
        f"The user asked for: '{query}'. "
        f"Explain in ONE sentence (max 25 words) why '{title}' ({genre}, {year}) "
        f"is a great match. Be specific and conversational."
    )
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
        "temperature": 0.7,
    }, ensure_ascii=True).encode("utf-8")
    ctx  = ssl.create_default_context()
    conn = http.client.HTTPSConnection("api.openai.com", context=ctx, timeout=6)
    try:
        conn.request("POST", "/v1/chat/completions", body=payload, headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        })
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip().strip('"')
    finally:
        conn.close()


def _rule_reason(item: dict, query: str, genres: list[str]) -> str:
    title  = item.get("title", "this title")
    genre  = item.get("primary_genre", "")
    rating = item.get("avg_rating", 0)
    score  = item.get("score", 0)
    q      = query.lower()
    g      = genre.lower() if genre else ""

    if "similar to" in q or " like " in q:
        match = re.search(r"(?:similar to| like )\s*(.+?)(?:\s*$)", q)
        ref = match.group(1).strip().strip("'\"").title() if match else "your selection"
        return f"Shares the same {genre or 'genre'} DNA as {ref} — fans consistently love both."

    reason_by_genre = {
        "thriller":     f"A gripping {genre} with nail-biting tension that keeps you on the edge.",
        "comedy":       f"A feel-good {genre} that delivers exactly the laughs you're looking for.",
        "romance":      f"A heartfelt {genre} with a beautiful love story you'll remember.",
        "horror":       f"A genuinely scary {genre} that delivers real chills and atmosphere.",
        "sci-fi":       f"A mind-bending {genre} with thought-provoking ideas and great storytelling.",
        "drama":        f"An emotionally powerful {genre} with outstanding performances.",
        "action":       f"High-octane {genre} with spectacular sequences — exactly what you asked for.",
        "documentary":  f"A compelling, real-world story that will leave you thinking for days.",
        "crime":        f"A gripping {genre} with complex characters and a twisting plot.",
        "animation":    f"A visually stunning {genre} that works for all ages.",
        "fantasy":      f"A magical {genre} world full of wonder and adventure.",
        "adventure":    f"An exciting {genre} journey that will keep you hooked start to finish.",
        "family":       f"A wholesome {genre} film perfect for watching with the whole family.",
        "mystery":      f"A clever {genre} with a satisfying puzzle you'll want to solve.",
    }
    for gkey, reason in reason_by_genre.items():
        if gkey in g:
            return reason

    if score > 0.8:
        return f"Ranked as an exceptional {int(score*100)}% match based on your request."
    if rating > 7.5:
        return f"Rated {rating:.1f}/10 — one of the highest-rated titles in this category."
    return f"Recommended because it closely matches what you asked for."


# ── Control / navigate / explain / compare ────────────────────────────────────

async def _handle_explain(user_id, item_id, context_ids):
    try:
        target_id = item_id or (context_ids[-1] if context_ids else None)
        if not target_id:
            return {"tool": "explain_item", "explanation": "No item specified to explain.", "items": []}
        result = _call_explain_endpoint(user_id, [target_id])
        explanations = result.get("explanations", [])
        text = explanations[0].get("reason", "No explanation available.") if explanations else "No explanation available."
        return {"tool": "explain_item", "item_id": target_id, "explanation": text, "items": []}
    except Exception as e:
        return {"tool": "explain_item", "explanation": str(e), "items": []}


async def _handle_compare(user_id, context_ids):
    if len(context_ids) < 2:
        return {"tool": "compare_items", "items": [], "message": "Add at least 2 items to compare."}
    items = []
    for iid in context_ids[:3]:
        try:
            items.append(_call_item_endpoint(iid))
        except Exception:
            pass
    return {"tool": "compare_items", "items": items}


async def _handle_navigate(rank, user_id):
    return {"tool": "open_item", "action": "navigate", "target_rank": rank, "items": [],
            "message": f"Opening item #{rank}."}


async def _handle_control(action, intent, user_id, context_ids):
    if action == "add_to_list":
        item_id = intent.get("reference_item_id") or (context_ids[-1] if context_ids else None)
        return {"tool": "add_to_session", "action": "add_to_list", "item_id": item_id, "items": []}
    if action == "remove_from_session":
        return {"tool": "remove_from_session", "action": "remove",
                "item_id": intent.get("reference_item_id"), "items": []}
    if action == "clear_session":
        return {"tool": "clear_session", "action": "clear_session", "items": []}
    if action in ("like", "dislike"):
        return {"tool": "feedback", "action": action,
                "item_id": intent.get("reference_item_id"), "items": []}
    if action in ("play", "play_trailer", "open_item"):
        return {"tool": action, "action": action,
                "target_rank": intent.get("target_rank"),
                "item_id": intent.get("reference_item_id"), "items": []}
    return {"tool": "unknown", "action": action, "items": [],
            "message": f"Action '{action}' not recognized."}


def _call_explain_endpoint(user_id, item_ids):
    import http.client, json
    body = json.dumps({"user_id": user_id, "item_ids": item_ids}, ensure_ascii=True).encode("utf-8")
    conn = http.client.HTTPConnection("localhost", 8000, timeout=5)
    try:
        conn.request("POST", "/explain", body=body,
                     headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
        return json.loads(conn.getresponse().read().decode("utf-8"))
    except Exception:
        return {"explanations": []}
    finally:
        conn.close()


def _call_item_endpoint(item_id):
    import http.client, json
    conn = http.client.HTTPConnection("localhost", 8000, timeout=5)
    try:
        conn.request("GET", f"/item/{item_id}", headers={})
        return json.loads(conn.getresponse().read().decode("utf-8"))
    except Exception:
        return {"item_id": item_id, "title": f"Item #{item_id}"}
    finally:
        conn.close()
