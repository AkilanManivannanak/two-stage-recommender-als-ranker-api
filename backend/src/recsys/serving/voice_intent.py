"""
Voice Intent Parser — Production Grade v3
==========================================
Smart local fallback extracts genres/moods from query text
so voice search works even when OpenAI is unavailable.
Uses Chat Completions API (not Responses API) for reliability.
"""
from __future__ import annotations
import json, os, unicodedata, re

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

GENRE_KEYWORDS = {
    "action": ["action", "fight", "combat", "explosive", "high-octane", "chase"],
    "comedy": ["comedy", "funny", "hilarious", "laugh", "humorous", "comedies", "feel good", "feel-good", "lighthearted"],
    "drama": ["drama", "dramatic", "emotional", "powerful", "gripping"],
    "horror": ["horror", "scary", "terrifying", "creepy", "spooky", "haunted", "zombie"],
    "sci-fi": ["sci-fi", "science fiction", "scifi", "space", "alien", "futuristic", "dystopian", "cyberpunk", "mind-bending", "mind bending"],
    "romance": ["romance", "romantic", "love", "love story", "romcom", "rom-com", "heartfelt", "relationship"],
    "thriller": ["thriller", "suspense", "suspenseful", "tense", "dark", "psychological", "mystery", "twist", "edge-of-seat"],
    "documentary": ["documentary", "true story", "real life", "nature", "history", "docuseries"],
    "animation": ["animation", "animated", "cartoon", "anime", "pixar"],
    "crime": ["crime", "heist", "murder", "detective", "police", "gangster", "mafia", "cartel"],
    "adventure": ["adventure", "quest", "journey", "explore", "treasure"],
    "fantasy": ["fantasy", "magic", "magical", "mythical", "dragon", "medieval"],
}

TITLE_KEYWORDS = {
    "stranger things": ["sci-fi", "horror", "thriller"],
    "dark": ["sci-fi", "thriller"],
    "breaking bad": ["crime", "drama", "thriller"],
    "game of thrones": ["fantasy", "drama", "action"],
    "black mirror": ["sci-fi", "thriller"],
    "ozark": ["crime", "thriller", "drama"],
    "narcos": ["crime", "drama"],
    "money heist": ["crime", "thriller", "action"],
    "squid game": ["thriller", "drama", "action"],
    "wednesday": ["horror", "comedy"],
    "bridgerton": ["romance", "drama"],
    "the crown": ["drama"],
    "peaky blinders": ["crime", "drama"],
    "mindhunter": ["crime", "thriller"],
    "you": ["thriller"],
    "the witcher": ["action", "fantasy"],
    "lupin": ["crime", "thriller"],
    "alice in borderland": ["thriller", "action"],
    "dune": ["sci-fi", "action"],
    "interstellar": ["sci-fi", "drama"],
    "inception": ["sci-fi", "thriller"],
    "parasite": ["thriller", "drama"],
    "get out": ["horror", "thriller"],
}

MOOD_KEYWORDS = {
    "feel good": ["comedy", "romance"],
    "feel-good": ["comedy", "romance"],
    "uplifting": ["comedy", "romance", "drama"],
    "dark": ["thriller", "crime", "horror"],
    "intense": ["thriller", "action"],
    "relaxing": ["comedy", "romance", "documentary"],
    "scary": ["horror", "thriller"],
    "exciting": ["action", "thriller", "adventure"],
    "emotional": ["drama", "romance"],
    "mind-bending": ["sci-fi", "thriller"],
    "heartwarming": ["romance", "comedy", "drama"],
    "gritty": ["crime", "thriller", "drama"],
    "fun": ["comedy", "action", "animation"],
    "thought-provoking": ["sci-fi", "drama", "documentary"],
}

SYSTEM_PROMPT = """You are a voice intent parser for CineWave, a movie/show recommendation app.
Parse the user's spoken request into a strict JSON object.

Rules:
1. Extract genres, moods, filters as precisely as possible
2. If the request mentions a specific title (like 'something like Stranger Things'), extract the genres that title belongs to
3. Normalize informal language: 'k-drama' -> Korean, 'romcom' -> romance+comedy
4. confidence reflects how certain you are (0.0-1.0)
5. For discover/refine intents, ALWAYS set needs_clarification to false
6. spoken_query is a clean normalized version of what the user asked
7. Never invent item IDs
8. action is only set for control commands (play, like, etc.)

Return ONLY valid JSON with these exact keys:
intent, filters (with genres, exclude_genres, languages, countries, min_year, max_year, max_runtime_minutes, min_rating, moods, family_safe), reference_item_id, target_rank, action, needs_clarification, clarification_question, confidence, spoken_query"""


async def parse_intent(transcript, user_id, context_item_ids=None):
    if context_item_ids is None:
        context_item_ids = []
    if not OPENAI_KEY:
        return _smart_fallback_intent(transcript)
    normalized = _normalize_transcript(transcript)
    context_note = ""
    if context_item_ids:
        context_note = "\nCurrent session item IDs: " + str(context_item_ids)
    user_message = 'User said: "' + normalized + '"\nUser ID: ' + str(user_id) + context_note
    try:
        result = await _call_intent_api(user_message)
        if result and result.get("intent") != "unknown":
            if result.get("intent") == "discover" and result.get("filters", {}).get("genres"):
                result["needs_clarification"] = False
            return result
    except Exception as e:
        print("  [VoiceIntent] OpenAI call failed: " + str(e))
    return _smart_fallback_intent(transcript)


async def _call_intent_api(user_message):
    import http.client, ssl
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 500,
        "temperature": 0,
    }, ensure_ascii=True).encode("utf-8")
    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection("api.openai.com", context=ctx, timeout=10)
    try:
        conn.request("POST", "/v1/chat/completions", body=payload, headers={
            "Authorization": "Bearer " + OPENAI_KEY,
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        })
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()
    try:
        text = data["choices"][0]["message"]["content"]
        result = json.loads(text)
        result.setdefault("intent", "discover")
        result.setdefault("filters", {"genres": [], "exclude_genres": [], "languages": [],
                                       "countries": [], "min_year": None, "max_year": None,
                                       "max_runtime_minutes": None, "min_rating": None,
                                       "moods": [], "family_safe": None})
        result.setdefault("reference_item_id", None)
        result.setdefault("target_rank", None)
        result.setdefault("action", None)
        result.setdefault("needs_clarification", False)
        result.setdefault("clarification_question", None)
        result.setdefault("confidence", 0.8)
        result.setdefault("spoken_query", "")
        return result
    except Exception:
        pass
    return None


def _normalize_transcript(text):
    return unicodedata.normalize("NFKC", text).lower().strip()


def _smart_fallback_intent(transcript):
    """Smart local parsing that extracts genres and moods from the query text."""
    text = transcript.lower().strip()
    genres = []
    moods = []
    exclude = []
    spoken_query = transcript

    # Check for "something like [title]" pattern
    like_match = re.search(r"(?:like|similar to|same as|such as)\s+(.+?)(?:\s*$)", text)
    if like_match:
        ref_title = like_match.group(1).strip().strip("'\"")
        for title_key, title_genres in TITLE_KEYWORDS.items():
            if title_key in ref_title.lower():
                genres.extend(title_genres)
                break
        if not genres:
            genres = ["sci-fi", "thriller", "drama"]

    # Check for "not [genre]" exclusions
    words = text.split()
    for i, w in enumerate(words):
        if w in ("not", "no", "without", "exclude") and i + 1 < len(words):
            next_word = words[i + 1]
            for genre, keywords in GENRE_KEYWORDS.items():
                if next_word in keywords or next_word == genre:
                    exclude.append(genre)

    # Extract mood keywords
    for mood, mood_genres in MOOD_KEYWORDS.items():
        if mood in text:
            moods.append(mood)
            for mg in mood_genres:
                if mg not in genres and mg not in exclude:
                    genres.append(mg)

    # Extract genre keywords directly
    for genre, keywords in GENRE_KEYWORDS.items():
        for kw in keywords:
            if kw in text and genre not in genres and genre not in exclude:
                genres.append(genre)
                break

    # Deduplicate while preserving order
    seen = set()
    unique_genres = []
    for g in genres:
        if g not in seen and g not in exclude:
            seen.add(g)
            unique_genres.append(g)
    genres = unique_genres

    exclude = list(dict.fromkeys(exclude))
    has_signal = bool(genres or moods)

    return {
        "intent": "discover",
        "filters": {
            "genres": genres,
            "exclude_genres": exclude,
            "languages": [],
            "countries": [],
            "min_year": None,
            "max_year": None,
            "max_runtime_minutes": None,
            "min_rating": None,
            "moods": moods,
            "family_safe": None,
        },
        "reference_item_id": None,
        "target_rank": None,
        "action": None,
        "needs_clarification": False,
        "clarification_question": None,
        "confidence": 0.85 if has_signal else 0.4,
        "spoken_query": spoken_query,
    }
