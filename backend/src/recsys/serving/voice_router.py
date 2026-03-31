"""
Voice Interaction Router — CineWave Production v4
===================================================
FIXES:
  - Passes raw transcript as raw_transcript kwarg to execute_tool so the
    spoken query is always used for RAG semantic search
  - /voice/respond TTS endpoint confirmed working
  - Interactive spoken response names the query, genres, and titles
  - Never returns early on clarification — always gets results
"""
from __future__ import annotations
import asyncio, base64, json, os, time, uuid
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .voice_transcribe import transcribe_audio
from .voice_intent import parse_intent
from .voice_tools import execute_tool
from .voice_tts import synthesize_speech
from .voice_metrics import record_voice_event

router = APIRouter(prefix="/voice", tags=["voice"])
_PENDING = {}
_METRICS = {"total_requests": 0, "transcription_errors": 0, "intent_clarifications": 0,
            "tool_calls": 0, "tts_calls": 0, "avg_latency_ms": 0.0, "latency_samples": []}


class TranscribeRequest(BaseModel):
    audio_base64: str
    mime_type: str = "audio/webm"
    high_accuracy: bool = False
    multi_speaker: bool = False
    language_hint: Optional[str] = None

class IntentRequest(BaseModel):
    transcript: str
    user_id: int
    context_item_ids: list[int] = []

class AssistRequest(BaseModel):
    audio_base64: Optional[str] = None
    transcript: Optional[str] = None
    mime_type: str = "audio/webm"
    user_id: int
    context_item_ids: list[int] = []
    speak_response: bool = True
    high_accuracy: bool = False

class RespondRequest(BaseModel):
    text: str
    voice: str = "alloy"
    speed: float = 1.0

class ConfirmRequest(BaseModel):
    request_id: str
    confirmed: bool

class RealtimeSessionRequest(BaseModel):
    user_id: int
    context_item_ids: list[int] = []


@router.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    t0 = time.time()
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        result = await transcribe_audio(audio_bytes=audio_bytes, mime_type=req.mime_type,
                                         high_accuracy=req.high_accuracy,
                                         multi_speaker=req.multi_speaker,
                                         language_hint=req.language_hint)
        record_voice_event("transcription", time.time() - t0, success=True)
        return result
    except Exception as e:
        _METRICS["transcription_errors"] += 1
        record_voice_event("transcription", time.time() - t0, success=False)
        raise HTTPException(status_code=500, detail="Transcription failed: " + str(e))


@router.post("/intent")
async def intent(req: IntentRequest):
    t0 = time.time()
    result = await parse_intent(transcript=req.transcript, user_id=req.user_id,
                                 context_item_ids=req.context_item_ids)
    record_voice_event("intent", time.time() - t0, success=True)
    if result.get("needs_clarification"):
        _METRICS["intent_clarifications"] += 1
    return result


@router.post("/assist")
async def assist(req: AssistRequest):
    """Full voice pipeline — NEVER returns early on clarification. Always gets results."""
    t0 = time.time()
    request_id = str(uuid.uuid4())
    _METRICS["total_requests"] += 1

    # 1. Get transcript
    transcript = ""
    confidence = 1.0
    if req.transcript:
        transcript = req.transcript
        confidence = 1.0
    elif req.audio_base64:
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
            transcription = await transcribe_audio(
                audio_bytes=audio_bytes, mime_type=req.mime_type,
                high_accuracy=req.high_accuracy)
            transcript = transcription.get("transcript", "")
            confidence = transcription.get("confidence", 1.0)
        except Exception as e:
            return JSONResponse(status_code=200, content={
                "request_id": request_id, "stage": "transcription_failed",
                "error": str(e), "spoken": "Sorry, I couldn't hear that. Please try again.",
            })
    else:
        return JSONResponse(status_code=200, content={
            "request_id": request_id, "stage": "no_input",
            "spoken": "I didn't get any input. Please try again.",
        })

    # 2. Parse intent
    intent_result = await parse_intent(transcript=transcript, user_id=req.user_id,
                                        context_item_ids=req.context_item_ids)

    # 3. ALWAYS execute tool — pass the raw transcript so RAG can use it
    _METRICS["tool_calls"] += 1
    try:
        tool_result = await execute_tool(
            intent=intent_result,
            user_id=req.user_id,
            context_item_ids=req.context_item_ids,
            raw_transcript=transcript,   # ← KEY FIX: full query always reaches RAG
        )
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "request_id": request_id, "stage": "tool_failed",
            "transcript": transcript, "intent": intent_result,
            "error": str(e),
            "spoken": "I had trouble finding results. Try rephrasing your request.",
        })

    # 4. Confirm destructive actions
    action = intent_result.get("action")
    if action in ("play", "set_profile", "remove_from_session"):
        _PENDING[request_id] = {"intent": intent_result, "tool_result": tool_result,
                                 "user_id": req.user_id, "expires_at": time.time() + 30}
        question = _confirmation_prompt(action, intent_result, tool_result)
        spoken_audio = None
        if req.speak_response:
            try:
                spoken_audio = await synthesize_speech(question)
            except Exception:
                pass
        return {"request_id": request_id, "stage": "confirmation_needed",
                "transcript": transcript, "intent": intent_result,
                "tool_result": tool_result, "spoken": question,
                "spoken_audio_base64": spoken_audio}

    # 5. Build interactive spoken response referencing the actual query + titles
    spoken_text = _build_spoken_response(intent_result, tool_result, transcript)
    spoken_audio = None
    if req.speak_response:
        _METRICS["tts_calls"] += 1
        try:
            spoken_audio = await synthesize_speech(spoken_text)
        except Exception:
            spoken_audio = None

    latency_ms = (time.time() - t0) * 1000
    _update_latency(latency_ms)
    record_voice_event("assist", time.time() - t0, success=True)

    items = tool_result.get("items", [])
    return {
        "request_id":           request_id,
        "stage":                "complete",
        "transcript":           transcript,
        "confidence":           confidence,
        "intent":               intent_result,
        "tool_result":          tool_result,
        "items":                items,
        "spoken":               spoken_text,
        "spoken_response":      spoken_text,
        "spoken_audio_base64":  spoken_audio,
        "latency_ms":           round(latency_ms, 1),
    }


@router.post("/respond")
async def respond(req: RespondRequest):
    """TTS endpoint — converts text to MP3 audio and returns base64."""
    try:
        audio_b64 = await synthesize_speech(req.text, voice=req.voice, speed=req.speed)
        return {"audio_base64": audio_b64, "audio_b64": audio_b64, "format": "mp3"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="TTS failed: " + str(e))


@router.post("/realtime/session")
async def realtime_session(req: RealtimeSessionRequest):
    import http.client, ssl, json as _json
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    body = _json.dumps({
        "model": "gpt-4o-realtime-preview-2024-12-17", "voice": "alloy",
        "instructions": _realtime_system_prompt(req.user_id, req.context_item_ids),
        "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
        "turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
                           "silence_duration_ms": 500, "create_response": False},
        "input_audio_noise_reduction": {"type": "near_field"}, "interrupt_response": True,
    }, ensure_ascii=True).encode("utf-8")
    ctx  = ssl.create_default_context()
    conn = http.client.HTTPSConnection("api.openai.com", context=ctx, timeout=10)
    try:
        conn.request("POST", "/v1/realtime/sessions", body=body, headers={
            "Authorization": "Bearer " + api_key, "Content-Type": "application/json",
            "Content-Length": str(len(body))})
        resp = conn.getresponse()
        data = _json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()
    if "client_secret" not in data:
        raise HTTPException(status_code=502, detail="Realtime session error: " + str(data))
    return {"client_secret": data["client_secret"], "session_id": data.get("id"),
            "expires_at": data.get("expires_at"), "user_id": req.user_id}


@router.post("/confirm")
async def confirm(req: ConfirmRequest):
    pending = _PENDING.get(req.request_id)
    if not pending:
        raise HTTPException(status_code=404, detail="No pending action found")
    if time.time() > pending["expires_at"]:
        del _PENDING[req.request_id]
        raise HTTPException(status_code=410, detail="Confirmation expired")
    del _PENDING[req.request_id]
    if req.confirmed:
        return {"confirmed": True, "tool_result": pending["tool_result"]}
    return {"confirmed": False, "message": "Action cancelled"}


@router.get("/status/{request_id}")
async def status(request_id: str):
    if request_id in _PENDING:
        return {"status": "pending_confirmation", "request_id": request_id}
    return {"status": "not_found", "request_id": request_id}


@router.get("/metrics")
async def metrics():
    return {**_METRICS, "pending_confirmations": len(_PENDING),
            "avg_latency_ms": round(_METRICS["avg_latency_ms"], 1)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confirmation_prompt(action, intent, tool_result):
    if action == "play":
        title = (tool_result.get("items") or [{}])[0].get("title", "that title")
        return "Just to confirm -- play " + title + "?"
    if action == "set_profile":
        return "Switch to user " + str(intent.get("reference_item_id")) + "?"
    return "Are you sure you want to do that?"


def _build_spoken_response(intent: dict, tool_result: dict, transcript: str = "") -> str:
    """
    Build an interactive, conversational response.
    - References what the user actually asked
    - Names the top 3 titles
    - Describes the genre/mood
    - Invites follow-up
    """
    intent_type = intent.get("intent", "unknown")
    items       = tool_result.get("items", [])
    count       = len(items)
    query       = transcript or intent.get("spoken_query", "")
    genres_used = tool_result.get("genres_used", [])
    filters     = intent.get("filters", {})
    genres      = genres_used or filters.get("genres", [])
    moods       = filters.get("moods", [])

    if intent_type in ("discover", "refine") and items:
        # Genre descriptor
        descriptor = ""
        if genres:
            descriptor = " and ".join(g.title() for g in genres[:2])
        elif moods:
            descriptor = " and ".join(moods[:2])

        # Top title names
        top_titles = [i.get("title", "") for i in items[:3] if i.get("title")]
        if len(top_titles) >= 3:
            titles_str = top_titles[0] + ", " + top_titles[1] + ", and " + top_titles[2]
        elif len(top_titles) == 2:
            titles_str = top_titles[0] + " and " + top_titles[1]
        elif top_titles:
            titles_str = top_titles[0]
        else:
            titles_str = ""

        # "similar to X" path — most natural phrasing
        query_lower = query.lower()
        if "similar to" in query_lower or (" like " in query_lower and "like " in query_lower[:20]):
            import re
            m = re.search(r"(?:similar to| like )\s*(.+?)(?:\s*$)", query_lower)
            ref = m.group(1).strip().strip("'\"").title() if m else "your selection"
            if titles_str:
                return (
                    f"Great taste! I found {count} titles with the same vibe as {ref}. "
                    f"Leading the list: {titles_str}. "
                    f"Each one shares that same energy. Want me to explain why any of these were picked?"
                )
            return f"I found {count} titles similar to {ref}. Tap any to learn more!"

        # Genre/mood path
        if descriptor and titles_str:
            top_pick = items[0].get("title", "The top pick")
            top_reason = items[0].get("reason", "")
            reason_snippet = f" — {top_reason[:80]}..." if top_reason else ""
            return (
                f"Here are {count} {descriptor} picks for you! "
                f"Top recommendations: {titles_str}. "
                f"{top_pick} leads the list{reason_snippet} "
                f"Want me to tell you more about any of these?"
            )

        if titles_str:
            return (
                f"Here are {count} recommendations for you. "
                f"Starting with {titles_str}. "
                f"Want to know why any of these were chosen?"
            )

        return f"I found {count} titles that match what you're looking for. Tap any to explore!"

    if intent_type == "explain" and tool_result.get("explanation"):
        return tool_result["explanation"]

    if intent_type == "navigate":
        title = items[0].get("title", "that title") if items else "your selection"
        return "Opening " + title + " for you now."

    if intent_type == "control":
        action = intent.get("action", "done")
        return "Got it -- " + action + "."

    if not items:
        if query:
            return (
                f"I couldn't find specific matches for '{query}'. "
                f"Try asking for a genre like 'dark thriller' or 'feel-good comedy', "
                f"or name a title you liked — like 'something similar to Stranger Things'!"
            )
        return "Could you describe what you're in the mood for? Try a genre, mood, or a title you liked!"

    return f"Found {count} results for you. Want me to explain any of these picks?"


def _update_latency(ms):
    samples = _METRICS["latency_samples"]
    samples.append(ms)
    if len(samples) > 100:
        samples.pop(0)
    _METRICS["avg_latency_ms"] = sum(samples) / len(samples)


def _realtime_system_prompt(user_id, context_ids):
    ctx = "User ID: " + str(user_id) + ". Session items: " + str(context_ids or "none") + "."
    return (
        "You are CineWave, a cinematic AI assistant. "
        "Help users discover movies and shows through natural conversation. "
        "Be concise -- spoken answers should be 1-2 sentences. "
        "Never rank or recommend content yourself; always route to the recommender tools. " + ctx
    )
