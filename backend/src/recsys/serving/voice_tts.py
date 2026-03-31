"""
Voice TTS — Text-to-Speech Synthesis
=====================================
Uses gpt-4o-mini-tts for conversational replies (low latency).
Falls back to tts-1 for simpler responses.
Returns base64-encoded MP3 audio.
"""
from __future__ import annotations

import base64
import json
import os
import ssl
import http.client

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# Character limit for spoken responses — keep it concise
MAX_CHARS = 300


async def synthesize_speech(
    text: str,
    voice: str = "alloy",
    speed: float = 1.0,
    model: str = "gpt-4o-mini-tts",
) -> str | None:
    """
    Synthesize text to speech.
    Returns base64-encoded MP3 string, or None if unavailable.

    Voices: alloy | echo | fable | onyx | nova | shimmer
    """
    if not OPENAI_KEY or not text:
        return None

    # Truncate to keep responses concise and low-latency
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0] + "..."

    try:
        body = json.dumps({
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "mp3",
        }, ensure_ascii=True).encode("utf-8")

        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.openai.com", context=ctx, timeout=15)
        try:
            conn.request("POST", "/v1/audio/speech", body=body, headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            })
            resp = conn.getresponse()
            if resp.status != 200:
                return None
            audio_bytes = resp.read()
            return base64.b64encode(audio_bytes).decode("utf-8")
        finally:
            conn.close()

    except Exception:
        return None
