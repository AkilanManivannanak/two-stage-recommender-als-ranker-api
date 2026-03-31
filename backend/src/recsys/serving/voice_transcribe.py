"""
Voice Transcription — Dual-Pass STT
====================================
Pass 1: gpt-4o-mini-transcribe  (fast, cheap)
Pass 2: gpt-4o-transcribe       (high-accuracy rerun if confidence low)
Optional: gpt-4o-transcribe-diarize for multi-speaker

Confidence heuristic:
  - If the transcript is very short and has no punctuation → low confidence
  - If the mini model returns <no_speech> or empty → low confidence
  - Threshold: confidence < 0.6 → rerun with high-accuracy model
"""
from __future__ import annotations

import io
import os
import time
import unicodedata

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# Confidence threshold below which we rerun with the full model
RERUN_THRESHOLD = 0.6


async def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/webm",
    high_accuracy: bool = False,
    multi_speaker: bool = False,
    language_hint: str | None = None,
) -> dict:
    """
    Transcribe audio bytes. Returns:
    {
      "transcript": str,
      "confidence": float,
      "model_used": str,
      "duration_ms": int,
      "speakers": list | None,
    }
    """
    if not OPENAI_KEY:
        return {
            "transcript": "",
            "confidence": 0.0,
            "model_used": "none",
            "duration_ms": 0,
            "error": "OPENAI_API_KEY not configured",
        }

    t0 = time.time()

    # Choose model
    if multi_speaker:
        model = "gpt-4o-transcribe"   # diarize requires full model
        diarize = True
    elif high_accuracy:
        model = "gpt-4o-transcribe"
        diarize = False
    else:
        model = "gpt-4o-mini-transcribe"
        diarize = False

    result = await _call_transcription(audio_bytes, mime_type, model, language_hint, diarize)
    confidence = _estimate_confidence(result)

    # Dual-pass: rerun with full model if confidence low
    if not high_accuracy and not multi_speaker and confidence < RERUN_THRESHOLD:
        result2 = await _call_transcription(
            audio_bytes, mime_type, "gpt-4o-transcribe", language_hint, False
        )
        conf2 = _estimate_confidence(result2)
        if conf2 > confidence:
            result = result2
            confidence = conf2
            model = "gpt-4o-transcribe (rerun)"

    duration_ms = int((time.time() - t0) * 1000)

    return {
        "transcript": result.get("text", "").strip(),
        "confidence": round(confidence, 3),
        "model_used": model,
        "duration_ms": duration_ms,
        "speakers": result.get("speakers"),
        "words": result.get("words"),
    }


async def _call_transcription(
    audio_bytes: bytes,
    mime_type: str,
    model: str,
    language: str | None,
    diarize: bool,
) -> dict:
    """Call OpenAI transcription API using multipart/form-data."""
    import http.client, ssl, json

    ext = _ext_for_mime(mime_type)
    boundary = "----CineWaveVoiceBoundary"
    body_parts = []

    # file part
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.{ext}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    )
    body_bytes = b"".join(p.encode() if isinstance(p, str) else p for p in body_parts)
    body_bytes += audio_bytes + b"\r\n"

    # model part
    body_bytes += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"{model}\r\n"
    ).encode()

    # response_format
    body_bytes += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="response_format"\r\n\r\n'
        f"verbose_json\r\n"
    ).encode()

    # language
    if language:
        body_bytes += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="language"\r\n\r\n'
            f"{language}\r\n"
        ).encode()

    # diarize
    if diarize:
        body_bytes += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="diarize"\r\n\r\ntrue\r\n'
        ).encode()

    body_bytes += f"--{boundary}--\r\n".encode()

    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection("api.openai.com", context=ctx, timeout=30)
    try:
        conn.request("POST", "/v1/audio/transcriptions", body=body_bytes, headers={
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body_bytes)),
        })
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        return json.loads(raw)
    except Exception as e:
        return {"text": "", "error": str(e)}
    finally:
        conn.close()


def _estimate_confidence(result: dict) -> float:
    """Heuristic confidence score from 0.0 to 1.0."""
    text = result.get("text", "").strip()
    if not text:
        return 0.0
    if "error" in result:
        return 0.0

    score = 1.0

    # Short utterances with no punctuation are suspect
    if len(text) < 10:
        score -= 0.2

    # Check avg_logprob if available
    avg_logprob = result.get("avg_logprob")
    if avg_logprob is not None:
        # logprob of 0 = perfect, -1 = poor
        score = min(1.0, max(0.0, 1.0 + avg_logprob))
        return round(score, 3)

    # Check no_speech_prob
    no_speech = result.get("no_speech_prob", 0.0)
    if no_speech > 0.5:
        score -= 0.5

    return max(0.0, round(score, 3))


def _ext_for_mime(mime: str) -> str:
    return {
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/mp4": "mp4",
        "audio/wav": "wav",
        "audio/mpeg": "mp3",
        "audio/flac": "flac",
    }.get(mime, "webm")
