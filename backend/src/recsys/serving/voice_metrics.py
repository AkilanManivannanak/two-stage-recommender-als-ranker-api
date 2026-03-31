"""
Voice Metrics — Pipeline Observability
========================================
Tracks:
  - Transcription WER proxy
  - Intent accuracy signals
  - Tool selection precision
  - Clarification rate
  - Voice-to-click rate
  - Abandonment after voice
  - Avg latency per stage
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

# Rolling window metrics
_EVENTS: deque = deque(maxlen=1000)
_STAGE_LATENCIES: dict[str, list[float]] = {
    "transcription": [],
    "intent": [],
    "tool": [],
    "tts": [],
    "assist": [],
}


def record_voice_event(stage: str, duration_s: float, success: bool, meta: dict | None = None):
    _EVENTS.append({
        "stage": stage,
        "duration_ms": round(duration_s * 1000, 1),
        "success": success,
        "ts": time.time(),
        "meta": meta or {},
    })
    samples = _STAGE_LATENCIES.setdefault(stage, [])
    samples.append(duration_s * 1000)
    if len(samples) > 200:
        samples.pop(0)


def get_voice_metrics() -> dict:
    total = len(_EVENTS)
    if total == 0:
        return {"total_events": 0}

    successes = sum(1 for e in _EVENTS if e["success"])
    clarifications = sum(1 for e in _EVENTS if e["meta"].get("clarification"))
    tool_calls = sum(1 for e in _EVENTS if e["stage"] == "tool")

    latencies = {}
    for stage, samples in _STAGE_LATENCIES.items():
        if samples:
            latencies[f"{stage}_p50_ms"] = round(sorted(samples)[len(samples) // 2], 1)
            latencies[f"{stage}_p95_ms"] = round(sorted(samples)[int(len(samples) * 0.95)], 1)
            latencies[f"{stage}_avg_ms"] = round(sum(samples) / len(samples), 1)

    return {
        "total_events": total,
        "success_rate": round(successes / total, 3),
        "clarification_rate": round(clarifications / max(tool_calls, 1), 3),
        **latencies,
    }
