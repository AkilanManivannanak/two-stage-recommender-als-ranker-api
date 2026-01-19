from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"
    timeout_sec: float = 2.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 120
    use_structured_output: bool = True


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fallback_explanations(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        title = (it.get("title") or "").strip()
        genres = (it.get("genres") or "").strip()
        if title and genres:
            reason = f"Recommended because you tend to engage with {genres} titles like “{title}”."
        elif title:
            reason = f"Recommended because it matches your inferred taste from your recent history: “{title}”."
        elif genres:
            reason = f"Recommended because it matches your inferred taste for {genres} content."
        else:
            reason = "Recommended because it matches your inferred taste from recent history."
        out.append({"item_id": int(it["item_id"]), "reason": reason})
    return out


def _build_prompt(user_id: int, recent_titles: List[str], items: List[Dict[str, Any]]) -> str:
    recent = recent_titles[:10]
    items_compact = [
        {"item_id": int(x["item_id"]), "title": (x.get("title") or ""), "genres": (x.get("genres") or "")}
        for x in items
    ]

    return (
        "You are an assistant that writes short, non-hallucinated recommendation explanations.\n"
        "Rules:\n"
        "- Use ONLY the provided user recent titles and item metadata (title/genres).\n"
        "- Do NOT invent facts (actors, plots, awards).\n"
        "- Output JSON ONLY in the specified schema.\n"
        "- Each reason must be <= 20 words.\n\n"
        f"USER_ID: {user_id}\n"
        f"RECENT_TITLES: {json.dumps(recent, ensure_ascii=False)}\n"
        f"ITEMS: {json.dumps(items_compact, ensure_ascii=False)}\n\n"
        "Return JSON with schema:\n"
        '{ "explanations": [ { "item_id": 123, "reason": "..." }, ... ] }\n'
    )


def explain_with_ollama(cfg: OllamaConfig, user_id: int, recent_titles: List[str], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = _build_prompt(user_id, recent_titles, items)
    prompt_sha = _sha256(prompt)

    url = cfg.host.rstrip("/") + "/api/generate"
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_predict": cfg.max_tokens,
        },
    }

    try:
        r = requests.post(url, json=payload, timeout=cfg.timeout_sec)
        r.raise_for_status()
        data = r.json()
        text = (data.get("response") or "").strip()

        # Try to parse JSON
        parsed: Optional[dict] = None
        try:
            parsed = json.loads(text)
        except Exception:
            # Sometimes models wrap JSON in text; attempt extraction
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(text[start : end + 1])

        if not isinstance(parsed, dict) or "explanations" not in parsed:
            return {"used": False, "prompt_sha256": prompt_sha, "error": "Invalid JSON from model", "explanations": None}

        exps = parsed.get("explanations")
        if not isinstance(exps, list) or not exps:
            return {"used": False, "prompt_sha256": prompt_sha, "error": "Empty explanations", "explanations": None}

        # Validate fields
        cleaned = []
        item_set = {int(x["item_id"]) for x in items}
        for e in exps:
            if not isinstance(e, dict):
                continue
            if "item_id" not in e or "reason" not in e:
                continue
            iid = int(e["item_id"])
            if iid not in item_set:
                continue
            reason = str(e["reason"]).strip()
            if not reason:
                continue
            cleaned.append({"item_id": iid, "reason": reason[:140]})

        if not cleaned:
            return {"used": False, "prompt_sha256": prompt_sha, "error": "No valid explanations after validation", "explanations": None}

        return {"used": True, "prompt_sha256": prompt_sha, "error": None, "explanations": cleaned}

    except Exception as e:
        return {"used": False, "prompt_sha256": prompt_sha, "error": f"{type(e).__name__}: {e}", "explanations": None}
