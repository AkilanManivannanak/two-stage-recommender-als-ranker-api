from __future__ import annotations
import http.client, json, ssl, unicodedata
from typing import Any


def _sanitize_str(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "--")
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def openai_post(path: str, payload: dict, api_key: str, timeout: int = 8) -> dict:
    normalised = _normalise_payload(payload)
    body = json.dumps(normalised, ensure_ascii=True).encode("utf-8")
    ctx  = ssl.create_default_context()
    conn = http.client.HTTPSConnection("api.openai.com", timeout=timeout, context=ctx)
    try:
        conn.request("POST", path, body=body, headers={
            "Content-Type":   "application/json; charset=utf-8",
            "Authorization":  f"Bearer {api_key}",
            "Content-Length": str(len(body)),
        })
        return json.loads(conn.getresponse().read().decode("utf-8"))
    finally:
        conn.close()


def tmdb_get(path: str, api_key: str, timeout: int = 5) -> dict:
    ctx  = ssl.create_default_context()
    conn = http.client.HTTPSConnection("api.themoviedb.org", timeout=timeout, context=ctx)
    try:
        conn.request("GET", path, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        })
        return json.loads(conn.getresponse().read().decode("utf-8"))
    finally:
        conn.close()


def _normalise_payload(obj: Any) -> Any:
    if isinstance(obj, str):
        return _sanitize_str(obj)
    if isinstance(obj, dict):
        return {k: _normalise_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalise_payload(i) for i in obj]
    return obj
