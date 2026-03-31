from __future__ import annotations
import json, os, unicodedata
from pathlib import Path
import numpy as np


def _sanitize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018","'").replace("\u2019","'")
    text = text.replace("\u201c",'"').replace("\u201d",'"')
    text = text.replace("\u2013","-").replace("\u2014","--")
    text = text.encode("ascii","ignore").decode("ascii")
    return text.strip()

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
_CACHE_FILE = Path(os.environ.get("ARTIFACTS_DIR", "artifacts")) / "embed_cache.json"
_EMBED_CACHE: dict = {}

def _load_cache():
    if _CACHE_FILE.exists():
        try: _EMBED_CACHE.update(json.loads(_CACHE_FILE.read_text(encoding='utf-8')))
        except: pass

def _save_cache():
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(_EMBED_CACHE, ensure_ascii=True), encoding='utf-8')
    except: pass

_load_cache()


class EmbeddingIndex:
    def __init__(self):
        self.ids = []; self.vecs = None; self.meta = {}

    def add(self, iid, vec, meta):
        self.ids.append(iid)
        v = vec / (np.linalg.norm(vec) + 1e-9)
        self.vecs = np.vstack([self.vecs, v[None]]) if self.vecs is not None else v[None]
        self.meta[iid] = meta

    def search(self, q, top_k=20):
        if self.vecs is None: return []
        q = q / (np.linalg.norm(q) + 1e-9)
        s = self.vecs @ q
        return [(self.ids[i], float(s[i])) for i in np.argsort(-s)[:top_k]]

_INDEX = EmbeddingIndex()


def _embed(text):
    if not _OPENAI_KEY: return None
    text = _sanitize(text)
    if text in _EMBED_CACHE:
        return np.array(_EMBED_CACHE[text], dtype=np.float32)
    try:
        # Use openai_post from _http.py -- handles UTF-8 encoding correctly
        from recsys.serving._http import openai_post
        r = openai_post(
            "/v1/embeddings",
            {"model": "text-embedding-3-small", "input": text},
            _OPENAI_KEY,
        )
        v = np.array(r["data"][0]["embedding"], dtype=np.float32)
        _EMBED_CACHE[text] = v.tolist()
        _save_cache()
        return v
    except Exception as e:
        print(f"  [RAG] Embed error: {e}")
        return None


def build_index(catalog):
    if not _OPENAI_KEY:
        print("  [RAG] No OPENAI_API_KEY -- skipping semantic index.")
        return
    print(f"  [RAG] Building semantic embedding index for {len(catalog)} titles...")
    n = 0
    for mid, item in catalog.items():
        t = _sanitize(item.get("title", ""))
        g = _sanitize(item.get("primary_genre", ""))
        d = _sanitize(item.get("description", ""))
        v = _embed(f"{t}. Genre: {g}. {d}")
        if v is not None:
            _INDEX.add(mid, v, {"title": t, "genre": g})
            n += 1
    print(f"  [RAG] Indexed {n}/{len(catalog)} titles")


def semantic_retrieve(user_titles, user_descriptions, top_k=50):
    if not _OPENAI_KEY or _INDEX.vecs is None: return []
    vecs = [v for t, d in zip(user_titles[:10], user_descriptions[:10])
            if (v := _embed(f"{t}. {d}")) is not None]
    if not vecs: return []
    return _INDEX.search(np.mean(np.stack(vecs), axis=0), top_k)


def llm_rerank(user_id, candidates, user_watch_history, top_k=10):
    for i, c in enumerate(candidates[:top_k]):
        c["rag_rank"] = i + 1
        c["rag_score"] = round(0.9 - i * 0.05, 3)
        c["rag_reason"] = "Semantic similarity"
    return candidates[:top_k]
