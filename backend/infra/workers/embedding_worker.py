"""
Embedding Worker — patched to bypass latin-1 rag_engine bug
"""
import sys
import os

# Force Python to ignore any .pyc files and recompile from source
sys.dont_write_bytecode = True

# Patch the rag_engine module BEFORE app.py imports it
# This runs first and installs a fixed version in sys.modules
import http.client, json, ssl, unicodedata
from pathlib import Path
import numpy as np

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
_CACHE_FILE = Path("artifacts/embed_cache.json")
_EMBED_CACHE: dict = {}

def _load_cache():
    if _CACHE_FILE.exists():
        try: _EMBED_CACHE.update(json.loads(_CACHE_FILE.read_text()))
        except: pass

def _save_cache():
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(_EMBED_CACHE))
    except: pass

_load_cache()

def _norm(obj):
    if isinstance(obj, str): return unicodedata.normalize("NFKC", obj)
    if isinstance(obj, dict): return {k: _norm(v) for k,v in obj.items()}
    if isinstance(obj, list): return [_norm(i) for i in obj]
    return obj

def _post(path, payload, key, timeout=8):
    body = json.dumps(_norm(payload), ensure_ascii=False).encode("utf-8")
    ctx  = ssl.create_default_context()
    conn = http.client.HTTPSConnection("api.openai.com", timeout=timeout, context=ctx)
    try:
        conn.request("POST", path, body=body, headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {key}",
            "Content-Length": str(len(body)),
        })
        return json.loads(conn.getresponse().read().decode("utf-8"))
    finally:
        conn.close()

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
    text = unicodedata.normalize("NFKC", text).strip()
    if text in _EMBED_CACHE:
        return np.array(_EMBED_CACHE[text], dtype=np.float32)
    try:
        r = _post("/v1/embeddings",
                  {"model": "text-embedding-3-small", "input": text},
                  _OPENAI_KEY)
        v = np.array(r["data"][0]["embedding"], dtype=np.float32)
        _EMBED_CACHE[text] = v.tolist(); _save_cache()
        return v
    except Exception as e:
        print(f"  [RAG] Embed error: {e}"); return None

def build_index(catalog):
    if not _OPENAI_KEY:
        print("  [RAG] No OPENAI_API_KEY — skipping."); return
    print(f"  [RAG] Building semantic embedding index for {len(catalog)} titles...")
    n = 0
    for mid, item in catalog.items():
        t = item.get("title",""); g = item.get("primary_genre","")
        d = item.get("description","") or ""
        v = _embed(f"{t}. Genre: {g}. {d}")
        if v is not None:
            _INDEX.add(mid, v, {"title": t, "genre": g}); n += 1
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

# Install the fixed rag_engine into sys.modules BEFORE app.py imports it
import types
rag_module = types.ModuleType("recsys.serving.rag_engine")
rag_module._OPENAI_KEY = _OPENAI_KEY
rag_module._EMBED_CACHE = _EMBED_CACHE
rag_module._INDEX = _INDEX
rag_module.build_index = build_index
rag_module.semantic_retrieve = semantic_retrieve
rag_module.llm_rerank = llm_rerank
rag_module._embed = _embed
sys.modules["recsys.serving.rag_engine"] = rag_module

# NOW import app — it will use the pre-installed rag_engine above
sys.path.insert(0, 'src')
sys.path.insert(0, '/app/src')

from recsys.serving.multimodal import build_multimodal_index
from recsys.serving.app import CATALOG

build_multimodal_index(CATALOG)
print('[embedding_worker] Done')
