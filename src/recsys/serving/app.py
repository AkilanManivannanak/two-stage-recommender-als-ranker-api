from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recsys.serving.llm_explain import OllamaConfig, explain_with_ollama, fallback_explanations


BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", "artifacts/bundle"))


# -----------------------
# API Schemas
# -----------------------
class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations")


class ScoredItem(BaseModel):
    item_id: int
    score: float
    als_score: float
    ranker_score: float


class RecommendResponse(BaseModel):
    user_id: int
    k: int
    items: List[ScoredItem]
    model_version: Dict[str, Any]


class ExplainRequest(BaseModel):
    user_id: int
    item_ids: List[int] = Field(..., min_length=1, max_length=50)


class ExplainItem(BaseModel):
    item_id: int
    reason: str


class ExplainResponse(BaseModel):
    user_id: int
    model: Dict[str, Any]
    explanations: List[ExplainItem]


# -----------------------
# Runtime
# -----------------------
@dataclass
class Runtime:
    manifest: Dict[str, Any]
    feature_cols: List[str]
    serving_cfg: Dict[str, Any]

    # ALS artifacts
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_id_to_index: Dict[int, int]
    index_to_item_id: List[int]

    # Ranker
    ranker: lgb.Booster

    # Features
    user_feat: pd.DataFrame  # indexed by user_id
    item_feat: pd.DataFrame  # indexed by item_id
    pop_items: List[int]
    user_recent: Dict[int, set[int]]  # user_id -> set(item_id)

    # Metadata for explanations (optional but recommended)
    item_meta: Dict[int, Dict[str, str]]  # item_id -> {title, genres}
    user_recent_titles: Dict[int, List[str]]  # user_id -> list of titles


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _safe_read_parquet(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _load_runtime(bundle_dir: Path) -> Runtime:
    if not bundle_dir.exists():
        raise RuntimeError(f"Bundle directory not found: {bundle_dir}")

    manifest = _load_json(bundle_dir / "manifest.json")
    feature_spec = _load_json(bundle_dir / "feature_spec.json")
    serving_cfg = _load_json(bundle_dir / "serving_config.json")
    feature_cols = feature_spec["feature_cols"]

    # ALS
    als_dir = bundle_dir / "als"
    mappings = _load_json(als_dir / "mappings.json")

    user_factors = np.load(als_dir / "user_factors.npy")
    item_factors = np.load(als_dir / "item_factors.npy")

    n_users = len(mappings["index_to_user_id"])
    n_items = len(mappings["index_to_item_id"])

    # safety: swapped factors
    if user_factors.shape[0] == n_items and item_factors.shape[0] == n_users:
        user_factors, item_factors = item_factors, user_factors

    if user_factors.shape[0] != n_users or item_factors.shape[0] != n_items:
        raise RuntimeError(
            f"ALS factor shapes mismatch mappings: user_factors={user_factors.shape}, item_factors={item_factors.shape}, "
            f"n_users={n_users}, n_items={n_items}"
        )

    user_id_to_index = {int(k): int(v) for k, v in mappings["user_id_to_index"].items()}
    index_to_item_id = [int(x) for x in mappings["index_to_item_id"]]

    # Ranker
    ranker = lgb.Booster(model_file=str(bundle_dir / "ranker" / "model.txt"))

    # Features
    user_feat = pd.read_parquet(bundle_dir / "features" / "user_features.parquet").set_index("user_id")
    item_feat = pd.read_parquet(bundle_dir / "features" / "item_features.parquet").set_index("item_id")
    pop_items = _load_json(bundle_dir / "features" / "popularity.json")["items"]

    recent_df = pd.read_parquet(bundle_dir / "features" / "user_recent_items.parquet")
    user_recent: Dict[int, set[int]] = {
        int(r.user_id): set(int(x) for x in r.item_id) for r in recent_df.itertuples(index=False)
    }

    # Optional metadata (for /explain)
    item_meta: Dict[int, Dict[str, str]] = {}
    user_recent_titles: Dict[int, List[str]] = {}

    md_df = _safe_read_parquet(bundle_dir / "features" / "item_metadata.parquet")
    if md_df is not None:
        # expected cols: item_id,title,genres
        for r in md_df.itertuples(index=False):
            iid = int(getattr(r, "item_id"))
            title = str(getattr(r, "title", "") or "")
            genres = str(getattr(r, "genres", "") or "")
            item_meta[iid] = {"title": title, "genres": genres}

    # build user_recent_titles from user_recent items + item_meta
    if item_meta:
        for uid, seen in user_recent.items():
            titles = []
            for iid in list(seen)[:50]:
                m = item_meta.get(int(iid))
                if m and m.get("title"):
                    titles.append(m["title"])
            user_recent_titles[int(uid)] = titles

    return Runtime(
        manifest=manifest,
        feature_cols=feature_cols,
        serving_cfg=serving_cfg,
        user_factors=user_factors.astype(np.float32),
        item_factors=item_factors.astype(np.float32),
        user_id_to_index=user_id_to_index,
        index_to_item_id=index_to_item_id,
        ranker=ranker,
        user_feat=user_feat,
        item_feat=item_feat,
        pop_items=pop_items,
        user_recent=user_recent,
        item_meta=item_meta,
        user_recent_titles=user_recent_titles,
    )


rt: Optional[Runtime] = None
app = FastAPI(title="Two-Stage Recommender API", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    global rt
    rt = _load_runtime(BUNDLE_DIR)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> Dict[str, Any]:
    if rt is None:
        raise HTTPException(status_code=500, detail="Runtime not loaded")
    return rt.manifest


def _candidate_scores_for_user(uvec: np.ndarray, item_factors: np.ndarray, topn: int) -> tuple[np.ndarray, np.ndarray]:
    scores = item_factors @ uvec
    if topn >= scores.shape[0]:
        idx = np.argsort(-scores)
    else:
        idx_part = np.argpartition(-scores, topn)[:topn]
        idx = idx_part[np.argsort(-scores[idx_part])]
    return idx.astype(np.int64), scores[idx].astype(np.float32)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if rt is None:
        raise HTTPException(status_code=500, detail="Runtime not loaded")

    user_id = int(req.user_id)
    k = int(req.k)
    topn = int(rt.serving_cfg.get("topn_candidates", 500))

    # Unknown user -> popularity fallback
    if user_id not in rt.user_id_to_index:
        items = [ScoredItem(item_id=int(it), score=0.0, als_score=0.0, ranker_score=0.0) for it in rt.pop_items[:k]]
        return RecommendResponse(user_id=user_id, k=k, items=items, model_version=rt.manifest)

    uidx = rt.user_id_to_index[user_id]
    uvec = rt.user_factors[uidx]

    cand_idx, cand_als_scores = _candidate_scores_for_user(uvec, rt.item_factors, topn=topn)

    # Map to item_ids and filter recently seen items
    seen = rt.user_recent.get(user_id, set())
    cand_item_ids: List[int] = []
    cand_scores: List[float] = []
    for it_idx, s in zip(cand_idx, cand_als_scores):
        item_id = rt.index_to_item_id[int(it_idx)]
        if item_id in seen:
            continue
        cand_item_ids.append(int(item_id))
        cand_scores.append(float(s))
        if len(cand_item_ids) >= topn:
            break

    # If still too few, fill with popularity
    if len(cand_item_ids) < topn:
        for it in rt.pop_items:
            it = int(it)
            if it in seen or it in cand_item_ids:
                continue
            cand_item_ids.append(it)
            cand_scores.append(0.0)
            if len(cand_item_ids) >= topn:
                break

    # User features
    if user_id in rt.user_feat.index:
        uf = rt.user_feat.loc[user_id]
    else:
        uf = pd.Series({c: 0.0 for c in rt.user_feat.columns})

    rows: List[Dict[str, Any]] = []
    for item_id, als_s in zip(cand_item_ids, cand_scores):
        if item_id in rt.item_feat.index:
            itf = rt.item_feat.loc[item_id]
        else:
            itf = pd.Series({c: 0.0 for c in rt.item_feat.columns})

        row = {
            "als_score": float(als_s),
            "user_cnt_total": float(uf.get("user_cnt_total", 0.0)),
            "user_cnt_7d": float(uf.get("user_cnt_7d", 0.0)),
            "user_cnt_30d": float(uf.get("user_cnt_30d", 0.0)),
            "user_tenure_days": float(uf.get("user_tenure_days", 0.0)),
            "user_recency_days": float(uf.get("user_recency_days", 0.0)),
            "item_cnt_total": float(itf.get("item_cnt_total", 0.0)),
            "item_cnt_7d": float(itf.get("item_cnt_7d", 0.0)),
            "item_cnt_30d": float(itf.get("item_cnt_30d", 0.0)),
            "item_age_days": float(itf.get("item_age_days", 0.0)),
            "item_recency_days": float(itf.get("item_recency_days", 0.0)),
            "item_id": int(item_id),
            "als_score_out": float(als_s),
        }
        rows.append(row)

    X = pd.DataFrame(rows)
    ranker_scores = rt.ranker.predict(X[rt.feature_cols])

    X["ranker_score"] = ranker_scores.astype(float)
    X["final_score"] = X["ranker_score"]

    X = X.sort_values("final_score", ascending=False).head(k)

    items = [
        ScoredItem(
            item_id=int(r.item_id),
            score=float(r.final_score),
            als_score=float(r.als_score_out),
            ranker_score=float(r.ranker_score),
        )
        for r in X.itertuples(index=False)
    ]

    return RecommendResponse(user_id=user_id, k=k, items=items, model_version=rt.manifest)


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    """
    LLM explanations (Option A).
    - Deterministic: temperature=0
    - Safe: timeout + fallback
    - Uses item titles/genres when available
    """
    if rt is None:
        raise HTTPException(status_code=500, detail="Runtime not loaded")

    user_id = int(req.user_id)
    item_ids = [int(x) for x in req.item_ids]

    # Build item payload with metadata (best-effort)
    items: List[Dict[str, Any]] = []
    for iid in item_ids:
        meta = rt.item_meta.get(iid, {})
        items.append(
            {
                "item_id": iid,
                "title": meta.get("title", ""),
                "genres": meta.get("genres", ""),
            }
        )

    recent_titles = rt.user_recent_titles.get(user_id, [])

    cfg = OllamaConfig(
        host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        timeout_sec=float(os.getenv("OLLAMA_TIMEOUT_SEC", "2.0")),
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "120")),
        use_structured_output=os.getenv("OLLAMA_STRUCTURED", "1") == "1",
    )

    t0 = time.time()
    llm = explain_with_ollama(cfg, user_id, recent_titles, items)
    latency_ms = int((time.time() - t0) * 1000)

    exps = llm.get("explanations") if isinstance(llm, dict) else None
    if not llm.get("used", False) or not exps:
        # Fallback deterministic explanations
        exps = fallback_explanations(items)

    model_info = {
        "provider": "ollama",
        "model": cfg.model,
        "used": bool(llm.get("used", False)),
        "latency_ms": latency_ms,
        "prompt_sha256": llm.get("prompt_sha256"),
        "error": llm.get("error"),
        "note": "Fallback used if Ollama unavailable or output invalid.",
    }

    # Normalize schema
    out_exps = [{"item_id": int(e["item_id"]), "reason": str(e["reason"])} for e in exps]

    return ExplainResponse(user_id=user_id, model=model_info, explanations=out_exps)
