from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", "artifacts/bundle"))


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


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text())


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
    # scores = item_factors @ uvec
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

    # Map to item_ids and filter recently seen items (last N=200)
    seen = rt.user_recent.get(user_id, set())
    cand_item_ids = []
    cand_scores = []
    for it_idx, s in zip(cand_idx, cand_als_scores):
        item_id = rt.index_to_item_id[int(it_idx)]
        if item_id in seen:
            continue
        cand_item_ids.append(item_id)
        cand_scores.append(float(s))
        if len(cand_item_ids) >= topn:
            break

    # If still too few, fill with popularity
    if len(cand_item_ids) < topn:
        for it in rt.pop_items:
            if it in seen or it in cand_item_ids:
                continue
            cand_item_ids.append(int(it))
            cand_scores.append(0.0)
            if len(cand_item_ids) >= topn:
                break

    # Build feature matrix
    # User features (defaults to zeros if missing)
    if user_id in rt.user_feat.index:
        uf = rt.user_feat.loc[user_id]
    else:
        uf = pd.Series({c: 0.0 for c in rt.user_feat.columns})

    rows = []
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
    X["final_score"] = X["ranker_score"]  # place for future calibration/constraints

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
