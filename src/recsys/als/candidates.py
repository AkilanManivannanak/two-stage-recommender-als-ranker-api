from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix

from implicit.als import AlternatingLeastSquares


@dataclass(frozen=True)
class CandidateConfig:
    topn: int = 500
    alpha: float = 40.0


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_als_artifacts(als_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    als_dir = Path(als_dir)
    user_factors = np.load(als_dir / "user_factors.npy")
    item_factors = np.load(als_dir / "item_factors.npy")
    mappings = _load_json(als_dir / "mappings.json")

    n_users = len(mappings["index_to_user_id"])
    n_items = len(mappings["index_to_item_id"])

    # Safety: some implicit training setups can swap these. Fix in memory.
    if user_factors.shape[0] == n_items and item_factors.shape[0] == n_users:
        user_factors, item_factors = item_factors, user_factors

    # Hard check (fail loudly if still inconsistent)
    if user_factors.shape[0] != n_users or item_factors.shape[0] != n_items:
        raise ValueError(
            f"Factor shapes don't match mappings. "
            f"Expected user_factors[{n_users},*], item_factors[{n_items},*] "
            f"but got user_factors={user_factors.shape}, item_factors={item_factors.shape}"
        )

    return user_factors, item_factors, mappings


def build_user_item_matrix_from_train(train_df: pd.DataFrame, user_to_idx: Dict[int, int], item_to_idx: Dict[int, int], alpha: float) -> csr_matrix:
    u = train_df["user_id"].map(user_to_idx).to_numpy()
    it = train_df["item_id"].map(item_to_idx).to_numpy()
    w = train_df["weight"].to_numpy(dtype=np.float32)
    mat = coo_matrix((w * alpha, (u, it)), shape=(len(user_to_idx), len(item_to_idx))).tocsr()
    return mat


def generate_candidates(
    train_path: str,
    eligible_users: List[int],
    als_dir: str,
    out_path: str,
    cfg: CandidateConfig,
) -> Path:
    train_df = pd.read_parquet(train_path)

    user_factors, item_factors, mappings = load_als_artifacts(als_dir)
    user_to_idx = {int(k): int(v) for k, v in mappings["user_id_to_index"].items()}
    item_to_idx = {int(k): int(v) for k, v in mappings["item_id_to_index"].items()}
    idx_to_item = [int(x) for x in mappings["index_to_item_id"]]

    # Recreate model container for recommend()
    model = AlternatingLeastSquares(factors=user_factors.shape[1])
    model.user_factors = user_factors
    model.item_factors = item_factors

    X_ui_alpha = build_user_item_matrix_from_train(train_df, user_to_idx, item_to_idx, alpha=cfg.alpha)
    # recommend() expects user_items in CSR: user x items
    user_items = X_ui_alpha

    rows = []
    for u in eligible_users:
        u = int(u)
        if u not in user_to_idx:
            continue
        uidx = user_to_idx[u]
        recs, scores = model.recommend(uidx, user_items[uidx], N=cfg.topn, filter_already_liked_items=True)
        for it_idx, s in zip(recs, scores):
            rows.append((u, idx_to_item[int(it_idx)], float(s)))

    cand = pd.DataFrame(rows, columns=["user_id", "item_id", "als_score"])
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(outp, index=False)
    return outp


def candidate_recall_at_k(candidates_df: pd.DataFrame, holdout_df: pd.DataFrame, k: int) -> float:
    # candidates_df: user_id,item_id,als_score
    # holdout_df: user_id,item_id
    cand_sorted = candidates_df.sort_values(["user_id", "als_score"], ascending=[True, False])
    topk = cand_sorted.groupby("user_id").head(k)
    cand_sets = topk.groupby("user_id")["item_id"].apply(lambda x: set(int(i) for i in x.tolist())).to_dict()

    hold_sets = holdout_df.groupby("user_id")["item_id"].apply(lambda x: set(int(i) for i in x.tolist())).to_dict()

    users = [u for u in hold_sets.keys() if u in cand_sets and len(hold_sets[u]) > 0]
    if not users:
        return 0.0

    recalls = []
    for u in users:
        hit = len(cand_sets[u] & hold_sets[u])
        recalls.append(hit / len(hold_sets[u]))
    return float(np.mean(recalls))
