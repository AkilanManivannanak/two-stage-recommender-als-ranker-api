from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from implicit.als import AlternatingLeastSquares


@dataclass(frozen=True)
class ALSConfig:
    factors: int = 64
    regularization: float = 0.08
    iterations: int = 20
    alpha: float = 40.0
    random_state: int = 42


def build_mappings(train_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int], np.ndarray, np.ndarray]:
    users = np.sort(train_df["user_id"].unique())
    items = np.sort(train_df["item_id"].unique())
    user_to_idx = {int(u): int(i) for i, u in enumerate(users)}
    item_to_idx = {int(it): int(i) for i, it in enumerate(items)}
    return user_to_idx, item_to_idx, users, items


def build_user_item_matrix(train_df: pd.DataFrame, user_to_idx: Dict[int, int], item_to_idx: Dict[int, int]) -> csr_matrix:
    u = train_df["user_id"].map(user_to_idx).to_numpy()
    it = train_df["item_id"].map(item_to_idx).to_numpy()
    w = train_df["weight"].to_numpy(dtype=np.float32)
    mat = coo_matrix((w, (u, it)), shape=(len(user_to_idx), len(item_to_idx))).tocsr()
    return mat


def train_als(train_path: str, out_dir: str, cfg: ALSConfig) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(train_path)

    user_to_idx, item_to_idx, users, items = build_mappings(train_df)
    X_ui = build_user_item_matrix(train_df, user_to_idx, item_to_idx)

    # implicit expects item-user for training
    X_iu = (X_ui * cfg.alpha).T.tocsr()

    model = AlternatingLeastSquares(
        factors=cfg.factors,
        regularization=cfg.regularization,
        iterations=cfg.iterations,
        random_state=cfg.random_state,
    )
    model.fit(X_iu)

    # Save factors + mappings
    np.save(out / "user_factors.npy", model.user_factors)
    np.save(out / "item_factors.npy", model.item_factors)

    mappings = {
        "user_id_to_index": user_to_idx,
        "item_id_to_index": item_to_idx,
        "index_to_user_id": users.tolist(),
        "index_to_item_id": items.tolist(),
    }
    (out / "mappings.json").write_text(json.dumps(mappings))

    config = {
        "factors": cfg.factors,
        "regularization": cfg.regularization,
        "iterations": cfg.iterations,
        "alpha": cfg.alpha,
        "random_state": cfg.random_state,
        "train_path": train_path,
        "n_users": len(users),
        "n_items": len(items),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return out
