from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from recsys.baselines.popularity import fit_popularity, recommend_popularity
from recsys.baselines.cooccurrence import build_item_cooccurrence, recommend_cooccurrence
from recsys.eval.bootstrap_ci import per_user_metrics, bootstrap_ci, bootstrap_delta_ci
from recsys.eval.metrics import evaluate_rankings


def load_truth(holdout_path: str) -> Dict[int, set[int]]:
    df = pd.read_parquet(holdout_path)
    truth = {}
    for u, g in df.groupby("user_id")["item_id"]:
        truth[int(u)] = set(int(x) for x in g.tolist())
    return truth


def rankings_from_df(df: pd.DataFrame, score_col: str, k: int = 50) -> Dict[int, List[int]]:
    d = df.sort_values(["user_id", score_col], ascending=[True, False]).groupby("user_id").head(k)
    out: Dict[int, List[int]] = {}
    for u, g in d.groupby("user_id")["item_id"]:
        out[int(u)] = [int(x) for x in g.tolist()]
    return out


def main() -> None:
    Path("reports").mkdir(parents=True, exist_ok=True)

    # Load test truth/users
    eligible = pd.read_parquet("data/processed/eligible_users_test.parquet")["user_id"].tolist()
    truth = load_truth("data/processed/holdout_targets_test.parquet")

    # Baselines (recompute so we can bootstrap)
    train_hist = pd.read_parquet("data/processed/train.parquet")
    pop_rank = fit_popularity(train_hist)
    pop_recs = recommend_popularity(eligible, pop_rank, k=50)

    item_nbs = build_item_cooccurrence(train_hist, last_m_per_user=20, max_neighbors=200)
    co_recs = recommend_cooccurrence(eligible, train_hist, item_nbs, pop_rank, k=50, user_profile_last_m=10)

    # ALS-only and Ranker rankings from ranker_test.parquet
    test = pd.read_parquet("data/processed/ranker_test.parquet")
    if "ranker_score" not in test.columns:
        # If not persisted, load from ranker_metrics.json? We prefer recompute by reloading model.
        # But your train_ranker_lgbm script does not persist predictions. We'll recompute from saved model.
        import lightgbm as lgb
        model = lgb.Booster(model_file="artifacts/ranker_lgbm/model.txt")
        FEATURE_COLS = [
            "als_score","user_cnt_total","user_cnt_7d","user_cnt_30d","user_tenure_days","user_recency_days",
            "item_cnt_total","item_cnt_7d","item_cnt_30d","item_age_days","item_recency_days"
        ]
        test = test.copy()
        test["ranker_score"] = model.predict(test[FEATURE_COLS])

    als_rank = rankings_from_df(test, "als_score", k=50)
    lgbm_rank = rankings_from_df(test, "ranker_score", k=50)

    # Per-user metrics
    pu_pop = per_user_metrics(pop_recs, truth)
    pu_co = per_user_metrics(co_recs, truth)
    pu_als = per_user_metrics(als_rank, truth)
    pu_lgbm = per_user_metrics(lgbm_rank, truth)

    # CIs
    ci = {
        "popularity": bootstrap_ci(pu_pop, n_boot=1000, seed=42),
        "cooccurrence": bootstrap_ci(pu_co, n_boot=1000, seed=42),
        "als_only": bootstrap_ci(pu_als, n_boot=1000, seed=42),
        "als_plus_lambdarank": bootstrap_ci(pu_lgbm, n_boot=1000, seed=42),
        "delta": {
            "lgbm_minus_als_ndcg10": bootstrap_delta_ci(pu_als, pu_lgbm, "ndcg10", n_boot=1000, seed=42),
            "lgbm_minus_als_mrr10": bootstrap_delta_ci(pu_als, pu_lgbm, "mrr10", n_boot=1000, seed=42),
            "lgbm_minus_co_ndcg10": bootstrap_delta_ci(pu_co, pu_lgbm, "ndcg10", n_boot=1000, seed=42),
            "lgbm_minus_co_mrr10": bootstrap_delta_ci(pu_co, pu_lgbm, "mrr10", n_boot=1000, seed=42),
        },
        "n_users": len(set(pu_lgbm.keys())),
    }

    Path("reports/ranker_metrics_ci.json").write_text(json.dumps(ci, indent=2))

    # Markdown
    def row(name: str, d: dict) -> str:
        return (
            f"| {name} | "
            f"{d['ndcg10']['mean']:.6f} [{d['ndcg10']['ci95_lo']:.6f},{d['ndcg10']['ci95_hi']:.6f}] | "
            f"{d['mrr10']['mean']:.6f} [{d['mrr10']['ci95_lo']:.6f},{d['mrr10']['ci95_hi']:.6f}] | "
            f"{d['recall10']['mean']:.6f} [{d['recall10']['ci95_lo']:.6f},{d['recall10']['ci95_hi']:.6f}] | "
            f"{d['recall50']['mean']:.6f} [{d['recall50']['ci95_lo']:.6f},{d['recall50']['ci95_hi']:.6f}] |"
        )

    md = []
    md.append("# Ranker Metrics with 95% Bootstrap CI (TEST)\n")
    md.append(f"- Users bootstrapped: **{ci['n_users']}** (user-level bootstrap, 1000 resamples)\n")
    md.append("| Model | NDCG@10 (mean [lo,hi]) | MRR@10 (mean [lo,hi]) | Recall@10 (mean [lo,hi]) | Recall@50 (mean [lo,hi]) |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(row("Popularity", ci["popularity"]))
    md.append(row("Cooccurrence", ci["cooccurrence"]))
    md.append(row("ALS (no rerank)", ci["als_only"]))
    md.append(row("ALS + LambdaRank", ci["als_plus_lambdarank"]))

    md.append("\n## Lift deltas (B - A) with 95% CI\n")
    for k, v in ci["delta"].items():
        md.append(f"- **{k}**: {v['delta_mean']:.6f} [{v['ci95_lo']:.6f},{v['ci95_hi']:.6f}] (n_users={v['n_users']})")

    Path("reports/ranker_metrics_ci.md").write_text("\n".join(md) + "\n")
    print("[OK] wrote reports/ranker_metrics_ci.json and reports/ranker_metrics_ci.md")


if __name__ == "__main__":
    main()
