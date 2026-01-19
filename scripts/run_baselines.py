from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recsys.baselines.cooccurrence import build_item_cooccurrence, recommend_cooccurrence
from recsys.baselines.popularity import fit_popularity, recommend_popularity
from recsys.eval.metrics import evaluate_rankings


def load_truth(holdout_path: str) -> dict[int, set[int]]:
    df = pd.read_parquet(holdout_path)
    truth = {}
    for u, g in df.groupby("user_id")["item_id"]:
        truth[int(u)] = set(int(x) for x in g.tolist())
    return truth


def main() -> None:
    train = pd.read_parquet("data/processed/train.parquet")
    eligible_test = pd.read_parquet("data/processed/eligible_users_test.parquet")["user_id"].tolist()
    truth_test = load_truth("data/processed/holdout_targets_test.parquet")

    # Popularity
    pop_rank = fit_popularity(train)
    pop_recs = recommend_popularity(eligible_test, pop_rank, k=50)
    pop_res = evaluate_rankings(pop_recs, truth_test, k_cov=10)

    # Cooccurrence (restricted for compute)
    item_nbs = build_item_cooccurrence(train, last_m_per_user=20, max_neighbors=200)
    co_recs = recommend_cooccurrence(
        eligible_test, train, item_nbs, pop_rank, k=50, user_profile_last_m=10
    )
    co_res = evaluate_rankings(co_recs, truth_test, k_cov=10)

    metrics = {
        "split": "test",
        "n_users_eval": int(pop_res.n_users),
        "popularity": {
            "ndcg10": pop_res.ndcg10,
            "mrr10": pop_res.mrr10,
            "recall10": pop_res.recall10,
            "recall50": pop_res.recall50,
            "coverage10": pop_res.coverage10,
        },
        "cooccurrence": {
            "ndcg10": co_res.ndcg10,
            "mrr10": co_res.mrr10,
            "recall10": co_res.recall10,
            "recall50": co_res.recall50,
            "coverage10": co_res.coverage10,
        },
        "notes": {
            "cooccurrence_last_m_per_user_build": 20,
            "cooccurrence_user_profile_last_m": 10,
            "k_ranked": 50,
        },
    }

    Path("reports").mkdir(exist_ok=True, parents=True)
    Path("reports/baselines_metrics.json").write_text(json.dumps(metrics, indent=2))

    md = []
    md.append("# Baseline Metrics (TEST)\n")
    md.append(f"- Eligible test users evaluated: **{metrics['n_users_eval']}**\n")
    md.append("Cooccurrence is computed from each user's last-M train items (restricted for local compute).\n")
    md.append("\n")
    md.append("| Model | NDCG@10 | MRR@10 | Recall@10 | Recall@50 | Coverage@10 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    md.append(
        f"| Popularity | {metrics['popularity']['ndcg10']:.6f} | {metrics['popularity']['mrr10']:.6f} | "
        f"{metrics['popularity']['recall10']:.6f} | {metrics['popularity']['recall50']:.6f} | "
        f"{int(metrics['popularity']['coverage10'])} |"
    )
    md.append(
        f"| Cooccurrence | {metrics['cooccurrence']['ndcg10']:.6f} | {metrics['cooccurrence']['mrr10']:.6f} | "
        f"{metrics['cooccurrence']['recall10']:.6f} | {metrics['cooccurrence']['recall50']:.6f} | "
        f"{int(metrics['cooccurrence']['coverage10'])} |"
    )

    Path("reports/baselines_metrics.md").write_text("\n".join(md) + "\n")
    print("[OK] wrote reports/baselines_metrics.json and reports/baselines_metrics.md")


if __name__ == "__main__":
    main()
