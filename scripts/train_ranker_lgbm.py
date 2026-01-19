from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

from recsys.eval.metrics import evaluate_rankings


FEATURE_COLS = [
    "als_score",
    "user_cnt_total",
    "user_cnt_7d",
    "user_cnt_30d",
    "user_tenure_days",
    "user_recency_days",
    "item_cnt_total",
    "item_cnt_7d",
    "item_cnt_30d",
    "item_age_days",
    "item_recency_days",
]


# Simple diversity-aware calibration (post-ranker):
# Penalize very popular items a bit to increase coverage / long-tail exposure.
LAMBDA_POP_PENALTY = 0.02


def load_truth(holdout_path: str) -> Dict[int, set[int]]:
    df = pd.read_parquet(holdout_path)
    truth: Dict[int, set[int]] = {}
    for u, g in df.groupby("user_id")["item_id"]:
        truth[int(u)] = set(int(x) for x in g.tolist())
    return truth


def to_group_sizes(df: pd.DataFrame) -> List[int]:
    # LightGBM expects group sizes in the same order as rows.
    return df.groupby("group_id", sort=False).size().astype(int).tolist()


def rankings_from_df(df: pd.DataFrame, score_col: str, k: int = 50) -> Dict[int, List[int]]:
    d = df.sort_values(["user_id", score_col], ascending=[True, False]).groupby("user_id").head(k)
    out: Dict[int, List[int]] = {}
    for u, g in d.groupby("user_id")["item_id"]:
        out[int(u)] = [int(x) for x in g.tolist()]
    return out


def main() -> None:
    Path("artifacts/ranker_lgbm").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # Load datasets
    train = pd.read_parquet("data/processed/ranker_train.parquet")
    val = pd.read_parquet("data/processed/ranker_val.parquet")
    test = pd.read_parquet("data/processed/ranker_test.parquet")

    # Sanity: enforce sort by group (required for group sizes)
    train = train.sort_values(["group_id"]).reset_index(drop=True)
    val = val.sort_values(["group_id"]).reset_index(drop=True)

    X_train = train[FEATURE_COLS]
    y_train = train["label"].astype(int)
    g_train = to_group_sizes(train)

    X_val = val[FEATURE_COLS]
    y_val = val["label"].astype(int)
    g_val = to_group_sizes(val)

    dtrain = lgb.Dataset(X_train, label=y_train, group=g_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, group=g_val, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "force_row_wise": True,
    }

    print("[1/3] Training LambdaRank (LightGBM)...")
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )

    model_path = Path("artifacts/ranker_lgbm/model.txt")
    booster.save_model(str(model_path))
    print(f"[OK] saved model -> {model_path}")

    # Predict on TEST
    print("[2/3] Predicting on ranker_test.parquet...")
    test = test.copy()
    test["ranker_score"] = booster.predict(test[FEATURE_COLS], num_iteration=booster.best_iteration)

    # Calibrated score to reduce popularity bias
    # final_score = ranker_score - lambda * log1p(item_cnt_total)
    test["ranker_score_cal"] = test["ranker_score"] - (LAMBDA_POP_PENALTY * np.log1p(test["item_cnt_total"].astype(float)))

    # Build rankings
    als_rankings = rankings_from_df(test, "als_score", k=50)
    lgbm_rankings = rankings_from_df(test, "ranker_score", k=50)
    lgbm_cal_rankings = rankings_from_df(test, "ranker_score_cal", k=50)

    # Truth from holdout (TEST window)
    truth_test = load_truth("data/processed/holdout_targets_test.parquet")

    # Evaluate
    print("[3/3] Evaluating rankings...")
    als_res = evaluate_rankings(als_rankings, truth_test, k_cov=10)
    lgbm_res = evaluate_rankings(lgbm_rankings, truth_test, k_cov=10)
    lgbm_cal_res = evaluate_rankings(lgbm_cal_rankings, truth_test, k_cov=10)

    # Load baseline metrics if present
    baselines = {}
    bp = Path("reports/baselines_metrics.json")
    if bp.exists():
        baselines = json.loads(bp.read_text())

    out = {
        "split": "test",
        "n_users_eval": int(lgbm_res.n_users),
        "baselines": baselines,
        "als_only": {
            "ndcg10": als_res.ndcg10,
            "mrr10": als_res.mrr10,
            "recall10": als_res.recall10,
            "recall50": als_res.recall50,
            "coverage10": als_res.coverage10,
        },
        "als_plus_lambdarank": {
            "ndcg10": lgbm_res.ndcg10,
            "mrr10": lgbm_res.mrr10,
            "recall10": lgbm_res.recall10,
            "recall50": lgbm_res.recall50,
            "coverage10": lgbm_res.coverage10,
        },
        "als_plus_lambdarank_calibrated": {
            "ndcg10": lgbm_cal_res.ndcg10,
            "mrr10": lgbm_cal_res.mrr10,
            "recall10": lgbm_cal_res.recall10,
            "recall50": lgbm_cal_res.recall50,
            "coverage10": lgbm_cal_res.coverage10,
            "lambda_pop_penalty": LAMBDA_POP_PENALTY,
            "calibration": "ranker_score - lambda * log1p(item_cnt_total)",
        },
        "model": {
            "best_iteration": int(booster.best_iteration),
            "features": FEATURE_COLS,
            "params": params,
        },
    }

    Path("reports/ranker_metrics.json").write_text(json.dumps(out, indent=2))

    # Markdown table
    md: List[str] = []
    md.append("# Ranker Metrics (TEST)\n")
    md.append(f"- Eligible test users evaluated: **{out['n_users_eval']}**\n")

    md.append("\n## Offline ranking metrics\n")
    md.append("| Model | NDCG@10 | MRR@10 | Recall@10 | Recall@50 | Coverage@10 |")
    md.append("|---|---:|---:|---:|---:|---:|")

    # Baselines (if available)
    if baselines and "popularity" in baselines:
        b = baselines["popularity"]
        md.append(
            f"| Popularity | {b['ndcg10']:.6f} | {b['mrr10']:.6f} | {b['recall10']:.6f} | {b['recall50']:.6f} | {int(b['coverage10'])} |"
        )
    if baselines and "cooccurrence" in baselines:
        b = baselines["cooccurrence"]
        md.append(
            f"| Cooccurrence | {b['ndcg10']:.6f} | {b['mrr10']:.6f} | {b['recall10']:.6f} | {b['recall50']:.6f} | {int(b['coverage10'])} |"
        )

    a = out["als_only"]
    md.append(
        f"| ALS (no rerank) | {a['ndcg10']:.6f} | {a['mrr10']:.6f} | {a['recall10']:.6f} | {a['recall50']:.6f} | {int(a['coverage10'])} |"
    )

    r = out["als_plus_lambdarank"]
    md.append(
        f"| ALS + LambdaRank | {r['ndcg10']:.6f} | {r['mrr10']:.6f} | {r['recall10']:.6f} | {r['recall50']:.6f} | {int(r['coverage10'])} |"
    )

    rc = out["als_plus_lambdarank_calibrated"]
    md.append(
        f"| ALS + LambdaRank (calibrated, λ={rc['lambda_pop_penalty']}) | {rc['ndcg10']:.6f} | {rc['mrr10']:.6f} | {rc['recall10']:.6f} | {rc['recall50']:.6f} | {int(rc['coverage10'])} |"
    )

    Path("reports/ranker_metrics.md").write_text("\n".join(md) + "\n")

    print("[OK] wrote reports/ranker_metrics.json and reports/ranker_metrics.md")


if __name__ == "__main__":
    main()
