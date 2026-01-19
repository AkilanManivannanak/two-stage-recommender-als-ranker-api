from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recsys.als.train_als import ALSConfig, train_als
from recsys.als.candidates import CandidateConfig, generate_candidates, candidate_recall_at_k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="data/processed/train.parquet")
    ap.add_argument("--eligible_users", type=str, default="data/processed/eligible_users_test.parquet")
    ap.add_argument("--holdout", type=str, default="data/processed/holdout_targets_test.parquet")
    ap.add_argument("--als_dir", type=str, default="artifacts/als")
    ap.add_argument("--candidates_out", type=str, default="data/processed/candidates.parquet")
    ap.add_argument("--topn", type=int, default=500)
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--reg", type=float, default=0.08)
    ap.add_argument("--alpha", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    als_cfg = ALSConfig(
        factors=args.factors,
        iterations=args.iterations,
        regularization=args.reg,
        alpha=args.alpha,
        random_state=args.seed,
    )

    print("[1/3] Training ALS...")
    als_dir = train_als(args.train_path, args.als_dir, als_cfg)
    print(f"[OK] ALS artifacts -> {als_dir}")

    eligible = pd.read_parquet(args.eligible_users)["user_id"].tolist()
    holdout_df = pd.read_parquet(args.holdout)

    print("[2/3] Generating candidates...")
    cand_cfg = CandidateConfig(topn=args.topn, alpha=args.alpha)
    cand_path = generate_candidates(args.train_path, eligible, args.als_dir, args.candidates_out, cand_cfg)
    print(f"[OK] Candidates -> {cand_path}")

    print("[3/3] Candidate recall metrics...")
    cand_df = pd.read_parquet(cand_path)

    r50 = candidate_recall_at_k(cand_df, holdout_df, k=50)
    r200 = candidate_recall_at_k(cand_df, holdout_df, k=200)

    metrics = {
        "candidate_recall@50": r50,
        "candidate_recall@200": r200,
        "topn_generated": args.topn,
        "eligible_users": len(eligible),
    }
    Path("reports").mkdir(exist_ok=True, parents=True)
    Path("reports/als_candidate_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("[OK] wrote reports/als_candidate_metrics.json")


if __name__ == "__main__":
    main()
