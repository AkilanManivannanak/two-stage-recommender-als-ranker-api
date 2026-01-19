from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recsys.als.train_als import ALSConfig, train_als
from recsys.als.candidates import CandidateConfig, generate_candidates, candidate_recall_at_k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True)  # e.g. val or test
    ap.add_argument("--history_path", type=str, required=True)  # train.parquet or train_val.parquet
    ap.add_argument("--eligible_users_path", type=str, required=True)
    ap.add_argument("--holdout_path", type=str, required=True)
    ap.add_argument("--als_dir", type=str, required=True)
    ap.add_argument("--candidates_out", type=str, required=True)

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

    print(f"[{args.name}] [1/3] Train ALS on {args.history_path}")
    Path(args.als_dir).mkdir(parents=True, exist_ok=True)
    train_als(args.history_path, args.als_dir, als_cfg)

    eligible = pd.read_parquet(args.eligible_users_path)["user_id"].tolist()
    holdout = pd.read_parquet(args.holdout_path)

    print(f"[{args.name}] [2/3] Generate candidates topN={args.topn}")
    cand_path = generate_candidates(
        train_path=args.history_path,
        eligible_users=eligible,
        als_dir=args.als_dir,
        out_path=args.candidates_out,
        cfg=CandidateConfig(topn=args.topn, alpha=args.alpha),
    )

    print(f"[{args.name}] [3/3] Candidate recall@50/200")
    cand = pd.read_parquet(cand_path)
    r50 = candidate_recall_at_k(cand, holdout, k=50)
    r200 = candidate_recall_at_k(cand, holdout, k=200)

    metrics = {
        "name": args.name,
        "history_path": args.history_path,
        "eligible_users": len(eligible),
        "topn": args.topn,
        "candidate_recall@50": r50,
        "candidate_recall@200": r200,
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path(f"reports/als_candidate_metrics_{args.name}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"[OK] wrote reports/als_candidate_metrics_{args.name}.json")
    print(f"[OK] wrote {cand_path}")


if __name__ == "__main__":
    main()
