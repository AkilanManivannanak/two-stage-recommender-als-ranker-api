from __future__ import annotations

import argparse
from pathlib import Path

from recsys.io.ingest import IngestConfig, ingest_movielens_25m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True, help="Path to MovieLens folder containing ratings.csv")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--min_rating", type=float, default=4.0)
    ap.add_argument("--keep_latest_per_user_item", action="store_true")
    args = ap.parse_args()

    cfg = IngestConfig(
        min_rating=args.min_rating,
        keep_latest_per_user_item=args.keep_latest_per_user_item,
    )

    # Function name says 25m but it reads ratings.csv schema that is identical for ml-latest-small.
    interactions_path, stats_path = ingest_movielens_25m(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )
    print(f"[OK] wrote: {interactions_path}")
    print(f"[OK] wrote: {stats_path}")


if __name__ == "__main__":
    main()
