from __future__ import annotations

import argparse

from recsys.split.time_split import TimeSplitConfig, time_split_interactions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", type=str, default="data/processed/interactions.parquet")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_cutoff_ts", type=int, default=None)
    ap.add_argument("--val_cutoff_ts", type=int, default=None)
    args = ap.parse_args()

    cfg = TimeSplitConfig(
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        test_cutoff_ts=args.test_cutoff_ts,
        val_cutoff_ts=args.val_cutoff_ts,
    )

    paths = time_split_interactions(args.interactions, args.out_dir, cfg)
    print("[OK] wrote split artifacts:")
    for k, p in paths.items():
        print(f"  - {k}: {p}")


if __name__ == "__main__":
    main()
