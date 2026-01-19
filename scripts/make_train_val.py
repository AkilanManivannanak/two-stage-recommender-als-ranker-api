from __future__ import annotations

import pandas as pd

def main() -> None:
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    df = pd.concat([train, val], ignore_index=True)
    df = df.sort_values(["user_id", "ts"])
    df.to_parquet("data/processed/train_val.parquet", index=False)
    print("[OK] wrote data/processed/train_val.parquet", "rows=", len(df))

if __name__ == "__main__":
    main()
