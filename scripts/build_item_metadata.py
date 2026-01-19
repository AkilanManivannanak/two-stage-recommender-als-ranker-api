from __future__ import annotations

from pathlib import Path
import pandas as pd

def main() -> None:
    candidates = [
        Path("data/raw/ml-25m/movies.csv"),
        Path("data/raw/ml-latest-small/movies.csv"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise SystemExit("movies.csv not found. Download MovieLens dataset first.")

    df = pd.read_csv(src)
    out = df.rename(columns={"movieId": "item_id"})[["item_id", "title", "genres"]]
    Path("artifacts/bundle/features").mkdir(parents=True, exist_ok=True)
    out.to_parquet("artifacts/bundle/features/item_metadata.parquet", index=False)
    print("[OK] wrote artifacts/bundle/features/item_metadata.parquet rows=", len(out))

if __name__ == "__main__":
    main()
