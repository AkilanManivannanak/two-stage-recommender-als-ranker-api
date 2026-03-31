"""
Script: make_train_val.py
=========================
Temporal train/val/test split.  Users are split 80/10/10 at the user level;
for each user we take their last interaction as the test item, second-to-last
as val, and all prior as train.

Netflix Standard: No data leakage — val/test are strictly future interactions.
"""
import os, time
from pathlib import Path
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/processed"))
print("[split] Splitting ratings into train / val / test ...")
t0 = time.time()

ratings = pd.read_parquet(DATA_DIR/"ratings.parquet")
ratings = ratings.sort_values(["user_id","timestamp"])

train_rows, val_rows, test_rows = [], [], []

for uid, grp in ratings.groupby("user_id"):
    grp = grp.sort_values("timestamp")
    n = len(grp)
    if n < 3:
        train_rows.append(grp)
        continue
    test_rows.append(grp.iloc[-1:])
    val_rows.append(grp.iloc[-2:-1])
    train_rows.append(grp.iloc[:-2])

train = pd.concat(train_rows, ignore_index=True)
val   = pd.concat(val_rows,   ignore_index=True)
test  = pd.concat(test_rows,  ignore_index=True)

train.to_parquet(DATA_DIR/"train.parquet", index=False)
val.to_parquet(DATA_DIR/"val.parquet",     index=False)
test.to_parquet(DATA_DIR/"test.parquet",   index=False)

print(f"  train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
print(f"[split] Done in {time.time()-t0:.1f}s")
