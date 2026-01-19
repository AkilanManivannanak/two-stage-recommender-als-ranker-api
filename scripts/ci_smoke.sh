#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

mkdir -p data/raw data/processed artifacts reports

# Download MovieLens small (CI-safe)
curl -L -o data/raw/ml-latest-small.zip "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
unzip -q -o data/raw/ml-latest-small.zip -d data/raw

python scripts/ingest_movielens.py --raw_dir data/raw/ml-latest-small --out_dir data/processed --min_rating 4.0 --keep_latest_per_user_item
python scripts/time_split.py --interactions data/processed/interactions.parquet --out_dir data/processed --val_frac 0.10 --test_frac 0.10

python scripts/run_baselines.py
python scripts/make_train_val.py

# Small/fast ALS for CI
python scripts/run_als_split.py \
  --name test \
  --history_path data/processed/train_val.parquet \
  --eligible_users_path data/processed/eligible_users_test.parquet \
  --holdout_path data/processed/holdout_targets_test.parquet \
  --als_dir artifacts/als_test \
  --candidates_out data/processed/candidates_test.parquet \
  --topn 200

python scripts/build_ranker_datasets.py
python scripts/train_ranker_lgbm.py

# Faster CI: you should later add a flag to bootstrap_ci_report.py to use fewer resamples (e.g., 200).
python scripts/bootstrap_ci_report.py
python scripts/gate.py

echo "[OK] CI smoke passed"
