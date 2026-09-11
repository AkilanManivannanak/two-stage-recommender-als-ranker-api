"""
build_bundle.py — Stage 4: assemble the serving bundle.

Reads the artifacts the training stages produced and writes serve_payload.json,
the single file the FastAPI process loads at startup to learn what model it is
serving and what that model measured. If a stage has not run, this script fails
rather than emitting a payload with empty metrics — the old bundle shipped
`metrics: {}` and the serving layer quietly substituted literals for them.
"""
from __future__ import annotations
import json, os, pickle, sys, time
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts")) / "bundle"
DATA   = Path(os.environ.get("DATA_DIR", ROOT / "data" / "processed"))

REQUIRED = ["metrics.json", "feature_spec.json", "als_stats.json",
            "item_factors.pkl", "user_factors.pkl", "als_model.pkl", "ranker.pkl"]
missing = [f for f in REQUIRED if not (BUNDLE / f).exists()]
if missing:
    sys.exit(f"[bundle] FAILED — missing artifacts: {missing}\n"
             f"  run: python3 scripts/prepare_data.py && python3 scripts/train_als.py "
             f"&& python3 scripts/train_ranker.py && python3 scripts/evaluate.py")

metrics = json.loads((BUNDLE / "metrics.json").read_text())
spec    = json.loads((BUNDLE / "feature_spec.json").read_text())
als     = json.loads((BUNDLE / "als_stats.json").read_text())
item_factors = pickle.load(open(BUNDLE / "item_factors.pkl", "rb"))

h = metrics["headline"]
if not h.get("ndcg_at_10"):
    sys.exit("[bundle] FAILED — metrics.json has no ndcg_at_10; evaluate.py did not complete")

payload = {
    "generated_by":  "backend/scripts/build_bundle.py",
    "generated_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_version": f"cinewave-als{als['factors']}-lgbm-{time.strftime('%Y%m%d')}",
    "dataset":       metrics["dataset"],
    "retrieval": {
        "algorithm": "implicit ALS (Hu/Koren/Volinsky)",
        "factors": als["factors"], "regularization": als["regularization"],
        "alpha": als["alpha"], "iterations": als["iterations"],
        "n_item_factors": len(item_factors),
        "candidate_recall_at_200": als["candidate_recall@200_test"],
    },
    "ranker": {
        "algorithm": "LightGBM LGBMRanker",
        "objective": spec["objective"], "metric": spec["metric"],
        "best_iteration": spec["best_iteration"],
        "n_features": spec["n_features"],
    },
    "feature_cols":       spec["feature_cols"],
    "feature_importance": spec["gain_importance"],
    "metrics":            h,
    "baselines":          metrics["baselines"],
    "bootstrap":          metrics["bootstrap"],
    "slices":             metrics["slices_by_user_activity"],
    "movie_count":        metrics["dataset"]["n_items_catalog"],
    "caveats": [
        "Offline evaluation only. No live A/B test has been run.",
        "IPS-NDCG uses exposure propensities estimated from the same logs it "
        "corrects; it is a bias correction, not a randomised experiment.",
        "The ranker is trained on the validation period and evaluated on the "
        "test period; ALS never sees either.",
    ],
}
(BUNDLE / "serve_payload.json").write_text(json.dumps(payload, indent=2))

# ── Serving feature store ─────────────────────────────────────────────────────
# The ranker needs the same per-item and per-user statistics at request time that
# it was trained on. Exporting them here — from the train split, the only data a
# serving process is entitled to — is what makes training/serving parity
# checkable rather than aspirational.
import pandas as pd
sys.path.insert(0, str(ROOT / "scripts"))
from features import build_user_genre_stats

items_df  = pd.read_parquet(DATA / "items.parquet").set_index("item_id")
train_df  = pd.read_parquet(DATA / "train.parquet")
item_feat = pd.read_parquet(DATA / "features" / "item_features.parquet").set_index("item_id")
user_feat = pd.read_parquet(DATA / "features" / "user_features.parquet").set_index("user_id")
ugs, utop = build_user_genre_stats(train_df, items_df)

ITEM_KEYS = ["item_cnt_total", "item_pos_rate", "item_avg_rating",
             "item_recency_days", "item_cnt_30d", "propensity"]
USER_KEYS = ["user_cnt_total", "user_avg_rating", "user_pos_rate", "user_tenure_days"]

store = {
    "item_features": {int(i): {k: float(r[k]) for k in ITEM_KEYS}
                      for i, r in item_feat.iterrows()},
    "item_year":     {int(i): int(y or 0) for i, y in items_df["year"].items()},
    "user_features": {int(u): {k: float(r[k]) for k in USER_KEYS}
                      for u, r in user_feat.iterrows()},
    "user_genre_stats": {f"{u}|{g}": [round(a, 4), round(sh, 4)]
                         for (u, g), (a, sh) in ugs.items()},
    "user_top_genres":  {int(u): sorted(g) for u, g in utop.items()},
    "feature_cols":     spec["feature_cols"],
}
with open(BUNDLE / "serving_features.pkl", "wb") as f:
    pickle.dump(store, f, protocol=4)
print(f"[bundle] serving_features.pkl — {len(store['item_features']):,} items · "
      f"{len(store['user_features']):,} users · "
      f"{len(store['user_genre_stats']):,} user-genre pairs")
print(f"[bundle] wrote serve_payload.json  model_version={payload['model_version']}")
print(f"  NDCG@10={h['ndcg_at_10']}  lift_vs_als={h['ndcg_lift_pct']}%  "
      f"best_baseline={h['best_baseline']}({h['best_baseline_ndcg']})")
print(f"  item_factors={len(item_factors):,}  features={spec['n_features']}")
