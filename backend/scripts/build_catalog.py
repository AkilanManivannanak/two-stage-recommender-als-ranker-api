"""
build_catalog.py — reconcile the serving catalogue with the trained model.

The catalogue previously shipped in artifacts/bundle/movies.json used sequential
synthetic ids (movieId 1 = "The Shawshank Redemption"). MovieLens movieId 1 is
"Toy Story". Because the ALS factors are keyed by real MovieLens ids, the two
could never be joined: any "als_score" the API returned for a title was the
score of a different film.

This rebuilds movies.json from the MovieLens catalogue so ids match the factors,
and carries over TMDB poster art and descriptions by normalised-title match
where the old file had them.
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
DATA   = Path(os.environ.get("DATA_DIR", ROOT / "data" / "processed"))
BUNDLE = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts")) / "bundle"

def norm(t: str) -> str:
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", str(t)).lower()
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r",\s*(the|a|an)$", "", t)
    return re.sub(r"[^a-z0-9]", "", t)

items     = pd.read_parquet(DATA / "items.parquet")
item_feat = pd.read_parquet(DATA / "features" / "item_features.parquet").set_index("item_id")

# Existing TMDB art, keyed by normalised title.
tmdb = {}
old = BUNDLE / "movies.json"
if old.exists():
    for m in json.loads(old.read_text()):
        if isinstance(m, dict) and m.get("title"):
            tmdb[norm(m["title"])] = m

rows, matched = [], 0
for r in items.itertuples(index=False):
    art = tmdb.get(norm(r.title_clean), {})
    if art: matched += 1
    f = item_feat.loc[r.item_id] if r.item_id in item_feat.index else None
    rows.append({
        "item_id":       int(r.item_id),
        "movieId":       int(r.item_id),
        "title":         r.title_clean,
        "title_full":    r.title,
        "genres":        r.genres,
        "primary_genre": r.primary_genre,
        "year":          int(r.year) if r.year else None,
        "description":   art.get("description", ""),
        "poster_url":    art.get("poster_url", ""),
        "backdrop_url":  art.get("backdrop_url", ""),
        # Train-split statistics — what the serving layer knows at request time.
        "vote_count":    int(f["item_cnt_total"]) if f is not None else 0,
        "avg_rating":    round(float(f["item_avg_rating"]), 3) if f is not None else 0.0,
        "popularity":    int(f["item_cnt_total"]) if f is not None else 0,
        "in_training":   bool(f is not None),
    })

(BUNDLE / "movies.json").write_text(json.dumps(rows, indent=1))
n_art = sum(1 for r in rows if r["poster_url"])
print(f"[catalog] {len(rows):,} titles written · ids now match ALS factors")
print(f"  TMDB art carried over for {matched:,} ({matched/len(rows):.1%}) · "
      f"{n_art:,} with a poster url")
print(f"  {sum(1 for r in rows if r['in_training']):,} appear in the train split")
