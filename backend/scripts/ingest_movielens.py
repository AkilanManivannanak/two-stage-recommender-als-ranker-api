"""
Script: ingest_movielens.py
===========================
Downloads / generates the MovieLens dataset and writes to data/processed/.
In production this would pull from S3 or a Hive table.
For the demo it generates a high-quality synthetic dataset.

Netflix Standard: Every pipeline step has a data-drift monitor.
"""
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/processed"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_USERS   = int(os.environ.get("N_USERS",   "10000"))
N_ITEMS   = int(os.environ.get("N_ITEMS",   "2000"))
N_RATINGS = int(os.environ.get("N_RATINGS", "500000"))
SEED      = int(os.environ.get("SEED",      "42"))

GENRES = ["Action","Comedy","Drama","Horror","Sci-Fi","Romance","Thriller","Documentary","Animation","Crime"]
MATURITY = ["G","PG","PG-13","R","TV-MA"]

REAL_TITLES = [
    ("Stranger Things","Sci-Fi"),("Ozark","Thriller"),("Narcos","Crime"),
    ("The Crown","Drama"),("Money Heist","Crime"),("Dark","Sci-Fi"),
    ("Squid Game","Thriller"),("Wednesday","Horror"),("BoJack Horseman","Animation"),
    ("Peaky Blinders","Crime"),("Mindhunter","Crime"),("Black Mirror","Sci-Fi"),
    ("The Witcher","Action"),("Sex Education","Comedy"),("Bridgerton","Romance"),
    ("The OA","Drama"),("Lupin","Crime"),("Cobra Kai","Action"),
    ("Never Have I Ever","Comedy"),("Russian Doll","Comedy"),
    ("Inventing Anna","Drama"),("Wednesday","Horror"),("Glass Onion","Comedy"),
    ("The Irishman","Crime"),("Roma","Drama"),("Marriage Story","Drama"),
    ("Extraction","Action"),("Red Notice","Action"),("The Gray Man","Action"),
    ("Altered Carbon","Sci-Fi"),
]

print(f"[ingest] Generating synthetic MovieLens-style dataset seed={SEED}")
print(f"  N_USERS={N_USERS:,}  N_ITEMS={N_ITEMS:,}  N_RATINGS={N_RATINGS:,}")

rng = np.random.default_rng(SEED)
t0 = time.time()

# ── Items ─────────────────────────────────────────────────────────────
items_rows = []
for i in range(N_ITEMS):
    title, genre = REAL_TITLES[i] if i < len(REAL_TITLES) else (f"{GENRES[i%len(GENRES)]} Title {i}", GENRES[i%len(GENRES)])
    items_rows.append({
        "item_id": i+1, "title": title, "genres": genre,
        "year": int(rng.integers(1990,2025)),
        "avg_rating": round(float(rng.uniform(2.5,5.0)),1),
        "popularity": float(rng.exponential(100)),
        "runtime_min": int(rng.integers(75,180)),
        "maturity_rating": MATURITY[i%len(MATURITY)],
    })
items = pd.DataFrame(items_rows)
items.to_parquet(DATA_DIR/"items.parquet", index=False)
print(f"  Wrote {len(items):,} items → {DATA_DIR}/items.parquet")

# ── Users ─────────────────────────────────────────────────────────────
user_affinity = {u: list(rng.choice(GENRES, size=int(rng.integers(1,4)), replace=False))
                 for u in range(1, N_USERS+1)}
users = pd.DataFrame([{"user_id": u, "affinity_genres": "|".join(g)} for u, g in user_affinity.items()])
users.to_parquet(DATA_DIR/"users.parquet", index=False)
print(f"  Wrote {len(users):,} users → {DATA_DIR}/users.parquet")

# ── Ratings ───────────────────────────────────────────────────────────
rows = []
genre_map = items.set_index("item_id")["genres"].to_dict()
for _ in range(N_RATINGS):
    uid = int(rng.integers(1, N_USERS+1))
    if rng.random() < 0.7:
        pool = items[items["genres"].isin(user_affinity.get(uid, []))]
        if not len(pool): pool = items
        mid = int(pool.sample(1, random_state=int(rng.integers(9999))).iloc[0]["item_id"])
        base = float(rng.uniform(3.0,5.0))
    else:
        mid = int(items.sample(1, random_state=int(rng.integers(9999))).iloc[0]["item_id"])
        base = float(rng.uniform(1.0,5.0))
    r = round(float(np.clip(base+rng.normal(0,.5),.5,5.0))*2)/2
    rows.append({"user_id": uid, "item_id": mid, "rating": r,
                 "timestamp": int(time.time())-int(rng.integers(0,86400*365*3))})

ratings = pd.DataFrame(rows).drop_duplicates(["user_id","item_id"])
ratings.to_parquet(DATA_DIR/"ratings.parquet", index=False)
print(f"  Wrote {len(ratings):,} ratings → {DATA_DIR}/ratings.parquet")

# ── Reference stats for drift detection ──────────────────────────────
ref_stats = {
    "mean":    float(ratings["rating"].mean()),
    "std":     float(ratings["rating"].std()),
    "density": float(len(ratings)/ratings["user_id"].nunique()),
    "n_users": int(ratings["user_id"].nunique()),
    "n_items": int(ratings["item_id"].nunique()),
    "n_ratings": int(len(ratings)),
}
with open(DATA_DIR/"ref_stats.json","w") as f:
    json.dump(ref_stats, f, indent=2)
print(f"  Reference stats: mean={ref_stats['mean']:.3f}  std={ref_stats['std']:.3f}")
print(f"[ingest] Done in {time.time()-t0:.1f}s")
