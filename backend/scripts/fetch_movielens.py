"""
fetch_movielens.py — download MovieLens-1M into backend/data/raw/.

The dataset is ~24MB and is NOT committed to this repository. Every metric this
project reports is reproduced by running the pipeline over it:

    python scripts/fetch_movielens.py
    python scripts/prepare_data.py
    python scripts/train_als.py
    python scripts/train_ranker.py
    python scripts/evaluate.py
"""
from __future__ import annotations
import io, os, sys, zipfile
from pathlib import Path
from urllib.request import urlopen

RAW = Path(os.environ.get("RAW_DIR",
           Path(__file__).resolve().parents[1] / "data" / "raw"))
RAW.mkdir(parents=True, exist_ok=True)
NEEDED = ["ratings.dat", "movies.dat", "users.dat"]

if all((RAW / f).exists() for f in NEEDED):
    print(f"[fetch] already present in {RAW}")
    sys.exit(0)

SOURCES = [
    "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "https://raw.githubusercontent.com/vandit15/Movielens-Data/master/ml-1m/{name}",
]

# Primary: the official GroupLens archive.
try:
    print(f"[fetch] downloading {SOURCES[0]}")
    blob = urlopen(SOURCES[0], timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in NEEDED:
            member = next(m for m in z.namelist() if m.endswith(name))
            (RAW / name).write_bytes(z.read(member))
    print(f"[fetch] extracted {NEEDED} to {RAW}")
    sys.exit(0)
except Exception as exc:
    print(f"[fetch] primary source failed ({exc}); trying mirror")

# Fallback: a file-by-file mirror, for networks that block grouplens.org.
for name in NEEDED:
    url = SOURCES[1].format(name=name)
    print(f"[fetch] {url}")
    (RAW / name).write_bytes(urlopen(url, timeout=120).read())
print(f"[fetch] wrote {NEEDED} to {RAW}")
