"""
catalog_patch.py
================
Drop-in patch for app.py's /catalog/popular endpoint.

ROOT CAUSE OF ALL POSTER ISSUES:
  The backend /catalog/popular was returning ML-1M items (IDs 1-3883)
  which have no poster_url. The frontend then called getPosterForTitle()
  which could not match ML-1M titles (e.g. "Toy Story (1995)") to
  TMDB titles (e.g. "Toy Story") — so it returned the NUDE fallback.

FIX:
  Load movies.json (written by generate-movies-db.mjs with real TMDB
  posters) and serve those as the catalog. The TMDB item IDs (1-1200)
  are sequential integers so they work fine as item_ids.

USAGE in app.py:
  Replace the existing /catalog/popular route with:
    from recsys.serving.catalog_patch import get_tmdb_catalog
    @app.get("/catalog/popular")
    def catalog_popular(k: int = 1200):
        return {"items": get_tmdb_catalog(k)}
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

# Possible locations for movies.json
_BUNDLE_PATHS = [
    Path("artifacts/bundle/movies.json"),
    Path("/app/artifacts/bundle/movies.json"),
    Path(os.environ.get("BUNDLE_DIR", "artifacts/bundle")) / "movies.json",
]


@lru_cache(maxsize=1)
def _load_movies() -> List[Dict[str, Any]]:
    """Load TMDB movies.json — cached after first load."""
    for p in _BUNDLE_PATHS:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                # movies.json can be a list or {"items": [...]}
                movies = data if isinstance(data, list) else data.get("items", data)
                print(f"  [CatalogPatch] Loaded {len(movies)} TMDB movies from {p}")
                return movies
            except Exception as e:
                print(f"  [CatalogPatch] Failed to load {p}: {e}")
    print("  [CatalogPatch] WARNING: No movies.json found — returning empty catalog")
    return []


def _normalize(movie: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a movies.json entry to CatalogItem schema."""
    # Handle both TMDB-generated format and ML-1M format
    item_id   = int(movie.get("item_id")   or movie.get("movieId")  or 0)
    poster    = (movie.get("poster_url")   or movie.get("poster")   or "").strip()
    backdrop  = (movie.get("backdrop_url") or movie.get("backdrop") or "").strip()
    title     = (movie.get("title") or "").strip()
    genres_raw = movie.get("genres") or movie.get("primary_genre") or ""
    # genres field may be pipe-separated "Action|Adventure"
    genre_list = [g.strip() for g in str(genres_raw).split("|") if g.strip()]
    primary    = genre_list[0] if genre_list else "Drama"

    # Filter out broken TMDB placeholder images that contain "NUDE" or are empty
    if "NUDE" in poster or not poster.startswith("http"):
        poster = ""
    if "NUDE" in backdrop or not backdrop.startswith("http"):
        backdrop = ""

    return {
        "item_id":        item_id,
        "title":          title,
        "primary_genre":  primary,
        "genres":         "|".join(genre_list) or primary,
        "poster_url":     poster,
        "backdrop_url":   backdrop,
        "description":    (movie.get("description") or movie.get("overview") or "").strip(),
        "year":           int(movie.get("year") or 0),
        "avg_rating":     float(movie.get("avg_rating") or movie.get("tmdb_rating", 0) / 2 or 3.5),
        "runtime_min":    int(movie.get("runtime_min") or 100),
        "maturity_rating": str(movie.get("maturity_rating") or movie.get("rating") or "PG-13"),
        "popularity":     float(movie.get("popularity") or 0),
        "tmdb_id":        movie.get("tmdb_id"),
    }


def get_tmdb_catalog(k: int = 1200) -> List[Dict[str, Any]]:
    """
    Return up to k catalog items with real TMDB posters.
    Items with posters come first.
    """
    raw     = _load_movies()
    items   = [_normalize(m) for m in raw if m.get("title")]
    # Posters-first sort so the hero and Top 10 always have real images
    with_poster    = [i for i in items if i["poster_url"]]
    without_poster = [i for i in items if not i["poster_url"]]
    ordered = (with_poster + without_poster)[:k]
    return ordered


def reload_catalog() -> None:
    """Force reload of movies.json (call after generate-movies-db runs)."""
    _load_movies.cache_clear()
    _load_movies()
