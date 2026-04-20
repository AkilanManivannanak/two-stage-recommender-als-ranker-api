"""
data_curation.py — Quality filtering and data curation pipeline

WHAT THIS ADDS:
  A real data curation engine that filters the TMDB catalog before training.
  Removes low-quality movies based on vote count, rating distribution,
  poster availability, and title validity.

  This is a DATA CURATION / DATA ENGINE:
  - Quality gates: vote_count, avg_rating, poster_url, title validity
  - Deduplication: removes duplicate titles (case-insensitive)
  - Genre normalization: maps raw genre strings to canonical 8-arm genres
  - Year filtering: removes pre-1900 and future dates
  - Distribution stats: reports quality metrics before/after filtering

WHY THIS MATTERS:
  Training on low-quality items hurts ALS (items with 1-2 ratings
  create noisy co-occurrence signals). Netflix's real data engine
  applies similar quality filters before feeding items to their models.
"""

import re
from typing import Optional


# ── Canonical genre mapping (matches LinUCB 8 arms) ───────────────────────────
CANONICAL_GENRES = {
    "Action": ["Action", "Adventure", "War", "Western"],
    "Comedy": ["Comedy", "Animation", "Family"],
    "Drama":  ["Drama", "Biography", "History", "Music"],
    "Horror": ["Horror", "Mystery"],
    "Sci-Fi": ["Science Fiction", "Sci-Fi", "Fantasy"],
    "Romance":["Romance"],
    "Thriller":["Thriller", "Crime", "Noir"],
    "Documentary":["Documentary", "Reality", "News"],
}

def normalize_genre(raw_genre: str) -> str:
    """Map raw TMDB genre to canonical LinUCB arm genre."""
    for canonical, variants in CANONICAL_GENRES.items():
        if raw_genre in variants or raw_genre == canonical:
            return canonical
    return "Drama"  # default fallback


# ── Quality filters ────────────────────────────────────────────────────────────

def quality_score(item: dict) -> float:
    """
    Compute a quality score for a catalog item.
    Combines vote reliability, rating quality, and metadata completeness.

    Inspired by Bayesian average rating:
      quality = (v/(v+m)) * R + (m/(v+m)) * C
      where v = vote_count, m = min_votes threshold, R = avg_rating, C = global_avg
    """
    vote_count = item.get("vote_count", 0)
    avg_rating = item.get("avg_rating", 0.0)
    has_poster = bool(item.get("poster_url", ""))
    has_desc   = bool(item.get("description", ""))
    year       = item.get("year", 0)

    # Bayesian average (min_votes=50, global_avg=3.5)
    m, C = 50, 3.5
    bayesian_rating = (vote_count / (vote_count + m)) * avg_rating + (m / (vote_count + m)) * C

    # Quality components
    rating_score    = bayesian_rating / 5.0         # normalised 0-1
    vote_score      = min(vote_count / 500.0, 1.0)  # caps at 500 votes
    poster_score    = 0.2 if has_poster else 0.0
    desc_score      = 0.1 if has_desc   else 0.0
    recency_score   = 0.1 if year >= 1970 else 0.0

    return round(
        0.4 * rating_score +
        0.3 * vote_score   +
        0.2 * poster_score +
        0.05 * desc_score  +
        0.05 * recency_score,
        4
    )


def curate_catalog(
    catalog:           dict,
    min_vote_count:    int   = 5,
    min_avg_rating:    float = 1.5,
    min_quality_score: float = 0.10,
    require_poster:    bool  = False,
    require_year:      bool  = True,
    min_year:          int   = 1900,
    max_year:          int   = 2026,
    deduplicate:       bool  = True,
    normalize_genres:  bool  = True,
    verbose:           bool  = True,
) -> dict:
    """
    Data curation pipeline for the movie catalog.

    Filters applied in order:
      1. Vote count gate     → removes items with < min_vote_count ratings
      2. Rating gate         → removes items with avg_rating < min_avg_rating
      3. Quality score gate  → removes items below min_quality_score
      4. Poster gate         → optionally require poster_url
      5. Year gate           → removes items outside [min_year, max_year]
      6. Title validity      → removes empty/malformed titles
      7. Deduplication       → removes duplicate titles (keep highest quality)
      8. Genre normalization → map to canonical 8 LinUCB arm genres

    Returns: curated catalog dict + curation_report
    """
    n_original = len(catalog)
    removed    = {
        "low_vote_count":    0,
        "low_avg_rating":    0,
        "low_quality_score": 0,
        "missing_poster":    0,
        "invalid_year":      0,
        "invalid_title":     0,
        "duplicate":         0,
    }

    curated     = {}
    seen_titles = {}   # title_lower → (item_id, quality)

    for item_id, item in catalog.items():
        # 1. Vote count gate
        if item.get("vote_count", 0) < min_vote_count:
            removed["low_vote_count"] += 1
            continue

        # 2. Rating gate
        if item.get("avg_rating", 0.0) < min_avg_rating:
            removed["low_avg_rating"] += 1
            continue

        # 3. Quality score gate
        q = quality_score(item)
        if q < min_quality_score:
            removed["low_quality_score"] += 1
            continue

        # 4. Poster gate (optional)
        if require_poster and not item.get("poster_url"):
            removed["missing_poster"] += 1
            continue

        # 5. Year gate
        year = item.get("year", 0)
        if require_year and not (min_year <= year <= max_year):
            removed["invalid_year"] += 1
            continue

        # 6. Title validity
        title = str(item.get("title", "")).strip()
        if not title or len(title) < 2 or re.match(r'^[^a-zA-Z0-9]+$', title):
            removed["invalid_title"] += 1
            continue

        # 7. Deduplication (keep highest quality version)
        title_lower = title.lower()
        if deduplicate and title_lower in seen_titles:
            existing_id, existing_q = seen_titles[title_lower]
            if q > existing_q:
                # Replace with higher quality version
                del curated[existing_id]
                seen_titles[title_lower] = (item_id, q)
            else:
                removed["duplicate"] += 1
                continue
        else:
            seen_titles[title_lower] = (item_id, q)

        # 8. Genre normalization
        enriched = dict(item)
        if normalize_genres:
            raw_genre = item.get("primary_genre", "Drama")
            enriched["primary_genre"]    = normalize_genre(raw_genre)
            enriched["raw_genre"]        = raw_genre
        enriched["quality_score"]    = q
        enriched["curated"]          = True

        curated[item_id] = enriched

    total_removed = sum(removed.values())
    retention     = round(len(curated) / n_original * 100, 1)

    report = {
        "n_original":      n_original,
        "n_curated":       len(curated),
        "n_removed":       total_removed,
        "retention_pct":   retention,
        "removal_breakdown": removed,
        "filters": {
            "min_vote_count":    min_vote_count,
            "min_avg_rating":    min_avg_rating,
            "min_quality_score": min_quality_score,
            "require_poster":    require_poster,
            "year_range":        f"{min_year}–{max_year}",
            "deduplicate":       deduplicate,
            "normalize_genres":  normalize_genres,
        },
        "quality_stats": _quality_stats(curated),
        "description": (
            f"Data curation: {n_original} → {len(curated)} items "
            f"({retention}% retained). "
            f"Removed: {removed['low_vote_count']} low-vote, "
            f"{removed['low_quality_score']} low-quality, "
            f"{removed['duplicate']} duplicates."
        ),
    }

    if verbose:
        print(f"  [DataCuration] {n_original} → {len(curated)} items ({retention}% retained)")
        print(f"  [DataCuration] Removed: {removed}")

    return curated, report


def _quality_stats(catalog: dict) -> dict:
    """Compute quality distribution statistics for the curated catalog."""
    if not catalog:
        return {}
    scores = [v.get("quality_score", 0) for v in catalog.values()]
    votes  = [v.get("vote_count", 0)   for v in catalog.values()]
    import numpy as np
    return {
        "quality_score_mean":   round(float(np.mean(scores)), 4),
        "quality_score_median": round(float(np.median(scores)), 4),
        "quality_score_p10":    round(float(np.percentile(scores, 10)), 4),
        "vote_count_mean":      round(float(np.mean(votes)), 1),
        "vote_count_median":    round(float(np.median(votes)), 1),
        "has_poster_pct":       round(
            sum(1 for v in catalog.values() if v.get("poster_url")) / len(catalog) * 100, 1
        ),
    }


class DataCurationEngine:
    """
    Production data curation engine.
    Run before ALS training to ensure clean, high-quality training data.
    """
    def __init__(self):
        self._report   = {}
        self._curated  = {}
        self._fitted   = False

    def curate(self, catalog: dict, **kwargs) -> dict:
        self._curated, self._report = curate_catalog(catalog, **kwargs)
        self._fitted = True
        return self._curated

    def report(self) -> dict:
        return self._report

    def quality_score(self, item: dict) -> float:
        return quality_score(item)


DATA_CURATION_ENGINE = DataCurationEngine()
print("  [DataCuration] Engine ready — call DATA_CURATION_ENGINE.curate(CATALOG)")
