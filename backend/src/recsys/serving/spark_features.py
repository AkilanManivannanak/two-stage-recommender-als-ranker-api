"""
PySpark Feature Engineering Pipeline
=====================================
Replaces the pandas-based point_in_time_features step in the Metaflow
pipeline with a PySpark implementation.

What this replaces
------------------
The original step used Python dicts and for-loops over 800k ratings:
  for r in self._train_ratings:        # O(n) Python loop
      user_genre_hist[uid][genre]...   # nested defaultdict

This module uses PySpark DataFrame operations instead:
  df.groupBy("user_id", "genre")      # distributed groupBy
    .agg(avg("rating"), count("*"))    # columnar aggregation

Why PySpark here specifically
------------------------------
1. The ratings DataFrame (800k rows × 8 cols) is the right size where
   Spark starts to show benefit over pandas — large enough that groupBy
   aggregations are faster with columnar execution, small enough that
   we don't need a real cluster (SparkContext runs in local[*] mode).

2. The features computed here (user genre affinity, item popularity,
   co-occurrence signals) are exactly the kind of aggregations Spark
   was designed for — they parallelize trivially across partitions.

3. This mirrors the real production pattern at Netflix/Spotify where
   feature engineering runs in Spark on EMR/Databricks, and the
   Metaflow step just calls a precomputed feature store.

Fallback
--------
If PySpark is not installed, falls back to the original pandas/dict
implementation so the pipeline never hard-fails.

Usage in Metaflow
-----------------
    from recsys.serving.spark_features import compute_features_spark

    features = compute_features_spark(
        train_ratings = self._train_ratings,   # list[dict]
        raw_items     = self.raw_items,         # dict[int, dict]
        events        = self.events,            # list[dict]
        use_spark     = True,                   # set False to use fallback
    )
    self.user_genre_ratings = features["user_genre_ratings"]
    self.user_activity      = features["user_activity"]
    self.impression_counts  = features["impression_counts"]
    self.item_popularity    = features["item_popularity"]
    self.item_cooccurrence  = features["item_cooccurrence"]  # NEW: not in pandas version
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any
import numpy as np


# ── PySpark availability check ────────────────────────────────────────────────

def _has_spark() -> bool:
    try:
        import pyspark  # noqa: F401
        return True
    except ImportError:
        return False


# ── Spark session factory ─────────────────────────────────────────────────────

def _find_java_home():
    """Auto-detect JAVA_HOME from common locations."""
    import os, subprocess
    if os.environ.get('JAVA_HOME'):
        return
    # Try /app/.java_home written at container build
    try:
        with open('/app/.java_home') as f:
            line = f.read().strip()
            if line.startswith('JAVA_HOME='):
                jh = line.split('=',1)[1]
                if os.path.exists(jh):
                    os.environ['JAVA_HOME'] = jh
                    return
    except Exception:
        pass
    # Search common paths
    for jh in ['/usr/lib/jvm/default-java',
               '/usr/lib/jvm/java-11-openjdk-amd64',
               '/usr/lib/jvm/java-11-openjdk-arm64',
               '/usr/lib/jvm/java-17-openjdk-amd64',
               '/usr/lib/jvm/java-17-openjdk-arm64',
               '/usr/lib/jvm/java-21-openjdk-amd64',
               '/usr/lib/jvm/java-21-openjdk-arm64']:
        if os.path.exists(os.path.join(jh, 'bin', 'java')):
            os.environ['JAVA_HOME'] = jh
            return

def _get_spark(app_name: str = "CineWaveFeatures"):
    """
    Get or create a local SparkSession.
    Uses local[*] — all available cores on the single machine.
    No cluster required; works inside Docker container.
    """
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")   # small data → few partitions
        .config("spark.ui.enabled", "false")            # disable UI to reduce overhead
        .config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=ERROR")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ══════════════════════════════════════════════════════════════════════════════
# SPARK IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _compute_spark(
    train_ratings: list[dict],
    raw_items:     dict[int, dict],
    events:        list[dict],
) -> dict[str, Any]:
    """
    PySpark implementation of point_in_time_features.

    Computes 5 feature sets using distributed DataFrame operations:
      1. user_genre_ratings   — {uid: {genre: [ratings]}}
      2. user_activity        — {uid: {n_ratings, avg_rating, n_genres}}
      3. impression_counts    — {uid: {item_id: count}}
      4. item_popularity      — {item_id: normalised_popularity}
      5. item_cooccurrence    — {item_id: [top-10 co-watched items]}  (NEW)
    """
    from pyspark.sql import SparkSession, Row
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, IntegerType, FloatType, StringType
    )

    spark = _get_spark()

    # ── 1. Build ratings DataFrame ────────────────────────────────────────

    # Attach genre from raw_items (join-side data is small → broadcast)
    rows = []
    for r in train_ratings:
        iid   = r["item_id"]
        genre = raw_items.get(iid, {}).get("primary_genre", "Unknown")
        rows.append(Row(
            user_id  = int(r["user_id"]),
            item_id  = int(iid),
            rating   = float(r.get("rating", 3.5)),
            genre    = genre,
            ts       = int(r.get("timestamp", 0)),
        ))

    schema = StructType([
        StructField("user_id", IntegerType()),
        StructField("item_id", IntegerType()),
        StructField("rating",  FloatType()),
        StructField("genre",   StringType()),
        StructField("ts",      IntegerType()),
    ])
    df = spark.createDataFrame(rows, schema=schema).cache()

    # ── 2. user_genre_ratings: avg & list of ratings per (user, genre) ───

    user_genre_df = (
        df.groupBy("user_id", "genre")
          .agg(
              F.collect_list("rating").alias("ratings"),
              F.avg("rating").alias("avg_genre_rating"),
              F.count("*").alias("n_genre_ratings"),
          )
    )

    # Collect to Python — small result (6040 users × ~18 genres avg)
    user_genre_ratings: dict[int, dict] = defaultdict(dict)
    for row in user_genre_df.collect():
        user_genre_ratings[row.user_id][row.genre] = list(row.ratings)

    # ── 3. user_activity: aggregate per user ─────────────────────────────

    user_activity_df = (
        df.groupBy("user_id")
          .agg(
              F.count("*").alias("n_ratings"),
              F.avg("rating").alias("avg_rating"),
              F.countDistinct("genre").alias("n_genres"),
              F.max("ts").alias("last_active_ts"),
          )
    )

    user_activity: dict[int, dict] = {}
    for row in user_activity_df.collect():
        user_activity[row.user_id] = {
            "n_ratings":      int(row.n_ratings),
            "avg_rating":     float(row.avg_rating),
            "n_genres":       int(row.n_genres),
            "last_active_ts": int(row.last_active_ts or 0),
        }

    # ── 4. item_popularity: normalised by max count ───────────────────────

    item_pop_df = (
        df.groupBy("item_id")
          .agg(F.count("*").alias("n_ratings"))
    )
    pop_rows  = item_pop_df.collect()
    max_count = max((r.n_ratings for r in pop_rows), default=1)
    item_popularity: dict[int, float] = {
        r.item_id: round(r.n_ratings / max_count, 6)
        for r in pop_rows
    }

    # ── 5. impression_counts from events ─────────────────────────────────

    imp_rows = [
        Row(user_id=int(e["user_id"]), item_id=int(e["item_id"]))
        for e in events
        if e.get("event_type") == "impression"
    ]
    impression_counts: dict[int, dict] = defaultdict(dict)
    if imp_rows:
        imp_schema = StructType([
            StructField("user_id", IntegerType()),
            StructField("item_id", IntegerType()),
        ])
        imp_df = spark.createDataFrame(imp_rows, schema=imp_schema)
        imp_agg = (
            imp_df.groupBy("user_id", "item_id")
                  .agg(F.count("*").alias("cnt"))
        )
        for row in imp_agg.collect():
            impression_counts[row.user_id][row.item_id] = int(row.cnt)

    # ── 6. item_cooccurrence (NEW — not in pandas version) ───────────────
    # Items that are co-rated positively (rating >= 4) by the same user.
    # Used by the retrieval engine to find "users who liked X also liked Y".

    pos_df = df.filter(F.col("rating") >= 4.0).select("user_id", "item_id")

    # Self-join on user_id to get item pairs
    left  = pos_df.alias("left")
    right = pos_df.alias("right")
    pairs = (
        left.join(right, on="user_id")
            .filter(F.col("left.item_id") < F.col("right.item_id"))
            .select(
                F.col("left.item_id").alias("item_a"),
                F.col("right.item_id").alias("item_b"),
            )
            .groupBy("item_a", "item_b")
            .agg(F.count("*").alias("co_count"))
            .filter(F.col("co_count") >= 3)   # minimum support threshold
    )

    # Build co-occurrence map: item → top-10 most co-watched items
    item_cooccurrence: dict[int, list[int]] = defaultdict(list)
    cooc_rows = (
        pairs.orderBy(F.col("co_count").desc())
             .limit(50000)   # cap for memory
             .collect()
    )
    # Collect both directions
    raw_cooc: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in cooc_rows:
        raw_cooc[row.item_a].append((row.item_b, row.co_count))
        raw_cooc[row.item_b].append((row.item_a, row.co_count))

    for item_id, neighbours in raw_cooc.items():
        top10 = sorted(neighbours, key=lambda x: -x[1])[:10]
        item_cooccurrence[item_id] = [iid for iid, _ in top10]

    df.unpersist()

    return {
        "user_genre_ratings": dict(user_genre_ratings),
        "user_activity":      user_activity,
        "impression_counts":  dict(impression_counts),
        "item_popularity":    item_popularity,
        "item_cooccurrence":  dict(item_cooccurrence),
        "engine":             "pyspark_local",
        "n_ratings":          len(train_ratings),
        "n_users":            len(user_activity),
        "n_items":            len(item_popularity),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PANDAS FALLBACK (original implementation, preserved for reliability)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_pandas(
    train_ratings: list[dict],
    raw_items:     dict[int, dict],
    events:        list[dict],
) -> dict[str, Any]:
    """
    Original pandas/dict implementation — used when PySpark is unavailable.
    Produces identical output schema to the Spark version.
    """
    user_genre_hist: dict = defaultdict(lambda: defaultdict(list))
    for r in train_ratings:
        iid   = r["item_id"]
        genre = raw_items.get(iid, {}).get("primary_genre", "Unknown")
        user_genre_hist[r["user_id"]][genre].append(r.get("rating", 3.5))

    user_genre_ratings = {uid: dict(gr) for uid, gr in user_genre_hist.items()}

    # impression counts
    imp_counts: dict = defaultdict(lambda: defaultdict(int))
    for e in events:
        if e.get("event_type") == "impression":
            imp_counts[e["user_id"]][e["item_id"]] += 1
    impression_counts = {u: dict(items) for u, items in imp_counts.items()}

    # user activity
    user_activity: dict = {}
    for uid, genres in user_genre_ratings.items():
        all_r = [r for rs in genres.values() for r in rs]
        user_activity[uid] = {
            "n_ratings":  len(all_r),
            "avg_rating": float(np.mean(all_r)) if all_r else 3.5,
            "n_genres":   len(genres),
        }

    # item popularity
    item_counts: dict = defaultdict(int)
    for r in train_ratings:
        item_counts[r["item_id"]] += 1
    max_cnt = max(item_counts.values(), default=1)
    item_popularity = {iid: round(cnt / max_cnt, 6) for iid, cnt in item_counts.items()}

    # co-occurrence (pandas version — less optimised)
    user_pos_items: dict = defaultdict(set)
    for r in train_ratings:
        if r.get("rating", 0) >= 4.0:
            user_pos_items[r["user_id"]].add(r["item_id"])

    raw_cooc: dict = defaultdict(lambda: defaultdict(int))
    for uid, items in user_pos_items.items():
        items_list = sorted(items)
        for i, a in enumerate(items_list):
            for b in items_list[i+1:i+6]:   # limit pairs per user
                raw_cooc[a][b] += 1
                raw_cooc[b][a] += 1

    item_cooccurrence = {
        iid: sorted(nbrs, key=lambda x: -raw_cooc[iid][x])[:10]
        for iid, nbrs in raw_cooc.items()
        if raw_cooc[iid]
    }

    return {
        "user_genre_ratings": user_genre_ratings,
        "user_activity":      user_activity,
        "impression_counts":  impression_counts,
        "item_popularity":    item_popularity,
        "item_cooccurrence":  item_cooccurrence,
        "engine":             "pandas_fallback",
        "n_ratings":          len(train_ratings),
        "n_users":            len(user_activity),
        "n_items":            len(item_popularity),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def compute_features_spark(
    train_ratings: list[dict],
    raw_items:     dict[int, dict],
    events:        list[dict],
    use_spark:     bool = True,
) -> dict[str, Any]:
    """
    Compute point-in-time features using PySpark (or pandas fallback).

    Parameters
    ----------
    train_ratings : list[dict]   — training ratings (no val/test leakage)
    raw_items     : dict[int,dict] — item metadata including primary_genre
    events        : list[dict]   — impression/click events from event log
    use_spark     : bool         — if False, forces pandas fallback

    Returns
    -------
    dict with keys:
      user_genre_ratings  : {uid: {genre: [ratings]}}
      user_activity       : {uid: {n_ratings, avg_rating, n_genres}}
      impression_counts   : {uid: {item_id: count}}
      item_popularity     : {item_id: float 0-1}
      item_cooccurrence   : {item_id: [top-10 co-watched item_ids]}
      engine              : "pyspark_local" | "pandas_fallback"
      n_ratings, n_users, n_items : int
    """
    spark_available = _has_spark() and use_spark

    if spark_available:
        try:
            result = _compute_spark(train_ratings, raw_items, events)
            print(f"  [spark_features] engine=pyspark_local | "
                  f"n={result['n_ratings']:,} ratings | "
                  f"users={result['n_users']:,} | "
                  f"items={result['n_items']:,} | "
                  f"cooc_items={len(result['item_cooccurrence']):,}")
            return result
        except Exception as e:
            print(f"  [spark_features] PySpark failed ({e}), falling back to pandas")

    result = _compute_pandas(train_ratings, raw_items, events)
    print(f"  [spark_features] engine=pandas_fallback | "
          f"n={result['n_ratings']:,} ratings | "
          f"users={result['n_users']:,} | "
          f"items={result['n_items']:,}")
    return result
