#!/usr/bin/env python3
"""
docker cp ~/Downloads/verify_notebooks.py recsys_api:/app/vn.py
docker exec recsys_api python3 /app/vn.py
"""
import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app")

print("=" * 55)
print("  Verifying notebook-converted files")
print("=" * 55)

# ── Netflix Prize Recommender ─────────────────────────────────────────────────
print("\n[1] netflix_prize_recommender.py")
try:
    from recsys.serving.netflix_prize_recommender import NETFLIX_PRIZE_MODEL
    loaded = getattr(NETFLIX_PRIZE_MODEL, '_loaded', False)
    score  = NETFLIX_PRIZE_MODEL.predict_rating(user_id=42, item_id=1)
    similar = NETFLIX_PRIZE_MODEL.get_similar_items(item_id=1, top_k=5)
    print(f"  loaded={loaded}")
    print(f"  predict_rating(42, 1) = {score:.3f}")
    print(f"  get_similar_items(1, top_k=5) = {similar[:3]}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# ── Content Clustering ────────────────────────────────────────────────────────
print("\n[2] netflix_content_clustering.py")
try:
    from recsys.serving.netflix_content_clustering import train_content_clusterer
    from recsys.serving.catalog_patch import get_tmdb_catalog
    import os
    from pathlib import Path

    cluster_path = "/app/artifacts/clustering/"
    Path(cluster_path).mkdir(parents=True, exist_ok=True)

    # Check if already trained
    if os.path.exists(f"{cluster_path}cluster_model.pkl") or \
       os.path.exists(f"{cluster_path}clusterer.pkl"):
        print("  Cluster model already exists — loading")
        try:
            from recsys.serving.netflix_content_clustering import ContentClusterer
            c = ContentClusterer.load(cluster_path)
            print(f"  Loaded: {len(c.cluster_items)} clusters")
            for cid, items in list(c.cluster_items.items())[:3]:
                theme = getattr(c, 'CLUSTER_THEMES', {}).get(cid, f'cluster_{cid}')
                print(f"    Cluster {cid}: {len(items)} items — {theme}")
            print("  PASS")
        except Exception as e2:
            print(f"  Load failed ({e2}), retraining...")
            raise
    else:
        print("  Training content clusterer (6 clusters)...")
        catalog_raw = get_tmdb_catalog(2000)
        catalog = {int(c["item_id"]): c for c in catalog_raw if c.get("item_id")}
        print(f"  Catalog: {len(catalog)} items")
        clusterer = train_content_clusterer(catalog, cluster_path, n_clusters=6)
        sil = getattr(clusterer, 'silhouette_score', 'N/A')
        print(f"  Trained — silhouette={sil}")
        for cid, items in list(clusterer.cluster_items.items())[:3]:
            theme = getattr(clusterer, 'CLUSTER_THEMES', {}).get(cid, f'cluster_{cid}')
            print(f"    Cluster {cid}: {len(items)} items — {theme}")
        print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 55)
print("  Done")
print("=" * 55)
