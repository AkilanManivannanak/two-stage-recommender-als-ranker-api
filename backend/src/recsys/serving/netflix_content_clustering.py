"""
netflix_content_clustering.py  —  CineWave
============================================
Converted from Netflix_Movies_And_TV_Shows_Clustering__Unsupervised_ML_.ipynb

WHAT THIS IS
------------
The clustering notebook built an unsupervised content understanding system:
  - Loaded the Netflix catalog (8800+ titles, 12 columns)
  - Combined description + rating + country + genres + cast into a 'tags' column
  - Preprocessed: lowercase → remove punctuation → remove stopwords → stem
  - Vectorised with TF-IDF (9000 features)
  - Reduced with PCA (2500 components, explains 80% variance)
  - Clustered with KMeans (k=6) and Agglomerative (k=6)
  - Built content-based recommender using cluster membership

HOW IT ADDS VALUE TO CINEWAVE
-------------------------------
1. COLD-START:  New users get recommendations from their preferred cluster
                instead of pure popularity — much more relevant.

2. CONTENT-BASED RETRIEVAL: When a user says "something like Dark" via voice,
   we find Dark's cluster and retrieve top-similar items within that cluster.
   This is independent of collaborative filtering (adds a 4th retrieval signal).

3. CATALOG UNDERSTANDING: Each cluster captures a semantic theme:
   Cluster 0: International crime/thriller
   Cluster 1: Family/animation/kids content
   Cluster 2: US drama/romance
   Cluster 3: Documentary/reality
   Cluster 4: Action/adventure/sci-fi
   Cluster 5: Comedy/stand-up

4. DIVERSITY: The Slate Optimizer can enforce cross-cluster diversity —
   ensuring the page shows items from at least 3 different content clusters.

INTEGRATION POINT IN APP.PY
-----------------------------
    from recsys.serving.netflix_content_clustering import ContentClusterer

    _CLUSTERER = ContentClusterer.load("/app/artifacts/clustering/")

    # In /recommend, add cluster as a retrieval source:
    cluster_recs = _CLUSTERER.cluster_recommend(
        user_id=uid,
        catalog=CATALOG,
        user_genres=ug,
        top_k=20,
    )

    # In /similar/{item_id}, add cluster-based similarity:
    cluster_sims = _CLUSTERER.similar_items(item_id, top_k=10)

WHERE TO PLACE THIS FILE
--------------------------
    backend/src/recsys/serving/netflix_content_clustering.py

HOW TO TRAIN
-------------
    from recsys.serving.netflix_content_clustering import train_content_clusterer
    clusterer = train_content_clusterer(
        catalog=CATALOG,
        output_path="/app/artifacts/clustering/",
        n_clusters=6,
    )
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import string
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("cinewave.clustering")

# ── Optional imports ───────────────────────────────────────────────────────────

try:
    import pandas as pd
    _PD = True
except ImportError:
    _PD = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    # Download required NLTK data silently
    for pkg in ["stopwords", "punkt", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}" if pkg == "stopwords" else f"taggers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    _NLTK = True
except ImportError:
    _NLTK = False
    logger.warning("[Clustering] NLTK not installed — install: pip install nltk")


# ── Text preprocessing (from notebook) ────────────────────────────────────────

def _build_tags(item: Dict) -> str:
    """
    FROM NOTEBOOK:
        data['tags'] = (data['description'] + ' ' + data['rating'] + ' ' +
                        data['country'] + ' ' + data['listed_in'] + ' ' + data['cast'])

    Combines all text signals from a catalog item into a single tag string.
    """
    parts = [
        str(item.get("description", "") or ""),
        str(item.get("maturity_rating", "") or item.get("rating", "") or ""),
        str(item.get("country", "") or ""),
        str(item.get("genres", "") or item.get("primary_genre", "") or ""),
        str(item.get("cast", "") or item.get("director", "") or ""),
        str(item.get("title", "") or ""),
    ]
    return " ".join(p for p in parts if p.strip())


def preprocess_text(text: str, stemmer=None, stop_words=None) -> str:
    """
    FROM NOTEBOOK — full pipeline:
      1. Lowercase
      2. Remove punctuation
      3. Remove words/digits containing digits
      4. Remove stopwords
      5. Stem (SnowballStemmer)

    This is exactly what the notebook did to build the 'tags' column.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    # 3. Remove words containing digits
    text = re.sub(r"\w*\d\w*", "", text)

    # 4. Remove stopwords
    if stop_words:
        words = text.split()
        text  = " ".join(w for w in words if w not in stop_words)

    # 5. Stem
    if stemmer:
        words = text.split()
        text  = " ".join(stemmer.stem(w) for w in words)

    return text.strip()


def build_tag_corpus(
    catalog: Dict[int, Dict],
    stemmer=None,
    stop_words=None,
) -> Tuple[List[int], List[str]]:
    """
    Build (item_ids, preprocessed_tags) lists from catalog.
    Returns lists aligned to each other.
    """
    item_ids = []
    tags     = []
    for item_id, item in catalog.items():
        raw_tag = _build_tags(item)
        clean   = preprocess_text(raw_tag, stemmer=stemmer, stop_words=stop_words)
        if clean.strip():
            item_ids.append(item_id)
            tags.append(clean)
    return item_ids, tags


# ── TF-IDF vectorisation ───────────────────────────────────────────────────────

def vectorize_tags(
    tags: List[str],
    max_features: int = 9000,
) -> Tuple["np.ndarray", "TfidfVectorizer"]:
    """
    FROM NOTEBOOK:
        tfidf = TfidfVectorizer(stop_words='english', lowercase=False, max_features=9000)
        tfidf.fit(data['tags'])
        vector = tfidf.transform(data['tags']).toarray()

    Returns (dense_vector_matrix, fitted_tfidf).
    """
    if not _SKLEARN:
        raise RuntimeError("scikit-learn required")

    logger.info(f"[TF-IDF] Vectorising {len(tags)} documents, max_features={max_features}")
    tfidf = TfidfVectorizer(
        stop_words="english",
        lowercase=False,
        max_features=max_features,
        ngram_range=(1, 2),    # bigrams add phrase-level signal (e.g. "action thriller")
        sublinear_tf=True,     # log(1+tf) — dampens high-frequency terms
    )
    tfidf.fit(tags)
    vector = tfidf.transform(tags).toarray()
    logger.info(f"[TF-IDF] Vector shape: {vector.shape}")
    return vector, tfidf


# ── PCA reduction ──────────────────────────────────────────────────────────────

def reduce_dimensions(
    vector: "np.ndarray",
    n_components: int = 300,
    random_state: int = 32,
    explained_variance_target: float = 0.80,
) -> Tuple["np.ndarray", "PCA"]:
    """
    FROM NOTEBOOK:
        pca = PCA(n_components=2500, random_state=32)
        pca.fit(vector)
        X = pca.transform(vector)

    Notebook used 2500 components to capture 80% variance on 8800 items × 9000 features.
    We use 300 by default (sufficient for CineWave's ~5000 item catalog).
    Auto-adjusts n_components to catalogue size.
    """
    if not _SKLEARN:
        raise RuntimeError("scikit-learn required")

    # Can't have more components than min(n_samples, n_features)
    max_components = min(vector.shape[0] - 1, vector.shape[1], n_components)
    if max_components < n_components:
        logger.info(f"[PCA] Capping n_components: {n_components} → {max_components}")
        n_components = max_components

    logger.info(f"[PCA] Reducing {vector.shape} → {n_components} components")
    t0  = time.time()
    pca = PCA(n_components=n_components, random_state=random_state)
    X   = pca.fit_transform(vector)

    explained = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    logger.info(f"[PCA] Done in {time.time()-t0:.1f}s — {n_components} components explain {explained:.1%} variance")
    return X, pca


# ── KMeans clustering ──────────────────────────────────────────────────────────

def cluster_kmeans(
    X: "np.ndarray",
    n_clusters: int = 6,
    random_state: int = 42,
) -> Tuple["np.ndarray", "KMeans", float]:
    """
    FROM NOTEBOOK:
        kmean = KMeans(n_clusters=6)
        kmean.fit(X)
        y_kmean = kmean.predict(X)

    Notebook chose k=6 using the elbow method (KElbowVisualizer on distortion).
    We default to k=6 and also run silhouette scoring to validate.

    Returns (cluster_labels, fitted_model, silhouette_score).
    """
    if not _SKLEARN:
        raise RuntimeError("scikit-learn required")

    logger.info(f"[KMeans] Clustering {X.shape[0]} items into {n_clusters} clusters")
    t0    = time.time()
    model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    sil   = float(silhouette_score(X, labels, metric="euclidean", sample_size=min(5000, len(labels))))
    logger.info(f"[KMeans] Done in {time.time()-t0:.1f}s — silhouette={sil:.4f}")

    # Log cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        logger.info(f"  Cluster {cluster}: {count} items ({count/len(labels):.1%})")

    return labels, model, sil


def find_optimal_k(
    X: "np.ndarray",
    k_range: range = range(2, 12),
    random_state: int = 42,
) -> int:
    """
    FROM NOTEBOOK:
        for n_clusters in range(2,15):
            km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=51)
            score = silhouette_score(X, preds, metric='euclidean')
            print("For n_clusters = %d, silhouette score is %0.4f" % (n_clusters, score))

    Returns the k with highest silhouette score.
    """
    if not _SKLEARN:
        raise RuntimeError("scikit-learn required")

    best_k, best_score = 6, -1.0
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", random_state=random_state, n_init=5)
        labels = model.fit_predict(X)
        score  = float(silhouette_score(X, labels, metric="euclidean", sample_size=min(3000, len(labels))))
        logger.info(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k     = k

    logger.info(f"[KMeans] Optimal k={best_k} (silhouette={best_score:.4f})")
    return best_k


# ── Content-based similarity ───────────────────────────────────────────────────

def compute_cluster_similarity(
    X: "np.ndarray",
    item_ids: List[int],
    labels: "np.ndarray",
    top_k: int = 50,
) -> Dict[int, List[int]]:
    """
    For each item, find its top_k most similar items within the SAME cluster.
    This is the content-based recommendation step from the notebook.

    The notebook used KMeans cluster membership to group items, then
    ranked within each cluster by cosine similarity.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info(f"[ClusterSim] Computing within-cluster similarity for {len(item_ids)} items")
    t0 = time.time()

    similar_items: Dict[int, List[int]] = {}

    unique_labels = np.unique(labels)
    for cluster in unique_labels:
        cluster_mask    = labels == cluster
        cluster_ids     = [item_ids[i] for i in range(len(item_ids)) if cluster_mask[i]]
        cluster_vectors = X[cluster_mask]

        if len(cluster_ids) < 2:
            for iid in cluster_ids:
                similar_items[iid] = []
            continue

        # Normalise for cosine similarity
        normed = normalize(cluster_vectors, norm="l2")
        sims   = normed @ normed.T   # (n_cluster, n_cluster)

        for local_idx, item_id in enumerate(cluster_ids):
            row     = sims[local_idx]
            top_idx = np.argsort(row)[::-1][1:top_k + 1]  # exclude self
            similar_items[item_id] = [cluster_ids[i] for i in top_idx]

    logger.info(f"[ClusterSim] Done in {time.time()-t0:.1f}s")
    return similar_items


# ── Main clusterer class ───────────────────────────────────────────────────────

class ContentClusterer:
    """
    Serves content-based recommendations using the clustering approach
    from Netflix_Movies_And_TV_Shows_Clustering__Unsupervised_ML_.ipynb.

    Three serving methods:
    1. cluster_id(item_id) → which of the 6 clusters this item belongs to
    2. similar_items(item_id, top_k) → top-k content-similar items (within cluster)
    3. cluster_recommend(catalog, user_genres, top_k) → diverse cross-cluster recs
    """

    # Human-readable cluster themes (from notebook's wordcloud analysis)
    CLUSTER_THEMES = {
        0: "International crime & thriller",
        1: "Family, animation & kids",
        2: "US drama & romance",
        3: "Documentary & reality",
        4: "Action, adventure & sci-fi",
        5: "Stand-up comedy & music",
    }

    def __init__(self):
        self.tfidf_vectorizer    = None
        self.pca_model           = None
        self.kmeans_model        = None
        self.item_clusters: Dict[int, int] = {}    # item_id → cluster_id
        self.similar_items_map: Dict[int, List[int]] = {}
        self.cluster_items: Dict[int, List[int]] = {}  # cluster_id → [item_ids]
        self.n_clusters          = 6
        self.silhouette_score    = 0.0
        self._loaded             = False

    def cluster_id(self, item_id: int) -> Optional[int]:
        """Return the cluster this item belongs to. None if unknown."""
        return self.item_clusters.get(item_id)

    def similar_items(self, item_id: int, top_k: int = 10) -> List[Dict]:
        """Return top_k content-similar item_ids from the same cluster."""
        similar = self.similar_items_map.get(item_id, [])[:top_k]
        return [{"item_id": iid, "source": "content_cluster"} for iid in similar]

    def cluster_recommend(
        self,
        catalog: Dict[int, Dict],
        user_genres: List[str],
        top_k: int = 20,
        ensure_cross_cluster: bool = True,
    ) -> List[Dict]:
        """
        Recommend items using cluster diversity.
        Selects top_k // n_clusters items from each cluster, prioritising
        clusters that match the user's genre preferences.

        This is the cold-start path — used when ALS/two-tower don't have
        enough signal for a user.
        """
        # Map user genres to preferred clusters
        genre_cluster_affinity: Dict[int, float] = {}
        for cid, theme in self.CLUSTER_THEMES.items():
            affinity = 0.0
            theme_lower = theme.lower()
            for genre in user_genres:
                if genre.lower() in theme_lower:
                    affinity += 1.0
            genre_cluster_affinity[cid] = affinity

        recs: List[Dict] = []
        per_cluster = max(1, top_k // self.n_clusters)

        # Sort clusters by affinity descending
        sorted_clusters = sorted(genre_cluster_affinity.keys(), key=lambda c: -genre_cluster_affinity[c])

        for cid in sorted_clusters:
            cluster_item_ids = self.cluster_items.get(cid, [])
            # Pick items by popularity within cluster
            cluster_catalog = [
                catalog[iid] for iid in cluster_item_ids if iid in catalog
            ]
            cluster_catalog.sort(key=lambda x: -x.get("popularity", 0))
            for item in cluster_catalog[:per_cluster]:
                item_copy = dict(item)
                item_copy["cluster_id"]    = cid
                item_copy["cluster_theme"] = self.CLUSTER_THEMES.get(cid, "")
                item_copy["retrieval_source"] = "content_cluster"
                item_copy["score"]         = 0.5 + 0.1 * genre_cluster_affinity.get(cid, 0)
                recs.append(item_copy)

            if len(recs) >= top_k:
                break

        return recs[:top_k]

    def predict_cluster_for_new_item(self, item: Dict) -> int:
        """
        Predict cluster for a new item not in the training set.
        Used when TMDB enriches the catalog with a new title.
        """
        if self.tfidf_vectorizer is None or self.pca_model is None or self.kmeans_model is None:
            return 0
        try:
            raw = _build_tags(item)

            # Preprocess
            stop_words = None
            stemmer    = None
            if _NLTK:
                try:
                    stop_words = set(stopwords.words("english"))
                    stemmer    = SnowballStemmer("english")
                except Exception:
                    pass
            clean = preprocess_text(raw, stemmer=stemmer, stop_words=stop_words)

            vector  = self.tfidf_vectorizer.transform([clean]).toarray()
            reduced = self.pca_model.transform(vector)
            cluster = int(self.kmeans_model.predict(reduced)[0])
            return cluster
        except Exception as e:
            logger.error(f"[Clusterer] predict_cluster_for_new_item failed: {e}")
            return 0

    def save(self, path: str) -> None:
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.tfidf_vectorizer:
            with open(save_dir / "tfidf.pkl", "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
        if self.pca_model:
            with open(save_dir / "pca.pkl", "wb") as f:
                pickle.dump(self.pca_model, f)
        if self.kmeans_model:
            with open(save_dir / "kmeans.pkl", "wb") as f:
                pickle.dump(self.kmeans_model, f)

        with open(save_dir / "similar_items.pkl", "wb") as f:
            pickle.dump(self.similar_items_map, f)

        meta = {
            "item_clusters":  {str(k): int(v) for k, v in self.item_clusters.items()},
            "cluster_items":  {str(k): v for k, v in self.cluster_items.items()},
            "n_clusters":     self.n_clusters,
            "silhouette_score": self.silhouette_score,
            "cluster_themes": {str(k): v for k, v in self.CLUSTER_THEMES.items()},
        }
        with open(save_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"[Clusterer] Saved to {save_dir}")

    @classmethod
    def load(cls, path: str) -> "ContentClusterer":
        save_dir = Path(path)
        c = cls()
        if not save_dir.exists():
            logger.warning(f"[Clusterer] Artifact path not found: {save_dir}")
            return c

        for attr, fname in [("tfidf_vectorizer", "tfidf.pkl"), ("pca_model", "pca.pkl"), ("kmeans_model", "kmeans.pkl")]:
            p = save_dir / fname
            if p.exists():
                with open(p, "rb") as f:
                    setattr(c, attr, pickle.load(f))

        sim_path = save_dir / "similar_items.pkl"
        if sim_path.exists():
            with open(sim_path, "rb") as f:
                c.similar_items_map = pickle.load(f)

        meta_path = save_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            c.item_clusters   = {int(k): int(v) for k, v in meta.get("item_clusters", {}).items()}
            c.cluster_items   = {int(k): v for k, v in meta.get("cluster_items", {}).items()}
            c.n_clusters      = meta.get("n_clusters", 6)
            c.silhouette_score = meta.get("silhouette_score", 0.0)

        c._loaded = True
        items_with_clusters = len(c.item_clusters)
        logger.info(f"[Clusterer] Loaded — {items_with_clusters} items across {c.n_clusters} clusters, sil={c.silhouette_score:.4f}")
        return c


# ── Training entry point ───────────────────────────────────────────────────────

def train_content_clusterer(
    catalog: Dict[int, Dict],
    output_path: str,
    n_clusters: int = 6,
    n_tfidf_features: int = 9000,
    n_pca_components: int = 300,
    auto_find_k: bool = False,
    random_state: int = 42,
) -> ContentClusterer:
    """
    Full training pipeline from the clustering notebook.

    Call from Metaflow step or standalone:
        from recsys.serving.netflix_content_clustering import train_content_clusterer
        clusterer = train_content_clusterer(
            catalog=CATALOG,
            output_path="/app/artifacts/clustering/",
        )

    Steps:
      1. Build tag corpus from catalog items
      2. Preprocess text (lower, punct, digits, stopwords, stem)
      3. TF-IDF vectorise (9000 features)
      4. PCA reduce (300 components)
      5. KMeans cluster (k=6 or auto-find)
      6. Compute within-cluster content similarity
      7. Save all artifacts
    """
    if not _SKLEARN:
        raise RuntimeError("scikit-learn required — pip install scikit-learn")

    # Setup NLP tools
    stop_words = None
    stemmer    = None
    if _NLTK:
        try:
            stop_words = set(stopwords.words("english"))
            stemmer    = SnowballStemmer("english")
        except Exception as e:
            logger.warning(f"[Train] NLTK resources missing: {e}")

    logger.info(f"[Train] Building tag corpus from {len(catalog)} catalog items")
    item_ids, tags = build_tag_corpus(catalog, stemmer=stemmer, stop_words=stop_words)
    logger.info(f"[Train] {len(item_ids)} items with valid tags")

    # TF-IDF
    vector, tfidf_model = vectorize_tags(tags, max_features=n_tfidf_features)

    # PCA
    X, pca_model = reduce_dimensions(
        vector, n_components=n_pca_components, random_state=random_state
    )

    # Find optimal k if requested
    if auto_find_k:
        logger.info("[Train] Finding optimal k (this takes a few minutes)...")
        n_clusters = find_optimal_k(X, k_range=range(2, 13), random_state=random_state)

    # KMeans clustering
    labels, kmeans_model, sil = cluster_kmeans(X, n_clusters=n_clusters, random_state=random_state)

    # Build item→cluster and cluster→items maps
    item_clusters: Dict[int, int]      = {}
    cluster_items: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
    for item_id, cluster_label in zip(item_ids, labels.tolist()):
        item_clusters[item_id] = cluster_label
        cluster_items[cluster_label].append(item_id)

    # Within-cluster content similarity
    similar_items_map = compute_cluster_similarity(X, item_ids, labels, top_k=50)

    # Build and save clusterer
    c = ContentClusterer()
    c.tfidf_vectorizer = tfidf_model
    c.pca_model        = pca_model
    c.kmeans_model     = kmeans_model
    c.item_clusters    = item_clusters
    c.cluster_items    = cluster_items
    c.similar_items_map = similar_items_map
    c.n_clusters       = n_clusters
    c.silhouette_score = sil

    c.save(output_path)
    logger.info(f"[Train] Complete — {n_clusters} clusters, silhouette={sil:.4f}, artifacts at {output_path}")
    return c


# ── Singleton for serving ──────────────────────────────────────────────────────

_CLUSTERING_PATH = os.environ.get("CLUSTERING_PATH", "/app/artifacts/clustering/")
CONTENT_CLUSTERER = ContentClusterer.load(_CLUSTERING_PATH)
