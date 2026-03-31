"""
Qdrant Collection Initialiser
===============================
Creates collections for:
  - title_embeddings: 64-dim fused text+metadata vectors
  - user_taste_vectors: 64-dim ALS user factors

Called once on setup. Safe to re-run (idempotent).
Qdrant can be fully rebuilt from MinIO artifacts if storage is lost.
"""
from __future__ import annotations
import os

QDRANT_HOST = os.environ.get("QDRANT_HOST","localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT","6333"))

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
    for name, dim in [("title_embeddings",64),("user_taste_vectors",64)]:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"  [Qdrant] Created: {name} ({dim}-dim cosine)")
    print("  [Qdrant] Collections ready")
except ImportError:
    print("  [Qdrant] pip install qdrant-client to enable")
except Exception as e:
    print(f"  [Qdrant] {e}")
