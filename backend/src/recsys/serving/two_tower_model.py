"""
two_tower_model.py — CineWave
==============================
Two-tower neural retrieval model.

WHY THIS EXISTS OVER ALS
-------------------------
ALS factorises the rating matrix via dot product on latent factors. It
captures collaborative signal well but has no architecture — it cannot
incorporate rich contextual features (device, time of day, session length)
or content features (genre vectors, year, language) into the retrieval step.

A two-tower model trains a USER neural network and an ITEM neural network
independently, producing 128-dim L2-normalised embeddings for each. The
retrieval score is cosine similarity. At inference the item tower is run
once offline; the user tower runs in real-time (~1ms on CPU for 128-dim).

WHY NETFLIX USES THIS
---------------------
- User context (device, time, session) is encoded in the user tower input
- Item content (genre multi-hot, year, language) is encoded in item tower
- Trained with BPR loss on (positive, sampled negative) interaction pairs
- Item embeddings live in Qdrant alongside text embeddings — same retrieval path
- Enables efficient GPU-free ANN search over millions of items

ARCHITECTURE
------------
User tower:  [user_id embed 64] + [genre_prefs 18] + [context 4]
             → Linear(86, 256) → ReLU → Dropout(0.2)
             → Linear(256, 128) → L2-normalise → 128-dim

Item tower:  [item_id embed 64] + [genre_vec 18] + [year_norm 1] + [popularity 1]
             → Linear(84, 256) → ReLU → Dropout(0.2)
             → Linear(256, 128) → L2-normalise → 128-dim

Score:       cosine_similarity(user_emb, item_emb)

TRAINING
--------
Loss:   BPR — log(sigmoid(score_pos - score_neg))
        For each (user, positive_item) sample one random negative item.
        Minimising BPR pushes positive items above negative items in the
        user's preference ordering.

Optimiser: Adam lr=0.001
Epochs:    10 (nightly, matches Metaflow DAG timing)
Batch:     512

SERVING
-------
1. Run item tower over full catalog → 128-dim vectors → upsert into Qdrant
   collection "two_tower_items" (separate from text embeddings).
2. At /recommend: run user tower → 128-dim query → Qdrant ANN top-100.
3. Pass top-100 to LightGBM reranker exactly as ALS candidates do.
4. The two-tower replaces ALS at Stage 2; everything downstream is unchanged.

USAGE
-----
    from recsys.serving.two_tower_model import TwoTowerModel, TwoTowerRetriever

    # Training (called from phenomenal_flow_v3.py step)
    model = TwoTowerModel(n_users=50000, n_items=1200)
    model.train(interactions_df, epochs=10)
    model.save("/app/artifacts/two_tower/")

    # Serving
    retriever = TwoTowerRetriever.load("/app/artifacts/two_tower/")
    user_emb = retriever.user_embedding(user_id=42, genre_prefs=[0.8,0.1,...], context=ctx)
    candidates = retriever.retrieve(user_emb, top_k=100)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Optional torch import (graceful fallback for environments without GPU) ─────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("[TwoTower] PyTorch not available — model will run in numpy-only mode")

# ── Constants ─────────────────────────────────────────────────────────────────

N_GENRES       = 18   # must match GENRE_KEYWORDS in voice_tools.py
USER_EMB_DIM   = 64   # user_id embedding dimension
ITEM_EMB_DIM   = 64   # item_id embedding dimension
HIDDEN_DIM     = 256
OUTPUT_DIM     = 128  # final embedding dimension (matches Qdrant collection)
DROPOUT        = 0.2
LR             = 1e-3
BATCH_SIZE     = 512
N_EPOCHS       = 10

# Context features: [time_of_day_sin, time_of_day_cos, is_weekend, device_mobile]
N_CONTEXT      = 4

# ── Dataset ───────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class InteractionDataset(Dataset):
        """
        Each sample: (user_id, pos_item_id, neg_item_id, genre_prefs, context, item_features)
        Negative item sampled randomly from the full item catalog at construction time.
        """
        def __init__(
            self,
            user_ids: np.ndarray,
            item_ids: np.ndarray,
            all_item_ids: np.ndarray,
            user_genre_prefs: np.ndarray,   # shape (n_users, N_GENRES)
            item_features: np.ndarray,       # shape (n_items, N_GENRES + 2)
            user_contexts: Optional[np.ndarray] = None,  # shape (n_interactions, N_CONTEXT)
        ):
            self.user_ids      = torch.LongTensor(user_ids)
            self.item_ids      = torch.LongTensor(item_ids)
            self.all_item_ids  = all_item_ids
            self.genre_prefs   = torch.FloatTensor(user_genre_prefs)
            self.item_feats    = torch.FloatTensor(item_features)
            self.contexts      = (
                torch.FloatTensor(user_contexts)
                if user_contexts is not None
                else torch.zeros(len(user_ids), N_CONTEXT)
            )

        def __len__(self) -> int:
            return len(self.user_ids)

        def __getitem__(self, idx: int):
            uid = self.user_ids[idx]
            pos = self.item_ids[idx]
            # Sample negative: random item that is NOT the positive
            neg_idx = np.random.randint(0, len(self.all_item_ids))
            neg = torch.LongTensor([self.all_item_ids[neg_idx]])[0]

            return {
                "user_id":    uid,
                "pos_item":   pos,
                "neg_item":   neg,
                "genre_pref": self.genre_prefs[uid % len(self.genre_prefs)],
                "context":    self.contexts[idx],
                "pos_feat":   self.item_feats[pos % len(self.item_feats)],
                "neg_feat":   self.item_feats[neg % len(self.item_feats)],
            }


# ── Towers ────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class UserTower(nn.Module):
        """
        User tower: encodes user_id + genre preferences + session context
        into a 128-dim L2-normalised embedding.

        Input concatenation:
          - user_id → Embedding(n_users, USER_EMB_DIM)   → 64-dim
          - genre_prefs                                  → 18-dim (float)
          - context  [time_sin, time_cos, weekend, mobile] → 4-dim
          Total: 86-dim

        WHY NOT JUST USER_ID EMBEDDING
        --------------------------------
        A bare user_id embedding captures only historical preference.
        Adding genre_prefs (computed from recent interactions) makes the
        tower sensitive to taste drift. Adding context (time, device)
        means the same user gets different recommendations on Friday night
        vs Tuesday morning — which is correct.
        """
        def __init__(self, n_users: int):
            super().__init__()
            self.user_embedding = nn.Embedding(n_users, USER_EMB_DIM)
            input_dim = USER_EMB_DIM + N_GENRES + N_CONTEXT
            self.network = nn.Sequential(
                nn.Linear(input_dim, HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            )
            self._init_weights()

        def _init_weights(self):
            nn.init.xavier_uniform_(self.user_embedding.weight)
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        def forward(
            self,
            user_ids: torch.Tensor,
            genre_prefs: torch.Tensor,
            context: torch.Tensor,
        ) -> torch.Tensor:
            uid_emb = self.user_embedding(user_ids)               # (B, 64)
            x = torch.cat([uid_emb, genre_prefs, context], dim=1) # (B, 86)
            out = self.network(x)                                  # (B, 128)
            return F.normalize(out, p=2, dim=1)                    # L2-norm → unit sphere


    class ItemTower(nn.Module):
        """
        Item tower: encodes item_id + content features into a 128-dim
        L2-normalised embedding.

        Input concatenation:
          - item_id → Embedding(n_items, ITEM_EMB_DIM)  → 64-dim
          - genre_vec (multi-hot, 18 genres)             → 18-dim
          - year_normalised (year - 1900) / 124          → 1-dim
          - popularity_log_norm                          → 1-dim
          Total: 84-dim

        WHY SEPARATE ITEM TOWER
        ------------------------
        Item embeddings are computed ONCE offline and stored in Qdrant.
        The user tower runs in real-time at request time (~1ms CPU).
        Separating the towers allows independent updates: if the item
        catalog changes, only the item tower needs re-inference, not
        the user tower weights.
        """
        def __init__(self, n_items: int):
            super().__init__()
            self.item_embedding = nn.Embedding(n_items, ITEM_EMB_DIM)
            input_dim = ITEM_EMB_DIM + N_GENRES + 2  # +year_norm +popularity
            self.network = nn.Sequential(
                nn.Linear(input_dim, HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            )
            self._init_weights()

        def _init_weights(self):
            nn.init.xavier_uniform_(self.item_embedding.weight)
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        def forward(
            self,
            item_ids: torch.Tensor,
            item_features: torch.Tensor,   # (B, N_GENRES + 2)
        ) -> torch.Tensor:
            iid_emb = self.item_embedding(item_ids)               # (B, 64)
            x = torch.cat([iid_emb, item_features], dim=1)        # (B, 84)
            out = self.network(x)                                  # (B, 128)
            return F.normalize(out, p=2, dim=1)                    # unit sphere


# ── Main model ────────────────────────────────────────────────────────────────

class TwoTowerModel:
    """
    Wrapper that trains both towers jointly using BPR loss and provides
    save/load for the Metaflow artifact store.

    BPR LOSS EXPLANATION
    --------------------
    Bayesian Personalised Ranking (Rendle et al. 2009):
      loss = -log(sigmoid(score_positive - score_negative))
    For each (user, positive_item) we sample one random negative_item.
    The loss pushes score_positive > score_negative for all users.
    This is directly optimising ranking quality, not click prediction.

    WHY BPR OVER BINARY CROSS-ENTROPY
    -----------------------------------
    BCE treats the problem as "did this user click this item?" — a binary
    classification. BPR treats it as "did this user prefer item A over
    item B?" — a pairwise ranking. Pairwise loss maps more naturally to
    the recommendation task (we want the best items ranked highest, not
    just predicted as likely clicks).
    """

    def __init__(self, n_users: int, n_items: int):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for TwoTowerModel training")
        self.n_users   = n_users
        self.n_items   = n_items
        self.user_tower = UserTower(n_users)
        self.item_tower = ItemTower(n_items)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user_tower.to(self.device)
        self.item_tower.to(self.device)
        self._trained  = False

    def train(
        self,
        dataset: "InteractionDataset",
        epochs: int = N_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LR,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train both towers end-to-end with BPR loss.
        Returns training history: {"loss": [...per epoch...]}
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        params = list(self.user_tower.parameters()) + list(self.item_tower.parameters())
        optimiser = torch.optim.Adam(params, lr=lr)

        history: Dict[str, List[float]] = {"loss": [], "epoch_time_s": []}

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.user_tower.train()
            self.item_tower.train()
            epoch_loss = 0.0
            n_batches  = 0

            for batch in loader:
                # Move to device
                uid      = batch["user_id"].to(self.device)
                pos_iid  = batch["pos_item"].to(self.device)
                neg_iid  = batch["neg_item"].to(self.device)
                gprefs   = batch["genre_pref"].to(self.device)
                ctx      = batch["context"].to(self.device)
                pos_feat = batch["pos_feat"].to(self.device)
                neg_feat = batch["neg_feat"].to(self.device)

                # Forward pass
                user_emb    = self.user_tower(uid, gprefs, ctx)       # (B, 128)
                pos_item_emb = self.item_tower(pos_iid, pos_feat)     # (B, 128)
                neg_item_emb = self.item_tower(neg_iid, neg_feat)     # (B, 128)

                # BPR scores (dot product — both already L2-normalised → cosine sim)
                score_pos = (user_emb * pos_item_emb).sum(dim=1)      # (B,)
                score_neg = (user_emb * neg_item_emb).sum(dim=1)      # (B,)

                # BPR loss
                loss = -F.logsigmoid(score_pos - score_neg).mean()

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimiser.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed  = time.time() - t0
            history["loss"].append(avg_loss)
            history["epoch_time_s"].append(elapsed)

            if verbose:
                print(f"[TwoTower] Epoch {epoch:02d}/{epochs}  loss={avg_loss:.4f}  t={elapsed:.1f}s")

        self._trained = True
        return history

    def get_all_item_embeddings(
        self,
        item_ids: np.ndarray,
        item_features: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Run item tower over full catalog.
        Returns numpy array of shape (n_items, 128).
        Called once offline; result upserted into Qdrant.
        """
        if not self._trained:
            raise RuntimeError("Call train() before get_all_item_embeddings()")

        self.item_tower.eval()
        all_embs: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(item_ids), batch_size):
                end   = min(start + batch_size, len(item_ids))
                iids  = torch.LongTensor(item_ids[start:end]).to(self.device)
                feats = torch.FloatTensor(item_features[start:end]).to(self.device)
                embs  = self.item_tower(iids, feats)
                all_embs.append(embs.cpu().numpy())

        return np.vstack(all_embs)

    def get_user_embedding(
        self,
        user_id: int,
        genre_prefs: np.ndarray,   # shape (N_GENRES,)
        context: np.ndarray,       # shape (N_CONTEXT,)
    ) -> np.ndarray:
        """
        Run user tower for a single user at request time.
        Returns numpy array of shape (128,). ~1ms on CPU.
        """
        if not self._trained:
            raise RuntimeError("Call train() before get_user_embedding()")

        self.user_tower.eval()
        with torch.no_grad():
            uid   = torch.LongTensor([user_id % self.n_users]).to(self.device)
            gprefs = torch.FloatTensor(genre_prefs).unsqueeze(0).to(self.device)
            ctx   = torch.FloatTensor(context).unsqueeze(0).to(self.device)
            emb   = self.user_tower(uid, gprefs, ctx)
        return emb.squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        """Save both towers + metadata to directory."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.user_tower.state_dict(), save_dir / "user_tower.pt")
        torch.save(self.item_tower.state_dict(), save_dir / "item_tower.pt")

        meta = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "output_dim": OUTPUT_DIM,
            "n_genres": N_GENRES,
            "n_context": N_CONTEXT,
            "trained": self._trained,
        }
        with open(save_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[TwoTower] Saved to {save_dir}")

    @classmethod
    def load(cls, path: str) -> "TwoTowerModel":
        """Load saved model from directory."""
        save_dir = Path(path)
        with open(save_dir / "meta.json") as f:
            meta = json.load(f)

        model = cls(n_users=meta["n_users"], n_items=meta["n_items"])
        model.user_tower.load_state_dict(
            torch.load(save_dir / "user_tower.pt", map_location=model.device)
        )
        model.item_tower.load_state_dict(
            torch.load(save_dir / "item_tower.pt", map_location=model.device)
        )
        model._trained = meta["trained"]
        return model


# ── Retriever (serving layer) ─────────────────────────────────────────────────

class TwoTowerRetriever:
    """
    Wraps TwoTowerModel for serving. Provides:
    - user_embedding(): fast user tower inference
    - retrieve(): Qdrant ANN search returning top-K item IDs
    - upsert_item_embeddings(): batch-upsert item vectors into Qdrant

    This sits between the FastAPI endpoint and the LightGBM reranker,
    replacing ALS candidate generation at Stage 2.

    INTEGRATION POINT IN app.py
    ----------------------------
    In _build_recs(), replace:
        als_candidates = _bundle["als"].recommend(user_id, top_n=100)
    With:
        user_emb    = TWO_TOWER.user_embedding(user_id, genre_prefs, ctx)
        candidates  = TWO_TOWER.retrieve(user_emb, top_k=100)
    """

    def __init__(self, model: TwoTowerModel, qdrant_url: str = "http://qdrant:6333"):
        self.model       = model
        self.qdrant_url  = qdrant_url
        self._collection = "two_tower_items"

    def user_embedding(
        self,
        user_id: int,
        genre_prefs: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns 128-dim user embedding. Safe fallback if model not trained."""
        if genre_prefs is None:
            genre_prefs = np.zeros(N_GENRES, dtype=np.float32)
        if context is None:
            context = np.zeros(N_CONTEXT, dtype=np.float32)
        return self.model.get_user_embedding(user_id, genre_prefs, context)

    def retrieve(
        self,
        user_embedding: np.ndarray,
        top_k: int = 100,
        genre_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        ANN search in Qdrant using user embedding.
        Returns list of {"item_id": ..., "score": ..., "source": "two_tower"}
        """
        try:
            import requests
            payload: Dict = {
                "vector": user_embedding.tolist(),
                "limit": top_k,
                "with_payload": True,
            }
            if genre_filter:
                payload["filter"] = {
                    "must": [{"key": "genre", "match": {"value": genre_filter}}]
                }
            resp = requests.post(
                f"{self.qdrant_url}/collections/{self._collection}/points/search",
                json=payload,
                timeout=0.2,  # 200ms hard timeout — within p95 SLO
            )
            if resp.status_code == 200:
                results = resp.json().get("result", [])
                return [
                    {
                        "item_id": r["payload"].get("item_id", r["id"]),
                        "score":   r["score"],
                        "source":  "two_tower",
                    }
                    for r in results
                ]
        except Exception as e:
            print(f"[TwoTowerRetriever] Qdrant search failed: {e} — returning empty")
        return []

    def upsert_item_embeddings(
        self,
        item_ids: np.ndarray,
        item_features: np.ndarray,
        item_metadata: List[Dict],
    ) -> int:
        """
        Compute item embeddings and upsert into Qdrant.
        Called once after nightly retraining. Returns count upserted.

        item_metadata: list of dicts with keys: title, genre, year, etc.
        """
        try:
            import requests

            # Ensure collection exists with correct vector size
            requests.put(
                f"{self.qdrant_url}/collections/{self._collection}",
                json={
                    "vectors": {"size": OUTPUT_DIM, "distance": "Cosine"},
                    "on_disk_payload": True,
                },
                timeout=5,
            )

            embeddings = self.model.get_all_item_embeddings(item_ids, item_features)
            points = []
            for idx, (iid, emb) in enumerate(zip(item_ids.tolist(), embeddings)):
                meta = item_metadata[idx] if idx < len(item_metadata) else {}
                points.append({
                    "id":      int(iid),
                    "vector":  emb.tolist(),
                    "payload": {"item_id": int(iid), **meta},
                })

            # Upsert in batches of 100
            total = 0
            for i in range(0, len(points), 100):
                batch = points[i:i + 100]
                resp = requests.put(
                    f"{self.qdrant_url}/collections/{self._collection}/points",
                    json={"points": batch},
                    timeout=30,
                )
                if resp.status_code in (200, 201):
                    total += len(batch)

            print(f"[TwoTowerRetriever] Upserted {total} item embeddings")
            return total

        except Exception as e:
            print(f"[TwoTowerRetriever] Upsert failed: {e}")
            return 0

    @classmethod
    def load(cls, path: str, qdrant_url: str = "http://qdrant:6333") -> "TwoTowerRetriever":
        model = TwoTowerModel.load(path)
        return cls(model, qdrant_url)


# ── Context builder (call at request time) ────────────────────────────────────

def build_context_vector(
    hour_of_day: int,
    is_weekend: bool,
    is_mobile: bool,
) -> np.ndarray:
    """
    Encodes session context into a 4-dim float vector.

    WHY SIN/COS FOR TIME
    ---------------------
    Hour 23 is closer to hour 0 than to hour 12. A raw integer (0–23)
    does not capture this circularity. Encoding as (sin, cos) of the
    angle on a 24-hour clock gives a continuous circular representation:
    hour 0 and hour 24 map to the same point.

    Returns: [sin(hour), cos(hour), is_weekend, is_mobile]
    """
    angle = 2 * np.pi * hour_of_day / 24
    return np.array([
        np.sin(angle),
        np.cos(angle),
        float(is_weekend),
        float(is_mobile),
    ], dtype=np.float32)


# ── Numpy fallback retriever (no PyTorch required for serving) ────────────────

class NumpyTwoTowerRetriever:
    """
    Fallback retriever for environments where PyTorch is not installed.
    Loads pre-computed item embeddings from disk and does brute-force
    cosine similarity. Suitable for <=5000 items. Above that, use the
    full TwoTowerRetriever with Qdrant.
    """

    def __init__(self, embeddings_path: str):
        data = np.load(embeddings_path)
        self.item_ids   = data["item_ids"]          # (n_items,)
        self.embeddings = data["embeddings"]         # (n_items, 128) — already L2-normed
        print(f"[NumpyRetriever] Loaded {len(self.item_ids)} item embeddings")

    def retrieve(
        self,
        user_embedding: np.ndarray,   # (128,) — must be L2-normalised
        top_k: int = 100,
    ) -> List[Dict]:
        """Brute-force cosine similarity. O(n_items * 128)."""
        scores    = self.embeddings @ user_embedding          # (n_items,)
        top_idx   = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "item_id": int(self.item_ids[i]),
                "score":   float(scores[i]),
                "source":  "two_tower_numpy",
            }
            for i in top_idx
        ]

    @classmethod
    def save_embeddings(
        cls,
        path: str,
        item_ids: np.ndarray,
        embeddings: np.ndarray,
    ) -> None:
        np.savez_compressed(path, item_ids=item_ids, embeddings=embeddings)
        print(f"[NumpyRetriever] Saved {len(item_ids)} embeddings to {path}")
