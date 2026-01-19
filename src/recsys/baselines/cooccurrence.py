from __future__ import annotations

from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List, Tuple

import pandas as pd


def build_item_cooccurrence(
    train_df: pd.DataFrame,
    last_m_per_user: int = 20,
    max_neighbors: int = 200,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build item-item cooccurrence similarities from train interactions.
    For scalability, we only consider each user's last M items (by ts).
    Similarity = co_count (raw). This is a baseline, not a polished similarity model.
    """
    # user -> last M items
    train_df = train_df.sort_values(["user_id", "ts"])
    user_items = train_df.groupby("user_id")["item_id"].apply(lambda x: x.tail(last_m_per_user).tolist())

    pair_counts: Counter[Tuple[int, int]] = Counter()
    item_counts: Counter[int] = Counter()

    for items in user_items:
        uniq = list(dict.fromkeys(items))  # preserve order, drop dups
        for i in uniq:
            item_counts[int(i)] += 1
        n = len(uniq)
        for a_idx in range(n):
            a = int(uniq[a_idx])
            for b_idx in range(a_idx + 1, n):
                b = int(uniq[b_idx])
                if a == b:
                    continue
                pair_counts[(a, b)] += 1
                pair_counts[(b, a)] += 1

    neighbors: DefaultDict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (a, b), c in pair_counts.items():
        neighbors[a].append((b, float(c)))

    # Keep top neighbors per item
    out: Dict[int, List[Tuple[int, float]]] = {}
    for a, nbs in neighbors.items():
        nbs.sort(key=lambda t: t[1], reverse=True)
        out[int(a)] = nbs[:max_neighbors]
    return out


def recommend_cooccurrence(
    users: List[int],
    train_df: pd.DataFrame,
    item_neighbors: Dict[int, List[Tuple[int, float]]],
    pop_ranking: List[int],
    k: int = 10,
    user_profile_last_m: int = 10,
) -> Dict[int, List[int]]:
    """
    For each user: take last m items in train, gather neighbor candidates, score by summed cooccurrence count.
    Fall back to popularity for cold/empty cases.
    """
    train_df = train_df.sort_values(["user_id", "ts"])
    user_hist = train_df.groupby("user_id")["item_id"].apply(lambda x: x.tail(user_profile_last_m).tolist())
    pop_top = pop_ranking[: max(k, 200)]

    recs: Dict[int, List[int]] = {}
    for u in users:
        u = int(u)
        hist = user_hist.get(u, [])
        seen = set(int(i) for i in hist)

        scores: Counter[int] = Counter()
        for it in hist:
            it = int(it)
            for nb, s in item_neighbors.get(it, []):
                nb = int(nb)
                if nb in seen:
                    continue
                scores[nb] += float(s)

        ranked = [it for it, _ in scores.most_common(k)]
        if len(ranked) < k:
            # fill with popularity, skipping seen
            for it in pop_top:
                it = int(it)
                if it in seen or it in ranked:
                    continue
                ranked.append(it)
                if len(ranked) >= k:
                    break

        recs[u] = ranked[:k]
    return recs
