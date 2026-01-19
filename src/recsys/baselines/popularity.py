from __future__ import annotations

from typing import Dict, List

import pandas as pd


def fit_popularity(train_df: pd.DataFrame) -> List[int]:
    # Most popular items in train by interaction count
    item_counts = train_df.groupby("item_id")["user_id"].count().sort_values(ascending=False)
    return item_counts.index.astype(int).tolist()


def recommend_popularity(users: List[int], pop_ranking: List[int], k: int) -> Dict[int, List[int]]:
    topk = pop_ranking[:k]
    return {int(u): list(topk) for u in users}
