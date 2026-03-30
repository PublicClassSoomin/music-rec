"""
Content-Based Filtering recommender using FAISS cosine similarity.

Differences from the base faiss_cbf.py sample:
  - recommend_for_user() averages feature vectors of ALL liked songs
    (not just the first one that returns results)
  - Excludes every song the user has previously interacted with
    (play, like, skip, unlike) from the output
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from algorithms.base import BaseRecommender
from data.database import get_user_liked_song_ids, get_user_interactions
from data.faiss_index import FEATURE_COLS


class CBFRecommender(BaseRecommender):
    """
    Audio-feature Content-Based Filtering recommender.

    Constructor arg:
        faiss_index: a built (or will-be-built) MusicFaissIndex instance.
                     Passed in from startup() so we share the same index object.
    """

    def __init__(self, faiss_index):
        super().__init__("cbf")
        self._index = faiss_index
        self._data: pd.DataFrame | None = None

    # ── BaseRecommender interface ─────────────────────────────────────────

    def fit(self, data: pd.DataFrame) -> None:
        """Store the song DataFrame for metadata lookups."""
        self._data = data
        self.is_fitted = True
        print(f"[CBF] fit() complete — {len(data)} songs loaded")

    def recommend(self, song_id: str, top_k: int = 10) -> dict[str, float]:
        """
        Return the top_k most similar songs to song_id using
        cosine similarity on the 17-dim audio feature vectors.
        """
        self._check_fitted()
        if not getattr(self._index, "is_built", False):
            return {}
        return self._index.search_by_id(song_id, top_k)

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> dict[str, float]:
        """
        1. Collect the feature vectors of all songs the user has liked.
        2. Average them into a single "taste vector".
        3. Search FAISS for nearest neighbours.
        4. Remove any song the user has already interacted with.
        """
        self._check_fitted()
        if not getattr(self._index, "is_built", False):
            return {}

        # All interactions → exclude set
        interactions_df = get_user_interactions(user_id)
        interacted_ids: set[str] = (
            set(interactions_df["song_id"].tolist())
            if not interactions_df.empty
            else set()
        )

        # Preference signal: liked songs (most accurate signal)
        liked_ids = get_user_liked_song_ids(user_id)

        # Fall back to played songs when there are no likes yet
        if not liked_ids and not interactions_df.empty:
            liked_ids = (
                interactions_df[interactions_df["action"] == "play"]["song_id"]
                .unique()
                .tolist()
            )

        if not liked_ids:
            return {}

        # Average the normalised feature vectors
        vectors = [self._index.get_vector(sid) for sid in liked_ids]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            return {}

        avg_vec = np.mean(vectors, axis=0).astype(np.float32)

        # Fetch more candidates than needed so filtering doesn't leave us short
        buffer = top_k + len(interacted_ids) + 10
        candidates = self._index.search_by_vector(avg_vec, top_k=buffer)

        # Remove already-interacted songs and return top_k
        filtered = {
            sid: score
            for sid, score in candidates.items()
            if sid not in interacted_ids
        }
        ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return dict(ranked[:top_k])
