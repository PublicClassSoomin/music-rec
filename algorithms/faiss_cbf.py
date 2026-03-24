"""
베이스용 샘플 추천기: 오디오 특성 FAISS 코사인 유사도(CBF).

팀원은 이 파일을 참고만 하고, 별도 모듈에서 BaseRecommender를 구현해도 됩니다.
"""

from __future__ import annotations

import pandas as pd

from algorithms.base import BaseRecommender
from data.database import get_user_liked_song_ids


class FaissContentRecommender(BaseRecommender):
    """MusicFaissIndex 기반 곡 유사도 추천 + 단순 메타 키워드 검색."""

    def __init__(self, faiss_index):
        super().__init__("faiss_cbf")
        self._index = faiss_index
        self._data: pd.DataFrame | None = None

    def fit(self, data: pd.DataFrame) -> None:
        self._data = data
        self.is_fitted = True

    def recommend(self, song_id: str, top_k: int = 10) -> dict[str, float]:
        self._check_fitted()
        if not getattr(self._index, "is_built", False):
            return {}
        return self._index.search_by_id(song_id, top_k)

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> dict[str, float]:
        self._check_fitted()
        if not getattr(self._index, "is_built", False):
            return {}
        likes = get_user_liked_song_ids(user_id)
        if not likes:
            return {}
        for sid in likes:
            out = self._index.search_by_id(sid, top_k)
            if out:
                return out
        return {}

    def search_by_query(self, query: str, top_k: int = 10) -> dict[str, float]:
        """데모용: 제목·가수 문자열에 질의 토큰이 포함되는 정도로 스코어."""
        self._check_fitted()
        if self._data is None or self._data.empty:
            return {}
        q = (query or "").lower().strip()
        if not q:
            return {}
        tokens = [t for t in q.split() if len(t) >= 2]
        if not tokens:
            tokens = [q]

        scores: dict[str, float] = {}
        for _, row in self._data.iterrows():
            sid = row.get("song_id")
            if not sid:
                continue
            text = f"{row.get('title') or ''} {row.get('artist') or ''}".lower()
            hit = sum(1 for t in tokens if t in text)
            if hit > 0:
                scores[str(sid)] = float(hit) / len(tokens)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(ranked)
