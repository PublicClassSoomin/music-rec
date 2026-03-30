"""
경량 하이브리드 추천기 템플릿

구성:
  - 콘텐츠 점수: 기준 곡과 후보 곡의 오디오 유사도
  - 사용자 선호 점수: 좋아요한 곡들과 후보 곡의 유사도
  - 행동 보정 점수: play / like / skip 로그 기반 보정
  - 최종 점수: 가중합

주의:
  - 이 파일은 바로 구현을 시작할 수 있도록 만든 템플릿입니다.
  - TODO 표시가 있는 부분을 채우면서 완성하면 됩니다.
"""

from __future__ import annotations

import re

import pandas as pd

from algorithms.base import BaseRecommender
from data.faiss_index import FEATURE_COLS
from data.database import get_song_interaction_stats, get_user_liked_song_ids


class HybridRecommender(BaseRecommender):
    """FAISS 기반 콘텐츠 추천에 유저 선호와 행동 보정을 합치는 경량 하이브리드 추천기."""

    def __init__(
        self,
        faiss_index,
        content_weight: float = 0.6,
        user_weight: float = 0.3,
        behavior_weight: float = 0.1,
    ):
        """
        하이브리드 추천기에 필요한 공통 상태와 가중치를 초기화하는 생성자.

        현재 프로젝트에서는 FAISS 인덱스를 이미 startup()에서 만들기 때문에,
        여기서는 그 인덱스를 주입받아 내부 상태로 저장합니다.

        Args:
            faiss_index: data/faiss_index.py의 MusicFaissIndex 인스턴스
            content_weight: 콘텐츠 유사도 점수 가중치
            user_weight: 사용자 선호 점수 가중치
            behavior_weight: 행동 보정 점수 가중치
        """
        super().__init__("hybrid")
        self._index = faiss_index
        self._data: pd.DataFrame | None = None
        self._song_ids: set[str] = set()
        self._searchable_data: pd.DataFrame | None = None
        self._feature_medians: dict[str, float] = {}
        self._feature_ranges: dict[str, float] = {}

        self.content_weight = content_weight
        self.user_weight = user_weight
        self.behavior_weight = behavior_weight

    def fit(self, data) -> None:
        """
        추천기에서 사용할 곡 데이터프레임을 저장하고 기본 상태를 준비하는 함수.

        이 템플릿에서는 별도의 무거운 학습을 하지 않고,
        서버 startup()에서 전달받은 song_df를 보관하는 정도로 시작합니다.
        이후 필요하면 song_id 집합, 장르 맵, 메타 캐시 등을 여기서 확장할 수 있습니다.

        Args:
            data: get_all_songs() 결과 DataFrame
        """
        self._data = data.copy()
        self._song_ids = set(self._data["song_id"].dropna().astype(str).tolist())
        self._prepare_search_cache()
        self.is_fitted = True

    def recommend(self, song_id: str, top_k: int = 10) -> dict[str, float]:
        """
        기준 곡 하나를 받아 하이브리드 점수로 추천 결과를 반환하는 곡 기반 추천 함수.

        기본 흐름:
          1. 기준 곡으로 후보 집합 생성
          2. 후보별 콘텐츠 점수 계산
          3. 행동 보정 점수 계산
          4. 가중합으로 최종 점수 계산
          5. 점수 내림차순으로 top_k 반환

        현재 인터페이스에는 user_id가 없으므로,
        곡 기반 recommend()에서는 콘텐츠 + 행동 보정을 중심으로 시작하는 것이 안전합니다.

        Args:
            song_id: 기준이 되는 곡 ID
            top_k: 반환할 추천 곡 수

        Returns:
            {song_id: score} 형태의 추천 결과
        """
        self._check_fitted()
        if not getattr(self._index, "is_built", False):
            return {}
        if song_id not in getattr(self._index, "id_to_idx", {}):
            return {}

        base_candidates = self._build_candidates_from_song(song_id, limit=max(top_k * 3, 30))
        if not base_candidates:
            return {}

        scores: dict[str, float] = {}
        for candidate_id in base_candidates:
            if candidate_id == song_id:
                continue

            content_score = self._content_score(song_id, candidate_id)
            user_score = 0.0
            behavior_score = self._behavior_score(candidate_id)

            final_score = self._combine_scores(
                content_score=content_score,
                user_score=user_score,
                behavior_score=behavior_score,
            )
            scores[candidate_id] = final_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(ranked)

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> dict[str, float]:
        """
        유저의 좋아요 이력을 기반으로 개인화 추천을 수행하는 함수.

        기본 흐름:
          1. 사용자의 좋아요 곡 목록 조회
          2. 좋아요 곡들로 후보 집합 생성
          3. 후보별 사용자 선호 점수 계산
          4. 행동 보정 점수 계산
          5. 이미 좋아요한 곡은 제외
          6. 최종 점수 순으로 top_k 반환

        이 함수는 경량 하이브리드에서 개인화 효과를 가장 잘 보여주는 핵심 함수입니다.

        Args:
            user_id: 추천 대상 사용자 ID
            top_k: 반환할 추천 곡 수

        Returns:
            {song_id: score} 형태의 추천 결과
        """
        self._check_fitted()
        if not getattr(self._index, "is_built", False):
            return {}

        liked_song_ids = get_user_liked_song_ids(user_id)
        if not liked_song_ids:
            return {}

        candidate_ids = self._build_candidates_from_user(user_id, limit=max(top_k * 3, 30))
        if not candidate_ids:
            return {}

        scores: dict[str, float] = {}
        liked_set = set(liked_song_ids)

        for candidate_id in candidate_ids:
            if candidate_id in liked_set:
                continue

            content_score = self._content_proxy_score(liked_song_ids, candidate_id)
            user_score = self._user_preference_score(user_id, candidate_id)
            behavior_score = self._behavior_score(candidate_id)

            final_score = self._combine_scores(
                content_score=content_score,
                user_score=user_score,
                behavior_score=behavior_score,
            )
            scores[candidate_id] = final_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(ranked)

    def search_by_query(self, query: str, top_k: int = 10) -> dict[str, float]:
        """
        자연어 질의를 간단한 의도 해석으로 바꿔 메타데이터+오디오 특성 검색을 수행한다.

        Args:
            query: 사용자가 입력한 검색 문장 또는 키워드
            top_k: 반환할 검색 결과 수

        Returns:
            {song_id: score} 형태의 검색 결과
        """
        self._check_fitted()
        if self._searchable_data is None or self._searchable_data.empty:
            return {}

        q = (query or "").lower().strip()
        if not q:
            return {}

        tokens = self._tokenize(q)
        intent = self._infer_query_intent(q)
        has_audio_intent = bool(intent["feature_targets"])

        scores: dict[str, float] = {}
        for _, row in self._searchable_data.iterrows():
            sid = row["song_id"]

            text_score = self._text_match_score(row, q, tokens)
            audio_score = self._audio_intent_score(row, intent["feature_targets"])
            keyword_bonus = self._keyword_bonus_score(row, intent["keyword_weights"])

            final_score = text_score
            if has_audio_intent:
                final_score += audio_score * 0.75
            if intent["keyword_weights"]:
                final_score += keyword_bonus * 0.25

            if final_score > 0:
                scores[str(sid)] = float(final_score)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(ranked)

    def _prepare_search_cache(self) -> None:
        """자연어 검색용 메타/오디오 캐시를 준비한다."""
        if self._data is None or self._data.empty:
            self._searchable_data = None
            self._feature_medians = {}
            self._feature_ranges = {}
            return

        df = self._data.copy()
        df["song_id"] = df["song_id"].astype(str)
        df["title"] = df["title"].fillna("").astype(str)
        df["artist"] = df["artist"].fillna("").astype(str)
        df["genre"] = df["genre"].fillna("").astype(str)
        df["search_text"] = (df["title"] + " " + df["artist"] + " " + df["genre"]).str.lower()
        self._searchable_data = df

        for col in FEATURE_COLS:
            series = pd.to_numeric(df[col], errors="coerce")
            median = float(series.median()) if not series.dropna().empty else 0.0
            col_min = float(series.min()) if not series.dropna().empty else median
            col_max = float(series.max()) if not series.dropna().empty else median
            self._feature_medians[col] = median
            self._feature_ranges[col] = max(col_max - col_min, 1e-6)

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[0-9a-z가-힣]+", (text or "").lower())
        return [token for token in tokens if len(token) >= 2]

    def _infer_query_intent(self, query: str) -> dict[str, dict[str, float]]:
        """
        질의의 분위기 표현을 오디오 특성 목표값으로 변환한다.

        반환:
            {
                "feature_targets": {"bpm": ..., "energy": ...},
                "keyword_weights": {"rain": 1.0, ...},
            }
        """
        feature_votes: dict[str, list[float]] = {}
        keyword_weights: dict[str, float] = {}

        def vote(feature: str, target: str):
            value = self._feature_target_value(feature, target)
            feature_votes.setdefault(feature, []).append(value)

        def keyword(name: str, weight: float = 1.0):
            prev = keyword_weights.get(name, 0.0)
            keyword_weights[name] = max(prev, weight)

        rules = [
            (("잔잔", "차분", "조용", "편안", "힐링", "relax", "calm", "lofi", "lo-fi"), [
                ("bpm", "low"), ("energy", "low"), ("spectral_centroid", "low"), ("zcr", "low"),
            ]),
            (("신나", "흥겨", "업템포", "파티", "dance", "party"), [
                ("bpm", "high"), ("energy", "high"), ("spectral_centroid", "high"),
            ]),
            (("운동", "헬스", "러닝", "run", "workout", "gym"), [
                ("bpm", "high"), ("energy", "high"), ("zcr", "high"),
            ]),
            (("새벽", "밤", "감성", "emo", "emotional"), [
                ("bpm", "low"), ("energy", "low"), ("spectral_centroid", "low"),
            ]),
            (("드라이브", "drive"), [
                ("bpm", "mid_high"), ("energy", "mid_high"), ("spectral_centroid", "mid_high"),
            ]),
            (("청량", "시원", "밝은", "bright", "fresh"), [
                ("energy", "mid_high"), ("spectral_centroid", "high"), ("zcr", "mid_high"),
            ]),
            (("강한", "센", "강렬", "intense", "hard"), [
                ("energy", "high"), ("spectral_centroid", "high"), ("zcr", "high"),
            ]),
            (("부드러운", "포근", "따뜻", "soft", "warm"), [
                ("energy", "low"), ("spectral_centroid", "low"),
            ]),
            (("비", "rain", "rainy"), [
                ("bpm", "low"), ("energy", "low"), ("spectral_centroid", "low"),
            ]),
        ]

        for phrases, targets in rules:
            if any(phrase in query for phrase in phrases):
                for feature, target in targets:
                    vote(feature, target)

        if "비 오는 날" in query or "rainy day" in query:
            keyword("rain", 1.0)
            vote("bpm", "low")
            vote("energy", "low")
            vote("spectral_centroid", "low")

        if "사랑" in query or "love" in query:
            keyword("love", 1.0)
        if "이별" in query or "breakup" in query:
            keyword("breakup", 1.0)
            vote("energy", "low")

        feature_targets = {
            feature: sum(values) / len(values)
            for feature, values in feature_votes.items()
            if values
        }
        return {
            "feature_targets": feature_targets,
            "keyword_weights": keyword_weights,
        }

    def _feature_target_value(self, feature: str, level: str) -> float:
        median = self._feature_medians.get(feature, 0.0)
        span = self._feature_ranges.get(feature, 1.0)

        offsets = {
            "low": -0.30,
            "mid_low": -0.15,
            "mid": 0.0,
            "mid_high": 0.15,
            "high": 0.30,
        }
        return median + span * offsets.get(level, 0.0)

    def _text_match_score(self, row, raw_query: str, tokens: list[str]) -> float:
        text = row["search_text"]
        if not tokens:
            return 0.0

        token_hits = sum(1 for token in tokens if token in text)
        score = float(token_hits) / len(tokens)

        title = row["title"].lower()
        artist = row["artist"].lower()
        if raw_query and raw_query in title:
            score += 0.8
        elif raw_query and raw_query in artist:
            score += 0.6

        return score

    def _audio_intent_score(self, row, feature_targets: dict[str, float]) -> float:
        if not feature_targets:
            return 0.0

        scores = []
        for feature, target in feature_targets.items():
            value = row.get(feature)
            if pd.isna(value):
                continue
            span = self._feature_ranges.get(feature, 1.0)
            distance = abs(float(value) - target) / span
            scores.append(max(0.0, 1.0 - distance))

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _keyword_bonus_score(self, row, keyword_weights: dict[str, float]) -> float:
        if not keyword_weights:
            return 0.0

        text = row["search_text"]
        aliases = {
            "rain": ("rain", "비"),
            "love": ("love", "사랑"),
            "breakup": ("breakup", "이별", "헤어"),
        }

        total = 0.0
        matched = 0
        for key, weight in keyword_weights.items():
            phrases = aliases.get(key, (key,))
            if any(phrase in text for phrase in phrases):
                total += weight
                matched += 1

        if matched == 0:
            return 0.0
        return total / matched

    def _build_candidates_from_song(self, song_id: str, limit: int = 30) -> set[str]:
        """
        기준 곡 하나로부터 초기 후보 곡 집합을 만드는 함수.

        가장 쉬운 구현은 FAISS의 search_by_id() 결과를 그대로 후보로 쓰는 방식입니다.
        이후 필요하면 장르 필터, 인기 필터, 중복 제거 규칙을 여기에 추가하면 됩니다.

        Args:
            song_id: 기준 곡 ID
            limit: 가져올 초기 후보 수

        Returns:
            후보 곡 ID 집합
        """
        similar = self._index.search_by_id(song_id, top_k=limit)
        return set(similar.keys())

    def _build_candidates_from_user(self, user_id: int, limit: int = 30) -> set[str]:
        """
        사용자가 좋아요한 곡들을 바탕으로 후보 곡 집합을 합치는 함수.

        가장 단순한 방식은 좋아요한 각 곡에 대해 FAISS 검색을 수행하고,
        나온 결과 song_id들을 하나의 집합으로 합치는 것입니다.

        Args:
            user_id: 사용자 ID
            limit: 좋아요한 곡 하나당 가져올 후보 수

        Returns:
            후보 곡 ID 집합
        """
        liked_song_ids = get_user_liked_song_ids(user_id)
        candidate_ids: set[str] = set()

        for liked_id in liked_song_ids:
            sims = self._index.search_by_id(liked_id, top_k=limit)
            candidate_ids.update(sims.keys())

        return candidate_ids

    def _content_score(self, base_song_id: str, candidate_id: str) -> float:
        """
        기준 곡과 후보 곡 사이의 오디오 유사도 점수를 계산하는 함수.

        현재 프로젝트에서는 FAISS 검색 결과 점수를 그대로 활용하는 것이 가장 쉽습니다.
        초기 버전에서는 search_by_id() 결과 딕셔너리에서 후보 곡 점수를 꺼내는 방식으로 충분합니다.

        Args:
            base_song_id: 기준 곡 ID
            candidate_id: 점수를 계산할 후보 곡 ID

        Returns:
            0.0 이상 float 점수
        """
        sims = self._index.search_by_id(base_song_id, top_k=50)
        return float(sims.get(candidate_id, 0.0))

    def _content_proxy_score(self, liked_song_ids: list[str], candidate_id: str) -> float:
        """
        유저 기반 추천에서 사용할 콘텐츠 대체 점수를 계산하는 함수.

        기준 곡이 하나가 아닐 때는 사용자가 좋아요한 곡 목록 전체를 기준으로 보고,
        후보 곡이 이 목록과 얼마나 비슷한지 최댓값 또는 평균값으로 계산할 수 있습니다.

        초심자 구현에서는 최댓값 방식을 추천합니다.

        Args:
            liked_song_ids: 사용자가 좋아요한 곡 ID 목록
            candidate_id: 점수를 계산할 후보 곡 ID

        Returns:
            0.0 이상 float 점수
        """
        max_score = 0.0
        for liked_id in liked_song_ids:
            sims = self._index.search_by_id(liked_id, top_k=50)
            max_score = max(max_score, float(sims.get(candidate_id, 0.0)))
        return max_score

    def _user_preference_score(self, user_id: int, candidate_id: str) -> float:
        """
        사용자의 좋아요 이력을 기준으로 후보 곡의 개인화 점수를 계산하는 함수.

        구현 아이디어:
          - 좋아요한 곡 각각과의 유사도 중 최댓값 사용
          - 또는 상위 2개 평균 사용
          - 또는 좋아요 횟수, 최근성 등을 추가 반영

        이 템플릿에서는 가장 단순한 최댓값 방식을 사용합니다.

        Args:
            user_id: 사용자 ID
            candidate_id: 점수를 계산할 후보 곡 ID

        Returns:
            0.0 이상 float 점수
        """
        liked_song_ids = get_user_liked_song_ids(user_id)
        if not liked_song_ids:
            return 0.0
        return self._content_proxy_score(liked_song_ids, candidate_id)

    def _behavior_score(self, candidate_id: str) -> float:
        """
        재생/좋아요/스킵 로그를 이용해 후보 곡의 행동 보정 점수를 계산하는 함수.

        이 프로젝트에서는 data/database.py의 get_song_interaction_stats()를 이용해
        곡별 play / like / skip / unlike 집계를 가져오고, 그 값을 단순 가중합한 뒤
        0~1 범위로 정규화해 보정 점수로 사용합니다.

        기본 설계:
          - like_count가 높을수록 가산점
          - play_count가 있으면 약한 가산점
          - skip_count, unlike_count가 높을수록 감점
          - 최종적으로 0~1 범위로 clip

        Args:
            candidate_id: 점수를 계산할 후보 곡 ID

        Returns:
            0.0 이상 float 점수
        """
        stats = get_song_interaction_stats(candidate_id)
        raw_score = (
            stats["like_count"] * 2.0
            + stats["play_count"] * 0.5
            - stats["skip_count"] * 1.5
            - stats["unlike_count"] * 1.0
        )
        return max(min(raw_score / 10.0, 1.0), 0.0)

    def _combine_scores(
        self,
        content_score: float,
        user_score: float,
        behavior_score: float,
    ) -> float:
        """
        세 종류의 점수를 가중합으로 결합해 최종 추천 점수를 만드는 함수.

        초심자 버전에서는 가장 단순한 weighted sum이 가장 안전합니다.
        나중에 실험하면서 가중치를 바꾸거나, 특정 조건에서 다른 식을 써도 됩니다.

        Args:
            content_score: 기준 곡과의 콘텐츠 유사도 점수
            user_score: 유저 선호 기반 점수
            behavior_score: 행동 로그 기반 보정 점수

        Returns:
            최종 결합 점수
        """
        return (
            self.content_weight * content_score
            + self.user_weight * user_score
            + self.behavior_weight * behavior_score
        )
