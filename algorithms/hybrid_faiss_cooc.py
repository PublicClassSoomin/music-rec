"""
HybridFaissCoocRecommender
    FAISS 기반 콘텐츠 유사도(오디오 특성)와,
    사용자 interactions 에서 만든 공동출현(co-occurrence)을 합친 하이브리드 추천기

FAISS (Facebook AI Similarity Search) 란?
Meta 의 기본 AI 연구팀에서 개발한 
'유사도 검색(Similarity Search)' 과 '밀집 벡터(Dense Vector) 클러스트링'을 위한 오픈소스 라이브러리 

역할 분리
    - FAISS: 오디오 특성 기반 유사도 검색 (MFCC, BPM 등 - 파이프라인에서 추출)
    - cooc: "같은 사용자 맥락에서 함께 등장한 곡"

왜 하이브리드인가?
    - 오디오는 앞 30초만 분석하므로 인트로 편향・구조 미반영 등 한계가 있다.
    - 좋아요 공동출현은 "사람들이 실제로 함께 선호한 패턴"이라 오디오 신호와 다른 축이다.
    - 두 점수를 섞으면 한쪽이 약할 때 다른 쪽이 보완하는 효과를 기대할 수 있다.

API 연동 (api/main.py)
    recommenders["hybrid_faiss_cooc"] = HybridFaissCoocRecommender(...)
    - POST /api/recommend          → recommend 또는 recommend_with_options
    - POST /api/recommend/user     → recommend_for_user_* 
    - POST /api/search             → search_by_query_* (옵션에 따라 임베딩/LLM 시도)

추천 options (body.options)
    - mode: "simple" | "advanced" — 어떤 공동출현 그래프를 쓸지
    - weights: {"content": float, "cooc": float} — 요청 한 번에만 적용 후 복구
    - threshold: float — 하이브리드 합산 점수 하한 (이하면 제외)
    - filters: {"genre": str} 등 — 구현된 키만 필터링 (선택)

검색 options — 상세 의미·기본값은 text_search_embed_llm.py 주석 참고.
    - use_full_embedding_search, use_embedding_rerank, use_llm_search,
      search_candidate_multiplier, min_token_score, llm_max_keywords
주의
    - cooc는 유저당 양수 선호 곡 쌍을 모두 보므로 좋아요가 많으면 O(n^2) 증가 가능.
    - Advanced 모드는 play_seconds·timestamp 품질에 민감할 수 있다.
      
- HybridFaissCoocSimple: FAISS + interactions 에서 action=='like' 만 모아 공동출현
- HybridFaissCoocAdvanced: 
    - (1) 현재 좋아요(unlike 반영) 공동출현
    - (2) like/play/skip/unlike 가중
    - (3) 시간 감쇠로 임시적 점수, 양수 곡끼리 공동출현 가중
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from rapidfuzz import fuzz

from algorithms.base import BaseRecommender
from algorithms.text_search_embed_llm import TextEmbeddingSearcher, EmbedSearchConfig
from data.database import (
    get_all_interactions,
    get_distinct_interaction_user_ids,
    get_user_liked_song_ids,
)

# ------------------------------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------------------------------

def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """
    점수 딕셔너리를 [0, 1] 범위로 min-max 정규화

    왜 필요한가?
    - FAISS 코사인 점수와 co-occurrence 카운트/가중치는 단위가 다르다.
    - FAISS는 0~1 범위, co-occurrence는 0~1000 범위
    - 그대로 더하면 특정 축이 과도하게 지배할 수 있어 정규화가 필요하다.

    특이 케이스:
    - 모든 점수가 동일하면 (max-min=0), 전부 1.0으로 설정해 후보를 살린다.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

def _parse_ts(raw: Any) -> datetime | None:
    """
    interactions.timestamp 값을 datetime으로 안전하게 파싱한다.

    지원 포맷:
    - ISO 문자열(예: 2026-03-25T10:20:30.123456)
    - 끝에 'Z'가 붙은 UTC 포맷도 처리

    실패 시 None 반환:
    - malformed 문자열이 들어와도 전체 학습이 깨지지 않게 방어한다.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        # "Z" suffix -> +00:00 로 치환
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        t = datetime.fromisoformat(s)
        # tz가 없으면 UTC로 가정(프로젝트 특성상 로컬 저장일 수 있어도 비교는 해야 함)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t
    except ValueError:
        return None

# ---------------------------------------------------------------------
# 유틸: 공동출현(co-occurrence) 그래프 유틸
# ---------------------------------------------------------------------
def _unique_in_order(items: Iterable[str]) -> list[str]:
    """
    리스트에서 중복을 제거하되, 처음 등장한 순서를 보존한다.
    """
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _add_symmetric_pairs(
    cooc: dict[str, dict[str, float]],
    songs: list[str],
    pair_weight: float = 1.0,
) -> None:
    """
    songs = [A, B, C] 같은 유저 선호 곡 집합이 있으면,
    모든 쌍 (A, B), (A, C), (B, C)에 대해 공동 출현 가중치를 더한다.

    '대칭(symmetric)'으로 누적하는 이유:
    - seed(추천의 기준곡)가 A인 경우에도 B를 찾고,
    - seed가 B인 경우에도 A를 찾을 수 있어야 하므로
      cooc[A][B] 와 cooc[B][A] 를 모두 증가시킨다.

    pair_weight 의미:
    함께 봐야 할 곡 쌍(A, B)에 얼마나 강하게 연결고리를 줄지를 나타내는 가중치 
    - Simple: 1.0(그 유저의 좋아요 집합에서 A, B 모두 등장하면 +1.0 같은 카우늩 부여)
    - Advanced(implicit): 선호 강도 기반 가중치 
        - 예: sqrt(aff_i * aff_j) 같은 값, 
        그 유저에 대해 곡 i, j에 쌓인 암시적 선호 점수
        (like/play는 +, skip/unlike는 -, 시간 감쇠 적용 후 양수인 곡들만 쌍을 만드는 식)
    """
    n = len(songs)
    # 2 곡 이하면 공동출현 없음, 가중치가 0이면 아무것도 안 함
    if n < 2 or pair_weight == 0:
        return

    for i in range(n):
        si = songs[i]
        for j in range(i + 1, n):
            sj = songs[j]
            if si == sj:
                continue

            cooc.setdefault(si, defaultdict(float))
            cooc.setdefault(sj, defaultdict(float))
            cooc[si][sj] += float(pair_weight)
            cooc[sj][si] += float(pair_weight)

def _freeze_cooc(cooc: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    defaultdict를 dict로 고정해두면 디버깅/직렬화 시 예측이 쉬워진다.

    왜 dict로 고정(freeze)해야 하는가?
    - defaultdict는 없는 키 접근하면 자동으로 0.0을 만들고 저장한다
    - 따라서 cooc["A"]["없는 키"]를 조회만 해도 키가 새로 생기기 때문에 
    디버깅하다가 프런트/탐색 중에 구조가 바뀔 수 있다.
    dict로 바꾸면 이런 자동 생성이 사라져서
    - 조회는 조회대로만 동작(없는 키는 에러/.get()으로 처리)
    - 로그/디버그 결과가 안정적
    - 직렬화(JSON 변환 등)할 때 타입이 단순해서 다루기 쉽다.
    """
    return {k: dict(v) for k, v in cooc.items()}

def _build_cooc_simple(inter_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    유저별로 action=='like' 인 song_id 만 모아 순서 유지 dedupe 후,
    그 집합 안의 모든 쌍에 동일 가중(1.0) 공동출현을 누적한다.
    """
    cooc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    if inter_df is None or inter_df.empty:
        return {}
    if "user_id" not in inter_df.columns or "song_id" not in inter_df.columns:
        return {}
    for _, grp in inter_df.groupby("user_id"):
        likes = grp.loc[grp["action"] == "like", "song_id"].astype(str).tolist()
        likes = _unique_in_order(likes)
        if len(likes) >= 2:
            _add_symmetric_pairs(cooc, likes, 1.0)
    return dict(cooc)

def _build_cooc_advanced(inter_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    유저별로 시간순으로 이벤트를 훑으며 곡별 affinity 를 누적한다.
    - like: +1 (시간 감쇠 곱)
    - play: 재생 길이에 비례한 작은 양수 (상한 1, 120초 기준 스케일 예시)
    - skip: 음수 페널티
    - unlike: 음수 페널티
    타임스탬프가 있으면 exp(-lambda * age_days) 로 최근 이벤트에 더 가중.
    마지막에 affinity > 0 인 곡들만 골라, 쌍 (i,j)에 sqrt(aff_i * aff_j) 를 더한다.
    """
    cooc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    if inter_df is None or inter_df.empty:
        return {}
    
    now = datetime.now(timezone.utc)
    lambda_per_day = 0.02 # 시간 감쇠 상수 (0.02일 당 20% 감쇠)

    for _, grp in inter_df.groupby("user_id"):
        if "timestamp" in grp.columns:
            grp = grp.sort_values("timestamp", na_position="last")
        aff: dict[str, float] = defaultdict(float)

        for _, row in grp.iterrows():
            sid = str(row["song_id"])
            action = str(row.get("action") or "")
            ts = _parse_ts(row.get("timestamp"))
            decay = 1.0 # decay란? 시간 감쇠 계수
            if ts is not None:
                age_days = max(0.0, (now - ts).total_seconds() / (24 * 3600))
                decay = math.exp(-lambda_per_day * age_days)

            if action == "like":
                aff[sid] += 1.0 * decay
            elif action == "play":
                ps = int(row.get("play_seconds") or 0)
                aff[sid] += min(1.0, ps / 120.0) * 0.35 * decay
            elif action == "skip":
                aff[sid] -= 0.45 * decay
            elif action == "unlike":
                aff[sid] -= 1.0 * decay

        positive = [s for s, v in aff.items() if v > 1e-9]
        positive.sort(key=lambda s: aff[s], reverse=True)
        n = len(positive)
        for i in range(n):
            si = positive[i]
            for j in range(i + 1, n):
                sj = positive[j]
                w = math.sqrt(aff[si] * aff[sj])
                if w <= 0:
                    continue
                cooc.setdefault(si, defaultdict(float))
                cooc.setdefault(sj, defaultdict(float))
                cooc[si][sj] += w
                cooc[sj][si] += w

    return dict(cooc)
# ---------------------------------------------------------------------
# 공통 믹스인: FAISS 점수 + cooc 점수 결합
# ---------------------------------------------------------------------
class _HybridMixer: 
    """
    Simple/Advanced 모두가 공유하는 "점수 결합" 로직만 모아둔 믹스인.

    이 믹스인이 가정하는 것:
    - self._index: MusicFaissIndex처럼 search_by_id()가 있는 객체 (FAISS) -> 오디오 유사곡을 찾아주는 검색기
    - self._cooc: {seed_song_id: {neighbo_song_id: score}} 형태의 공동출현 그래프 -> 함께 좋아한/함께 소비된 곡 관계도, {기준곡: {이웃곡: 관계강도}}
    - self._weight_content, self._weight_cooc: 결합 가중치 -> 두 축을 얼마나 믿을 지 정하는 비율
    - self._faiss_mult: FAISS 후보 풀 확장 계수 -> 최종 top_k보다 몇 배 넓게 FAISS 후보를 먼저 가져올지, 좁게 가져오면 cooc에서 좋은 곡이 후보군 밖으로 밀려서 하이브리드 효과가 줄어듦
    """
    _index: Any
    _cooc: dict[str, dict[str, float]]
    _weight_content: float
    _weight_cooc: float
    _faiss_mult: int

    def _normalized_weights(self) -> tuple[float, float]:
        """
        weight_content + weight_cooc가 1이 아니어도,
        비율만 유지하도록 정규화한다.
        """
        a = max(0.0, float(self._weight_content))
        b = max(0.0, float(self._weight_cooc))
        s = a + b
        if s < 1e-12:
            # 둘 다 0이면 콘텐츠만 쓰도록 fallback
            return 1.0, 0.0
        return a / s, b / s

    def _cooc_neighbors(self, seed_song_id: str) -> dict[str, float]:
        """seed 한 곡에 붙은 cooc 이웃만 복사본 dict 로 반환 (원본 변형 방지)."""
        return dict(self._cooc.get(seed_song_id, {}))

    def _hybrid_scores_for_seed(self, seed_song_id: str, pool_size: int) -> dict[str, float]:
        """
        seed_song_id 기준으로 후보 곡들의 최종 점수(가중합)를 계산한다.

        과정:
        1) FAISS로 후보 pool_size개(넉넉히) 가져옴 -> content_raw
        2) 공동출현 이웃 점수 -> coco_raw
        3) 후보집합 = content 후보 u cooc 이웃
        4) 두 점수 축을 각각 min-max 정규화
        5) wc * content + wk * cooc 로 합산
        """
        wc, wk = self._normalized_weights()

        # 콘텐츠(FAISS) 점수
        content_raw: dict[str, float] = {}
        # is_built: MusicFaissIndex의 속성, FAISS 인덱스가 구축되었는지 확인
        if self._index is not None and getattr(self._index, "is_built", False):
            content_raw = self._index.search_by_id(seed_song_id, top_k=pool_size)

        # cooc 점수
        cooc_raw = self._cooc_neighbors(seed_song_id)

        # 후보집합 (기준 곡 자신은 후보에서 제외)
        content_raw.pop(seed_song_id, None)
        cooc_raw.pop(seed_song_id, None)

        candidates = set(content_raw) | set(cooc_raw)
        if not candidates:
            return {}

        c_sub = {k: content_raw[k] for k in candidates if k in content_raw}
        k_sub = {k: cooc_raw[k] for k in candidates if k in cooc_raw}

        c_norm = _min_max_normalize(c_sub) if c_sub else {}
        k_norm = _min_max_normalize(k_sub) if k_sub else {}

        out: dict[str, float] = {}
        for sid in candidates:
            c_score = c_norm.get(sid, 0.0)
            k_score = k_norm.get(sid, 0.0)
            out[sid] = wc * c_score + wk * k_score
        return out
        
# ------------------------------------------------------------------------------
# 공개 추천기
# ------------------------------------------------------------------------------
class HybridFaissCoocRecommender(BaseRecommender, _HybridMixer):
    """
    하이브리드 추천 + 텍스트 검색은 TextEmbeddingSearcher 에 위임.
    """
    def __init__(
        self,
        faiss_index: Any,
        *,
        weight_content: float = 0.55,
        weight_cooc: float = 0.45,
        faiss_candidate_multiplier: int = 4,
    ) -> None:
        BaseRecommender.__init__(self, "hybrid_faiss_cooc")
        self._index = faiss_index
        self._weight_content = float(weight_content)
        self._weight_cooc = float(weight_cooc)
        self._default_wc = self._weight_content
        self._default_wk = self._weight_cooc
        self._faiss_mult = max(1, int(faiss_candidate_multiplier))
        self._data: pd.DataFrame | None = None
        self._song_genre: dict[str, str] = {}
        self._cooc_simple: dict[str, dict[str, float]] = {}
        self._cooc_advanced: dict[str, dict[str, float]] = {}
        self._cooc: dict[str, dict[str, float]] = {}
        self._text_searcher = TextEmbeddingSearcher(EmbedSearchConfig())

    def fit(self, data: pd.DataFrame) -> None:
        """
        곡 메타·장르 맵·공동출현 두 종류·텍스트 검색기 임베딩을 한 번에 학습한다.
        """
        self._data = data
        self._song_genre.clear()
        if data is not None and not data.empty and "song_id" in data.columns:
            for _, row in data.iterrows():
                sid = row.get("song_id")
                if sid is None or (isinstance(sid, float) and math.isnan(sid)):
                    continue
                self._song_genre[str(sid)] = str(row.get("genre") or "")

        inter = get_all_interactions()
        self._cooc_simple = _freeze_cooc(_build_cooc_simple(inter))
        self._cooc_advanced = _freeze_cooc(_build_cooc_advanced(inter))
        self._cooc = self._cooc_advanced

        self._text_searcher.fit(data if data is not None else pd.DataFrame())
        self.is_fitted = True

    def recommend(self, song_id: str, top_k: int = 10) -> dict[str, float]:
        return self.recommend_with_options(song_id, top_k, {})

    def recommend_with_options(
        self,
        song_id: str,
        top_k: int,
        options: dict[str, Any] | None,
    ) -> dict[str, float]:
        """
        mode 로 simple/advanced cooc 그래프 선택.
        weights 가 오면 해당 요청에만 임시로 반영하고 끝나면 기본값으로 복구한다.
        """
        self._check_fitted()
        opts = options or {}

        mode = str(opts.get("mode", "advanced")).lower().strip()
        if mode == "simple":
            self._cooc = self._cooc_simple
        else:
            self._cooc = self._cooc_advanced

        w = opts.get("weights") or {}
        wc = float(w.get("content", self._default_wc))
        wk = float(w.get("cooc", self._default_wk))
        old_c, old_k = self._weight_content, self._weight_cooc
        self._weight_content = wc
        self._weight_cooc = wk

        try:
            pool_size = max(top_k * self._faiss_mult, top_k, 1)
            scores = self._hybrid_scores_for_seed(song_id, pool_size)

            th = float(opts.get("threshold", 0.0))
            if th > 0:
                scores = {k: v for k, v in scores.items() if v >= th}

            scores.pop(song_id, None)

            filters = opts.get("filters") or {}
            genre_f = filters.get("genre") if isinstance(filters, dict) else None
            if genre_f and self._song_genre:
                g = str(genre_f)
                scores = {
                    sid: sc
                    for sid, sc in scores.items()
                    if self._song_genre.get(sid) == g
                }

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, top_k)]
            return dict(ranked)
        finally:
            self._weight_content = old_c
            self._weight_cooc = old_k

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> dict[str, float]:
        return self.recommend_for_user_with_options(user_id, top_k, {})

    def recommend_for_user_with_options(
        self,
        user_id: int,
        top_k: int,
        options: dict[str, Any] | None,
    ) -> dict[str, float]:
        """
        현재 좋아요 목록의 각 곡을 seed 로 추천을 구한 뒤,
        같은 후보에 대해서는 점수를 max 로 합성해 한 리스트로 만든다.
        (여러 seed 가 같은 후보를 밀어줄 때 과도한 합산을 피하기 위함)
        """
        self._check_fitted()
        likes = get_user_liked_song_ids(user_id)
        if not likes:
            return {}

        merged: dict[str, float] = {}
        pool = max(top_k * 3, top_k + 5)
        for sid in likes:
            part = self.recommend_with_options(sid, pool, options)
            for k, v in part.items():
                merged[k] = max(merged.get(k, 0.0), float(v))

        for sid in likes:
            merged.pop(sid, None)

        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[: max(1, top_k)]
        return dict(ranked)

    def search_by_query(self, query: str, top_k: int = 10) -> dict[str, float]:
        return self.search_by_query_with_options(query, top_k, {})

    def search_by_query_with_options(
        self,
        query: str,
        top_k: int,
        options: dict[str, Any] | None,
    ) -> dict[str, float]:
        """
        B 파이프라인 전체는 TextEmbeddingSearcher.search_with_options 에 위임하고, 텍스트 검색 점수를 하이브리드 점수에 추가한다.
        API 가 넘기는 dict 를 그대로 넘겨도 되고, 알 수 없는 키는 검색기 쪽에서 무시된다.
        """
        self._check_fitted()
        return self._text_searcher.search_with_options(query, top_k, options or {})
