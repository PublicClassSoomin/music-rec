from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd

from rapidfuzz import fuzz

from utils.config import GEMINI_API_KEY, LLM_MODEL
# sentence-transformers 는 텍스트 임베딩을 위한 라이브러리, 대규모 텍스트 데이터를 벡터 공간으로 변환하는 역할
from sentence_transformers import SentenceTransformer

def _safe_song_text(row: pd.Series) -> str:
    title = str(row.get("title") or "").strip()
    artist = str(row.get("artist") or "").strip()
    genre = str(row.get("genre") or "").strip()
    return "".join([title, artist, genre]).strip()

def _parse_keywords(text: str, max_items: int = 10) -> list[str]:
    if not text:
        return []
    # 쉼표/개행/슬래시 등으로 분리
    parts = re.split(r"[,/\n]+", text)
    out = []
    for p in parts:
        kw = p.strip()
        # 너무 짧은 토큰은 제거("a", "of" 같은 것 방지)
        if len(kw) < 2:
            continue
        out.append(kw)
        if len(out) >= max_items:
            break
    return out

class GeminiQueryExpander:
    """
    Gemini API로 query를 의미 있는 키워드/태그로 확장한 뒤,
    그 확장 텍스트를 다시 임베딩에 넣는 방식.
    """
    def __init__(self, model_name: str = LLM_MODEL, api_key: Optional[str] = None, temperature: float = 0.2):
        self.model_name = model_name
        self.api_key = api_key or GEMINI_API_KEY
        self.temperature = temperature # temperature 가 낮아야 일관되고 논리적인 답변 (높으면 창의적)

        # langchain-google-genai 를 사용하여 Gemini API 호출
        self._llm = None
        if self.api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=self.temperature,
                )
            except Exception:
                self._llm = None

    def is_available(self) -> bool:
        return self._llm is not None

    def expand(self, query: str, max_keywords: int = 8) -> str:
        """
        실패하면 query 그대로 반환
        """
        q = (query or "").strip()
        if not q or not self.is_available():
            return q

        prompt = (
            "You are an expert music listener. "
            "The user provided a short music preference query. "
            "Retrun 6 to {max_keywords} concise search keywords/tags that would be help retrieve songs matching the user's preference. "
            "using text search. Use Korean/English keywords if appropriate. "
            "Respond with keywords separated by commas only.\n\n" # 지시사항 끝
            "User query: {query}\n" # 입력데이터 구분을 위함
        ).format(max_keywords=max_keywords, query=q)

        try:
            from langchain_core.messages import HumanMessage
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            content = getattr(resp, "content", "") or ""
            kws = _parse_keywords(content, max_items=max_keywords)
            if not kws:
                return q
            return "".join(kws)
        except Exception as e:
            print(f"Error in keyword extraction: {e}")
            return q

# 검색 질의에 자주 붙지만 곡을 특정하지 못하는 일반어 (메타데이터에 과도하게 걸리는 것 방지)
_SEARCH_STOPWORDS = frozenset(
    {
        "노래", "음악", "곡", "뮤직", "music", "song", "songs", "track", "tracks", "audio",
        "플레이리스트", "playlist", "추천", "recommend", "recommended",
        "듣기", "listen", "listening",
        "요즘", "최신", "신곡", "new", "latest", "recent",
        "좋은", "best", "top", "인기", "popular", "hot", "hit",
        "some", "the", "a", "an", "and", "or", "for", "to", "of",
        "같은", "같이", "할때", "때", "위한", "으로", "해서", "있는",
    }
)


def _query_tokens_split(query: str) -> tuple[list[str], list[str]]:
    """
    공백 토큰을 나누고, (의미 토큰, 일반어 토큰)으로 분리한다.
    의미 토큰이 하나도 없으면 빈 리스트를 반환 → 호출부에서 임베딩 검색으로 넘긴다.
    """
    q = (query or "").strip()
    if not q:
        return [], []
    raw = [t for t in q.lower().split() if len(t) >= 2]
    if not raw:
        raw = [q.lower().strip()]
    significant = [t for t in raw if t not in _SEARCH_STOPWORDS]
    generic = [t for t in raw if t in _SEARCH_STOPWORDS]
    return significant, generic


# 문자 유사도 키워드 검색 (LLM 없음, 임베딩 없음)
def _similarity_token_score(token: str, corpus: str) -> float:
    """
    질의 토큰 하나와 곡 메타 문자열(corpus) 사이의 유사도 -> 0~1.

    - rapidfuzz: partial_ratio (부분 일치)
    """
    t = (token or "").lower().strip()
    c = (corpus or "").lower().strip()
    if not t or not c:
        return 0.0
    return float(fuzz.partial_ratio(t, c)) / 100.0

def _keyword_search_similarity(
    data: pd.DataFrame,
    query: str,
    pool_size: int,
    *,
    min_token_score: float = 0.55, # 토큰 유사도 최소 점수
) -> dict[str, float]:
    """
    검색 파이프라인 1단계 (메타데이터: 제목·가수·장르).

    - 일반어(노래, 음악, 요즘, 추천 …)는 곡을 특정하지 못하므로 **의미 토큰**으로 쓰지 않는다.
      질의가 일반어뿐이면 빈 dict 를 반환하고, 상위에서 임베딩 검색으로 넘긴다.
    - '슬픈', '발라드', 아티스트명 등 **의미 토큰**은 OR 에 가깝게: 하나라도 강하게 맞으면 후보에 들어가고,
      여러 개가 맞을수록 점수가 올라간다.
    - 질의 토큰이 **장르 필드**와 유사하면 가산점 (장르 컬럼만 따로 비교).
    """
    if data is None or data.empty:
        return {}

    q = (query or "").strip()
    if not q:
        return {}

    significant, generic = _query_tokens_split(q)
    if not significant:
        return {}

    scores: dict[str, float] = {}
    genre_floor = max(0.48, min_token_score - 0.07)

    for _, row in data.iterrows():
        sid = row.get("song_id")
        if not sid:
            continue
        title = str(row.get("title") or "")
        artist = str(row.get("artist") or "")
        genre_s = str(row.get("genre") or "").strip()
        corpus = " ".join([title, artist, genre_s]).strip()
        if not corpus:
            continue

        genre_lower = genre_s.lower()
        sig_hits: list[float] = []
        for tok in significant:
            s = _similarity_token_score(tok, corpus)
            if s >= min_token_score:
                sig_hits.append(s)

        if not sig_hits:
            continue

        # OR 성향: 최고 일치 + 평균을 섞어 다중 키워드(슬픈 발라드)도 반영
        best = max(sig_hits)
        mean_hit = sum(sig_hits) / len(significant)
        final_score = 0.55 * best + 0.45 * mean_hit

        if genre_lower:
            for tok in significant:
                g_sc = _similarity_token_score(tok, genre_lower)
                if g_sc >= genre_floor:
                    final_score = min(1.0, final_score + 0.14)
                    break

        for tok in generic:
            s = _similarity_token_score(tok, corpus)
            if s >= min_token_score:
                final_score = min(1.0, final_score + 0.04)

        scores[str(sid)] = float(final_score)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, pool_size)]
    return dict(ranked)


def _gate_scores_by_search_query(
    df: pd.DataFrame,
    query: str,
    scores: dict[str, float],
    *,
    min_token_score: float,
    fallback_if_empty: bool = True,
) -> dict[str, float]:
    """
    검색창에 입력한 질의(query)만으로 메타데이터 필터를 적용한다.
    제목·가수·장르 문자열에 대해 1단계(_keyword_search_similarity)와 같은 토큰 규칙으로
    '연결된다'고 판정된 곡 ID만 점수 dict 에 남긴다.

    전체 임베딩 검색(use_full_embedding_search)처럼 질의와 직접 무관한 이웃이 섞일 때
    사용자가 입력한 문장과 맞지 않는 결과를 걸러 내는 용도다.
    """
    if df is None or df.empty or not scores:
        return scores
    q = (query or "").strip()
    if not q:
        return scores
    pool = max(len(df), len(scores) * 4, 64)
    kw = _keyword_search_similarity(df, q, pool, min_token_score=min_token_score)
    allowed = set(kw.keys())
    gated = {k: v for k, v in scores.items() if k in allowed}
    if not gated and fallback_if_empty:
        return scores
    return gated


@dataclass
class EmbedSearchConfig:
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    # 벡터의 길이를 1로 맞추는 작업(L2 Normalization)을 할지 여부
    # 즉, 길이는 보지 않고 방향(의미) 유사도만 보게 된다.
    # 또한, 모든 벡터의 길이가 1이면 코사인 유사도 공식이 아주 단순한 내적 계산으로 바뀜
    use_normalized_embeddings: bool = True

class TextEmbeddingSearcher:
    """
    검색 파이프라인을 한 곳에서 담당.

    - fit: 곡 메타 임베딩 + DataFrame 보관(문자 1단계용)
    - search: 전 카탈로그 임베딩 top-k (레거시/비교용)
    - search_with_options: B 방식 (문자 1단계 → 옵션으로 임베딩 리랭크 또는 전체 임베딩)
    """
    def __init__(self, cfg: EmbedSearchConfig | None = None):
        self.cfg = cfg or EmbedSearchConfig()
        self._model = SentenceTransformer(self.cfg.embedding_model_name)
        self._song_ids: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._df: Optional[pd.DataFrame] = None
        self._normalization_mode = self.cfg.use_normalized_embeddings
        self._llm_expander = GeminiQueryExpander()

    def fit(self, data: pd.DataFrame) -> None:
        """
        data: get_all_songs() 결과(DataFrame).
        """
        self._df = None
        self._song_ids = []
        self._embeddings = None
        if data is None or data.empty:
            return

        # song_id 유효성 체크
        df = data.copy()
        df = df[df["song_id"].notna()]
        if df.empty:
            return

        self._df = df
        self._song_ids = [str(x) for x in df["song_id"].tolist()]
        corpus = df.apply(_safe_song_text, axis=1).tolist()

        # SenteceTransformers encode:
        # normalize_embedding=True 이면 dot product = cosine similarity 가 된다.
        emb = self._model.encode(
            corpus,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self._normalization_mode,
        )
        emb = np.asarray(emb, dtype=np.float32)
        self._embeddings = emb

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        use_llm_expansion: bool = True,
        llm_max_keywords: int = 8,
    ) -> dict[str, float]:
        """
        전 카탈로그 임베딩 검색
        return: {song_id: score}
        """
        if self._embeddings is None or not self._song_ids:
            return {}

        q = (query or "").strip()
        if not q:
            return {}

        expanded = q
        if use_llm_expansion:
            expanded = self._llm_expander.expand(q, max_keywords=llm_max_keywords)

        q_emb = self._model.encode(
            [expanded],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self._normalization_mode,
        )
        q_emb = np.asarray(q_emb[0], dtype=np.float32)

        # cosine similarity(정규화 했으니 dot product)
        scores = self._embeddings @ q_emb

        n = len(scores)
        k = min(top_k, n)
        # argpartition: 부분 정렬 후 상위 k개의 인덱스 반환
        idx_part = np.argpartition(scores, -k)[-k:]
        idx_sorted = idx_part[np.argsort(scores[idx_part])][::-1]

        return {self._song_ids[i]: float(scores[i]) for i in idx_sorted.tolist()}

    def _encode_query(self, query: str, *, use_llm: bool, llm_max_keywords: int) -> np.ndarray:
        q = (query or "").strip()
        if not q:
            return np.array([], dtype=np.float32)
        expanded = self._llm_expander.expand(q, max_keywords=llm_max_keywords) if use_llm else q
        q_emb = self._model.encode(
            [expanded],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self._normalization_mode,
        )
        return np.asarray(q_emb[0], dtype=np.float32)

    def _rerank_candidates(
        self,
        query: str,
        candidate_ids: list[str],
        top_k: int,
        *,
        use_llm: bool,
        llm_max_keywords: int = 8,
    ) -> dict[str, float]:
        """후보 song_id만 임베딩 유사도로 재순위."""
        if self._embeddings is None or not self._song_ids or not candidate_ids:
            return {}
        id_to_idx = {sid: i for i, sid in enumerate(self._song_ids)}
        indices = [id_to_idx[sid] for sid in candidate_ids if sid in id_to_idx]
        if not indices:
            return {}
        q_emb = self._encode_query(query, use_llm=use_llm, llm_max_keywords=llm_max_keywords)
        if q_emb.size == 0:
            return {}
        sub = self._embeddings[indices]
        sims = sub @ q_emb
        pairs = list(zip([self._song_ids[i] for i in indices], sims.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return {sid: float(sc) for sid, sc in pairs[:top_k]}
    
    def search_with_options(
        self,
        query: str,
        top_k: int = 10,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, float]:
        """
        B 파이프라인 (하이브리드·API 공통 진입점).
        options:
          use_full_embedding_search: bool — True면 1단계 생략하고 search()만
          use_embedding_rerank: bool — True면 1단계 후보만 _rerank_candidates
          use_llm_search: bool — 임베딩 단계 질의 확장 (1단계 문자 검색에는 미사용)
          search_candidate_multiplier: int — 1단계 pool = top_k * mult
          min_token_score: float — 1단계 토큰 유사도 하한
          require_keyword_match: bool — use_full_embedding_search 일 때만 사용. True면 임베딩 상위 결과를
            검색창 질의와 메타(제목·가수·장르) 키워드 규칙으로 걸러 냄 (기본 True)
        """
        opts = options or {}
        use_full = bool(opts.get("use_full_embedding_search", False))
        use_rerank = bool(opts.get("use_embedding_rerank", False))
        use_llm = bool(opts.get("use_llm_search", False))
        mult = max(1, int(opts.get("search_candidate_multiplier", 8)))
        min_tok = float(opts.get("min_token_score", 0.55))
        if use_full:
            out = self.search(
                query,
                top_k,
                use_llm_expansion=use_llm,
                llm_max_keywords=int(opts.get("llm_max_keywords", 8)),
            )
            if bool(opts.get("require_keyword_match", True)) and self._df is not None:
                out = _gate_scores_by_search_query(
                    self._df, query, out, min_token_score=min_tok, fallback_if_empty=True
                )
            return out
        if self._df is None or self._df.empty:
            return {}
        pool_size = max(top_k * mult, top_k + 5)
        stage1 = _keyword_search_similarity(
            self._df, query, pool_size, min_token_score=min_tok
        )
        # 질의가 '요즘 노래'처럼 일반어뿐이면 1단계는 비우고, 문장 전체 의미는 임베딩으로 찾는다.
        if not stage1:
            if self._embeddings is None:
                return {}
            out = self.search(
                query,
                top_k,
                use_llm_expansion=use_llm,
                llm_max_keywords=int(opts.get("llm_max_keywords", 8)),
            )
            if bool(opts.get("require_keyword_match", True)) and self._df is not None:
                out = _gate_scores_by_search_query(
                    self._df, query, out, min_token_score=min_tok, fallback_if_empty=True
                )
            return out
        if not use_rerank:
            ranked = sorted(stage1.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return dict(ranked)
        candidate_ids = [s for s, _ in sorted(stage1.items(), key=lambda x: x[1], reverse=True)]
        stage2 = self._rerank_candidates(
            query,
            candidate_ids,
            top_k,
            use_llm=use_llm,
            llm_max_keywords=int(opts.get("llm_max_keywords", 8)),
        )
        if stage2:
            return stage2

        return dict(sorted(stage1.items(), key=lambda x: x[1], reverse=True)[:top_k])

    