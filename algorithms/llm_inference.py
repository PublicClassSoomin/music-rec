"""
LLM 기반 음악 추천 알고리즘 (LangChain + LangGraph + OpenAI GPT)

흐름:
  1. 사용자가 자연어로 원하는 음악 분위기를 설명
     예: "새벽에 공부할 때 듣기 좋은 잔잔한 피아노 음악"
  2. LangGraph 워크플로우가 GPT에게 질의 → 17차원 오디오 특성 벡터 생성
  3. 생성된 벡터로 FAISS 인덱스에서 코사인 유사도 기반 검색
  4. {song_id: score} 형태로 추천 결과 반환

벡터 차원 (17개, data/faiss_index.py의 FEATURE_COLS 순서):
  mfcc_1~13, bpm, energy, spectral_centroid, zcr

등록:
  api/main.py의 startup()에서:
    from algorithms.llm_inference import LLMInferenceRecommender
    recommenders["llm_inference"] = LLMInferenceRecommender(faiss_index)
    recommenders["llm_inference"].fit(song_df)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from typing import TypedDict, Any

from algorithms.base import BaseRecommender
from data.database import get_user_liked_song_ids
from data.faiss_index import FEATURE_COLS

# ── LangChain / LangGraph imports ────────────────────────

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel as PydanticBaseModel, Field

from langgraph.graph import StateGraph, END


# ── Pydantic 스키마: GPT 반환 규격 ─────────────────────────

class AudioFeatureVector(PydanticBaseModel):
    """GPT가 반환해야 하는 17차원 오디오 특성 벡터 스키마."""
    mfcc_1: float = Field(description="MFCC 1번째 계수 (보통 -30 ~ 30 범위, 전체 에너지 분포)")
    mfcc_2: float = Field(description="MFCC 2번째 계수 (보통 -30 ~ 30 범위, 스펙트럼 기울기)")
    mfcc_3: float = Field(description="MFCC 3번째 계수 (보통 -20 ~ 20)")
    mfcc_4: float = Field(description="MFCC 4번째 계수 (보통 -20 ~ 20)")
    mfcc_5: float = Field(description="MFCC 5번째 계수 (보통 -15 ~ 15)")
    mfcc_6: float = Field(description="MFCC 6번째 계수 (보통 -15 ~ 15)")
    mfcc_7: float = Field(description="MFCC 7번째 계수 (보통 -15 ~ 15)")
    mfcc_8: float = Field(description="MFCC 8번째 계수 (보통 -10 ~ 10)")
    mfcc_9: float = Field(description="MFCC 9번째 계수 (보통 -10 ~ 10)")
    mfcc_10: float = Field(description="MFCC 10번째 계수 (보통 -10 ~ 10)")
    mfcc_11: float = Field(description="MFCC 11번째 계수 (보통 -10 ~ 10)")
    mfcc_12: float = Field(description="MFCC 12번째 계수 (보통 -10 ~ 10)")
    mfcc_13: float = Field(description="MFCC 13번째 계수 (보통 -10 ~ 10)")
    bpm: float = Field(description="BPM (beats per minute). 느린 발라드=60~80, 팝=100~130, EDM=120~160, 빠른 댄스=140+")
    energy: float = Field(description="에너지 (RMS 평균). 조용한 음악=0.01~0.03, 보통=0.03~0.08, 시끄러운=0.08~0.15")
    spectral_centroid: float = Field(description="스펙트럼 센트로이드 (밝기). 어두운/따뜻한=800~1500, 보통=1500~3000, 밝은/날카로운=3000~5000")
    zcr: float = Field(description="제로 크로싱 레이트. 클린/보컬=0.03~0.07, 보통=0.07~0.12, 노이즈/타악기=0.12~0.20")


# ── LangGraph State 정의 ──────────────────────────────────

class GraphState(TypedDict):
    """LangGraph 워크플로우 상태."""
    query: str                          # 사용자 자연어 입력
    feature_stats: str                  # DB 특성 통계 (프롬프트에 주입)
    vector: list[float] | None         # GPT가 생성한 17차원 벡터
    results: dict[str, float] | None   # FAISS 검색 결과
    error: str | None                   # 에러 메시지


# ── 시스템 프롬프트 ────────────────────────────────────────

SYSTEM_PROMPT = """당신은 음악 오디오 분석 전문가입니다.
사용자가 원하는 음악의 분위기, 감정, 상황을 설명하면,
해당 음악의 오디오 특성을 17차원 수치 벡터로 추정해야 합니다.

## 벡터 구성 (총 17차원)
1. mfcc_1 ~ mfcc_13: Mel-Frequency Cepstral Coefficients
   - 음색(timbre)의 수학적 표현
   - mfcc_1: 전체 에너지 레벨 (높으면 에너지 높은 음악)
   - mfcc_2: 스펙트럼 기울기 (양수=밝은, 음수=어두운)
   - mfcc_3~13: 세부 음색 디테일
2. bpm: 분당 박자수 (tempo)
3. energy: 음량/에너지 (RMS 평균)
4. spectral_centroid: 주파수 무게중심 (높으면 밝은 소리)
5. zcr: 제로 크로싱 레이트 (높으면 노이즈/타악기 많음)

## 현재 DB의 오디오 특성 통계 (참고용)
{feature_stats}

## 장르별 특성 가이드라인
- 잔잔한 발라드/피아노: 낮은 bpm(60-85), 낮은 energy(0.01-0.03), 낮은 spectral_centroid(800-1500), 낮은 zcr(0.03-0.06)
- 어쿠스틱/포크: 중간 bpm(80-110), 낮은 energy(0.02-0.04), 중간 spectral_centroid(1200-2000), 낮은 zcr(0.04-0.07)
- K-POP/팝: 중간~높은 bpm(100-130), 중간 energy(0.04-0.08), 중간~높은 spectral_centroid(2000-3500), 중간 zcr(0.07-0.11)
- 힙합/랩: 중간 bpm(80-110), 중간~높은 energy(0.05-0.10), 중간 spectral_centroid(1500-3000), 중간~높은 zcr(0.08-0.13)
- EDM/댄스: 높은 bpm(120-160), 높은 energy(0.07-0.15), 높은 spectral_centroid(3000-5000), 높은 zcr(0.10-0.18)
- Lo-fi/Chill: 낮은 bpm(70-95), 낮은 energy(0.02-0.04), 낮은~중간 spectral_centroid(1000-2000), 낮은 zcr(0.04-0.08)
- 재즈: 중간 bpm(80-140), 중간 energy(0.03-0.06), 중간 spectral_centroid(1500-2500), 중간 zcr(0.06-0.10)
- 클래식: 다양한 bpm(50-160), 낮은~중간 energy(0.02-0.06), 다양한 spectral_centroid, 낮은 zcr(0.03-0.07)
- 록/메탈: 높은 bpm(100-180), 높은 energy(0.08-0.15), 높은 spectral_centroid(2500-5000), 높은 zcr(0.10-0.20)

## 중요 규칙
- 반드시 JSON 형식으로만 응답하세요.
- 모든 17개 필드를 빠짐없이 포함하세요.
- DB 통계의 min/max 범위 안에서 값을 설정하세요.
- 사용자의 분위기 설명을 음악적 특성으로 정확히 변환하세요."""

HUMAN_PROMPT = """사용자 요청: {query}

위 요청에 맞는 음악의 오디오 특성 벡터를 JSON으로 반환해주세요."""


# ── LLMInferenceRecommender 본체 ──────────────────────────

class LLMInferenceRecommender(BaseRecommender):
    """
    GPT + LangGraph 기반 자연어 → 오디오 벡터 → FAISS 코사인 유사도 추천기.

    search_by_query("새벽에 듣기 좋은 잔잔한 음악", top_k=10)
    → GPT가 17차원 벡터 생성 → FAISS search_by_vector → {song_id: score}
    """

    def __init__(self, faiss_index, model: str = "gpt-5.4-2026-03-05", temperature: float = 0.3):
        super().__init__("llm_inference")
        self._index = faiss_index
        self._data: pd.DataFrame | None = None
        self._feature_stats: str = ""
        self._model = model
        self._temperature = temperature
        self._graph = None          # LangGraph 컴파일된 워크플로우
        self._llm = None

    # ── fit: 데이터 로드 + 통계 계산 + LangGraph 빌드 ──────

    def fit(self, data: pd.DataFrame) -> None:
        self._data = data
        self._feature_stats = self._compute_feature_stats(data)
        self._build_graph()
        self.is_fitted = True
        print(f"[LLM-Inference] fit 완료 (곡 수: {len(data)}, 모델: {self._model})")

    def _compute_feature_stats(self, data: pd.DataFrame) -> str:
        """DB에 있는 오디오 특성의 min/max/mean/std 통계를 문자열로 생성."""
        feat_df = data[FEATURE_COLS].dropna()
        if feat_df.empty:
            return "통계 데이터 없음 (audio_features 비어 있음)"

        lines = []
        for col in FEATURE_COLS:
            s = feat_df[col]
            lines.append(
                f"  {col}: min={s.min():.4f}, max={s.max():.4f}, "
                f"mean={s.mean():.4f}, std={s.std():.4f}"
            )
        return "\n".join(lines)

    # ── LangGraph 워크플로우 구축 ──────────────────────────

    def _build_graph(self) -> None:
        """LangGraph StateGraph: generate_vector → search_faiss."""
        from utils.config import OPENAI_API_KEY

        if not OPENAI_API_KEY:
            print("[LLM-Inference] ⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
            print("[LLM-Inference]    .env 파일에 OPENAI_API_KEY=sk-... 를 추가하세요.")
            return

        # LLM 초기화
        self._llm = ChatOpenAI(
            model=self._model,
            temperature=self._temperature,
            api_key=OPENAI_API_KEY,
        )

        # LangGraph 정의
        workflow = StateGraph(GraphState)

        workflow.add_node("generate_vector", self._node_generate_vector)
        workflow.add_node("search_faiss", self._node_search_faiss)

        workflow.set_entry_point("generate_vector")
        workflow.add_edge("generate_vector", "search_faiss")
        workflow.add_edge("search_faiss", END)

        self._graph = workflow.compile()
        print("[LLM-Inference] LangGraph 워크플로우 빌드 완료")

    # ── LangGraph 노드들 ──────────────────────────────────

    def _node_generate_vector(self, state: GraphState) -> dict[str, Any]:
        """
        노드 1: GPT에게 자연어 질의를 보내고 17차원 벡터를 JSON으로 받는다.
        LangChain ChatPromptTemplate + JsonOutputParser 사용.
        """
        query = state["query"]
        feature_stats = state["feature_stats"]

        # 프롬프트 구성
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])

        # JSON 파서 (AudioFeatureVector 스키마 기반)
        parser = JsonOutputParser(pydantic_object=AudioFeatureVector)

        # LangChain 체인 실행
        chain = prompt | self._llm | parser

        try:
            result = chain.invoke({
                "query": query,
                "feature_stats": feature_stats,
            })

            # 결과를 FEATURE_COLS 순서의 벡터로 변환
            vector = [float(result.get(col, 0.0)) for col in FEATURE_COLS]

            print(f"[LLM-Inference] 벡터 생성 완료: bpm={result.get('bpm')}, "
                  f"energy={result.get('energy')}, centroid={result.get('spectral_centroid')}")

            return {"vector": vector, "error": None}

        except Exception as e:
            print(f"[LLM-Inference] GPT 벡터 생성 실패: {e}")
            return {"vector": None, "error": str(e)}

    def _node_search_faiss(self, state: GraphState) -> dict[str, Any]:
        """
        노드 2: 생성된 벡터로 FAISS 인덱스에서 코사인 유사도 검색.
        """
        vector = state.get("vector")
        if vector is None:
            return {"results": {}, "error": state.get("error", "벡터 생성 실패")}

        if not getattr(self._index, "is_built", False):
            return {"results": {}, "error": "FAISS 인덱스가 구축되지 않았습니다."}

        try:
            vec_np = np.array(vector, dtype=np.float32)
            results = self._index.search_by_vector(vec_np, top_k=50)
            return {"results": results, "error": None}
        except Exception as e:
            print(f"[LLM-Inference] FAISS 검색 실패: {e}")
            return {"results": {}, "error": str(e)}

    # ── LangGraph 실행 ────────────────────────────────────

    def _run_graph(self, query: str, top_k: int = 10) -> dict[str, float]:
        """LangGraph 워크플로우 실행 → {song_id: score} 반환."""
        if self._graph is None:
            print("[LLM-Inference] LangGraph가 초기화되지 않았습니다. OPENAI_API_KEY를 확인하세요.")
            return {}

        initial_state: GraphState = {
            "query": query,
            "feature_stats": self._feature_stats,
            "vector": None,
            "results": None,
            "error": None,
        }

        final_state = self._graph.invoke(initial_state)

        results = final_state.get("results") or {}
        error = final_state.get("error")
        if error:
            print(f"[LLM-Inference] 워크플로우 에러: {error}")

        # top_k로 잘라서 반환
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_results)

    # ── BaseRecommender 인터페이스 구현 ────────────────────

    def recommend(self, song_id: str, top_k: int = 10) -> dict[str, float]:
        """
        곡 기반 추천: song_id의 메타데이터(제목+가수)를 자연어 질의로 변환 후 LLM 추천.
        기존 FAISS CBF보다 느리지만, 곡의 '느낌'을 이해한 추천이 가능.
        """
        self._check_fitted()

        if self._data is None or self._data.empty:
            return {}

        # song_id에서 곡 정보 추출 → 자연어 질의로 변환
        row = self._data[self._data["song_id"] == song_id]
        if row.empty:
            return {}

        r = row.iloc[0]
        title = r.get("title", "")
        artist = r.get("artist", "")
        genre = r.get("genre", "")

        query = f"'{title}' by {artist} ({genre}) 와 비슷한 분위기의 음악"
        results = self._run_graph(query, top_k + 1)

        # 자기 자신 제외
        results.pop(song_id, None)

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_results)

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> dict[str, float]:
        """
        유저 기반 추천: 좋아요한 곡들의 정보를 종합하여 LLM에게 질의.
        """
        self._check_fitted()

        likes = get_user_liked_song_ids(user_id)
        if not likes:
            return {}

        if self._data is None or self._data.empty:
            return {}

        # 좋아요한 곡들의 정보를 모아서 취향 설명 생성
        liked_descriptions = []
        for sid in likes[:5]:  # 최대 5곡만 사용 (토큰 절약)
            row = self._data[self._data["song_id"] == sid]
            if not row.empty:
                r = row.iloc[0]
                liked_descriptions.append(f"'{r.get('title', '')}' - {r.get('artist', '')}")

        if not liked_descriptions:
            return {}

        songs_text = ", ".join(liked_descriptions)
        query = f"다음 곡들을 좋아하는 사용자의 취향에 맞는 음악: {songs_text}"

        results = self._run_graph(query, top_k + len(likes))

        # 이미 좋아요한 곡 제외
        for sid in likes:
            results.pop(sid, None)

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_results)

    def search_by_query(self, query: str, top_k: int = 10) -> dict[str, float]:
        """
        자연어 검색 (이 알고리즘의 핵심 기능).
        사용자가 "새벽에 공부할 때 듣기 좋은 잔잔한 음악" 같이 입력하면
        GPT가 음악적 특성 벡터를 생성하고 FAISS로 유사곡을 찾아 반환.
        """
        self._check_fitted()

        if not query or not query.strip():
            return {}

        return self._run_graph(query.strip(), top_k)
