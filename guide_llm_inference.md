# 🤖 LLM Inference 추천 알고리즘 가이드

> **알고리즘 이름**: `llm_inference`
> **파일 위치**: `algorithms/llm_inference.py`
> **핵심 기술**: OpenAI GPT + LangChain + LangGraph + FAISS 코사인 유사도
> **브랜치**: `algo/KAN-14`

---

## 📖 목차

1. [한줄 요약](#1-한줄-요약)
2. [기존 알고리즘(faiss_cbf)과의 차이](#2-기존-알고리즘faiss_cbf과의-차이)
3. [전체 아키텍처](#3-전체-아키텍처)
4. [LangGraph 워크플로우 상세](#4-langgraph-워크플로우-상세)
5. [코드 구조 분석](#5-코드-구조-분석)
6. [프로젝트 연동 구조](#6-프로젝트-연동-구조)
7. [환경 설정 및 실행 방법](#7-환경-설정-및-실행-방법)
8. [사용자 시나리오별 동작 흐름](#8-사용자-시나리오별-동작-흐름)
9. [핵심 기술 개념 설명](#9-핵심-기술-개념-설명)
10. [프롬프트 엔지니어링 설계](#10-프롬프트-엔지니어링-설계)
11. [한계점 및 개선 방향](#11-한계점-및-개선-방향)
12. [트러블슈팅 FAQ](#12-트러블슈팅-faq)

---

## 1. 한줄 요약

> 사용자가 **자연어**로 원하는 음악 분위기를 말하면, **GPT가 그 느낌을 17차원 오디오 특성 벡터로 변환**하고, **FAISS가 DB에서 가장 비슷한 곡을 찾아** 추천합니다.

```
"새벽에 공부할 때 듣기 좋은 잔잔한 피아노 음악"
        │
        ▼ GPT (LangChain)
  [mfcc_1=-8.2, ..., bpm=72, energy=0.02, centroid=1100, zcr=0.04]
        │
        ▼ FAISS (코사인 유사도)
  🎵 추천: 잔잔한 피아노곡 3개 + 유사도 점수
```

---

## 2. 기존 알고리즘(faiss_cbf)과의 차이

| 비교 항목        | `faiss_cbf` (기존)                    | `llm_inference` (신규)                      |
| ---------------- | ------------------------------------- | ------------------------------------------- |
| **입력**         | song_id (곡 선택)                     | 자연어 텍스트 ("새벽에 듣기 좋은 음악")     |
| **추천 근거**    | 곡 A의 오디오 수치와 곡 B의 수치 비교 | GPT가 **분위기를 이해**하고 수치로 변환     |
| **자연어 검색**  | 키워드 매칭 (단순 문자열 포함 여부)   | **의미 기반 검색** (GPT가 의도 해석)        |
| **속도**         | 밀리초 단위 (즉시)                    | 1~3초 (GPT API 호출 필요)                   |
| **비용**         | 무료 (로컬 계산)                      | GPT API 토큰 비용 발생                      |
| **유저 추천**    | 첫 좋아요 곡 1개로 유사곡 검색        | 좋아요 곡 **최대 5개를 종합**하여 취향 분석 |
| **곡 기반 추천** | 곡의 오디오 벡터 직접 비교            | 곡의 제목+가수+장르를 GPT에 설명 후 추천    |

### 핵심 차이를 예시로 보면

```
사용자: "비 오는 날 카페에서 듣기 좋은 음악"

faiss_cbf의 search_by_query:
  → "비", "카페", "음악" 단어가 제목/가수에 포함된 곡만 찾음
  → 대부분 결과 없음 ❌

llm_inference의 search_by_query:
  → GPT가 "비 오는 날 카페" = 잔잔, 재즈, 어쿠스틱, 느린 템포로 해석
  → bpm=80, energy=0.03, spectral_centroid=1400 벡터 생성
  → FAISS에서 이 벡터와 가까운 곡 검색
  → 실제로 잔잔한 곡들이 추천됨 ✅
```

---

## 3. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        웹 UI (static/)                              │
│  사용자 입력: "새벽에 공부할 때 듣기 좋은 잔잔한 피아노 음악"          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ POST /api/search
                            │ { query: "...", algorithm: "llm_inference" }
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI (api/main.py)                           │
│                                                                     │
│  search_by_query(req) → recommenders["llm_inference"]               │
│                           .search_by_query(query, top_k)            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LLMInferenceRecommender (llm_inference.py)             │
│                                                                     │
│  _run_graph(query, top_k)                                          │
│       │                                                             │
│       ▼  LangGraph 워크플로우 실행                                   │
│  ┌──────────────────────────────────────────┐                       │
│  │  Node 1: generate_vector                 │                       │
│  │                                          │                       │
│  │  ChatPromptTemplate (시스템 + 사용자)     │                       │
│  │       │                                  │                       │
│  │       ▼                                  │                       │
│  │  ChatOpenAI (gpt-4o-mini)                │ ← OpenAI API 호출     │
│  │       │                                  │                       │
│  │       ▼                                  │                       │
│  │  JsonOutputParser                        │                       │
│  │  (AudioFeatureVector 스키마 강제)         │                       │
│  │       │                                  │                       │
│  │       ▼                                  │                       │
│  │  17차원 벡터: [mfcc_1, ..., zcr]         │                       │
│  └──────────────────┬───────────────────────┘                       │
│                     │                                               │
│  ┌──────────────────▼───────────────────────┐                       │
│  │  Node 2: search_faiss                    │                       │
│  │                                          │                       │
│  │  벡터 → np.float32 변환                   │                       │
│  │       │                                  │                       │
│  │       ▼                                  │                       │
│  │  MusicFaissIndex.search_by_vector()      │ ← L2 정규화 후 검색   │
│  │       │                                  │                       │
│  │       ▼                                  │                       │
│  │  {song_id: cosine_score} (상위 50곡)     │                       │
│  └──────────────────┬───────────────────────┘                       │
│                     │                                               │
│                     ▼                                               │
│  상위 top_k개만 잘라서 반환                                           │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ {song_id: score}
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  api/main.py → _format() → 곡 메타데이터 붙여서 JSON 응답            │
│  { algorithm: "llm_inference", recommendations: [{...}, ...] }      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. LangGraph 워크플로우 상세

### 4.1 왜 LangGraph를 쓰는가?

단순히 GPT를 호출하는 것이라면 LangChain 체인만으로 충분합니다. 하지만 LangGraph를 사용하는 이유:

1. **노드 단위 분리**: `벡터 생성`과 `FAISS 검색`을 독립된 노드로 관리 → 각 단계를 개별 디버깅·교체 가능
2. **상태 관리**: `GraphState` TypedDict로 워크플로우 전체 상태를 명시적으로 추적
3. **확장성**: 향후 노드 추가가 용이 (예: `validate_vector`, `rerank`, `explain_result` 등)
4. **에러 격리**: 한 노드가 실패해도 상태에 `error`를 기록하고 다음 노드에서 처리

### 4.2 GraphState (워크플로우 상태)

```python
class GraphState(TypedDict):
    query: str                        # 사용자 자연어 입력
    feature_stats: str                # DB 오디오 특성 통계 (프롬프트에 주입)
    vector: list[float] | None       # GPT가 생성한 17차원 벡터
    results: dict[str, float] | None # FAISS 검색 결과
    error: str | None                 # 에러 메시지
```

각 노드는 이 상태를 읽고, 자신이 담당하는 필드만 업데이트합니다:

```
초기 상태:  { query: "잔잔한 음악", feature_stats: "...", vector: None, results: None, error: None }
    │
    ▼ Node 1: generate_vector
    { ..., vector: [-8.2, 3.1, ..., 72.0, 0.02, 1100, 0.04], error: None }
    │
    ▼ Node 2: search_faiss
    { ..., results: {"abc123": 0.95, "def456": 0.88, ...}, error: None }
```

### 4.3 워크플로우 그래프 구조

```python
workflow = StateGraph(GraphState)

workflow.add_node("generate_vector", self._node_generate_vector)
workflow.add_node("search_faiss", self._node_search_faiss)

workflow.set_entry_point("generate_vector")        # 시작 노드
workflow.add_edge("generate_vector", "search_faiss") # 순차 연결
workflow.add_edge("search_faiss", END)              # 종료

self._graph = workflow.compile()  # 실행 가능한 그래프로 컴파일
```

시각화:

```
[START] → [generate_vector] → [search_faiss] → [END]
```

---

## 5. 코드 구조 분석

### 5.1 파일 전체 구성

```python
# algorithms/llm_inference.py

# ── 임포트 ──
from langchain_openai import ChatOpenAI          # GPT API 클라이언트
from langchain_core.prompts import ChatPromptTemplate  # 프롬프트 템플릿
from langchain_core.output_parsers import JsonOutputParser  # JSON 파싱
from pydantic import BaseModel, Field            # 출력 스키마 정의
from langgraph.graph import StateGraph, END      # 워크플로우 그래프

# ── 스키마 정의 ──
class AudioFeatureVector(PydanticBaseModel):      # GPT 반환 형식 강제
    mfcc_1: float = Field(description="...")
    ...
    zcr: float = Field(description="...")

# ── LangGraph 상태 ──
class GraphState(TypedDict):                      # 워크플로우 상태 구조

# ── 프롬프트 ──
SYSTEM_PROMPT = """..."""                         # GPT 시스템 프롬프트
HUMAN_PROMPT = """..."""                          # 사용자 질의 템플릿

# ── 메인 클래스 ──
class LLMInferenceRecommender(BaseRecommender):   # 추천기 본체
    fit()                    # 데이터 로드 + 통계 계산 + LangGraph 빌드
    _build_graph()           # LangGraph 워크플로우 구성
    _node_generate_vector()  # Node 1: GPT → 벡터
    _node_search_faiss()     # Node 2: FAISS 검색
    _run_graph()             # 워크플로우 실행
    recommend()              # 곡 기반 추천 (BaseRecommender 인터페이스)
    recommend_for_user()     # 유저 기반 추천
    search_by_query()        # 자연어 검색 (핵심 기능)
```

### 5.2 클래스 초기화

```python
class LLMInferenceRecommender(BaseRecommender):

    def __init__(self, faiss_index, model="gpt-4o-mini", temperature=0.3):
        super().__init__("llm_inference")  # 알고리즘 이름 등록
        self._index = faiss_index           # FAISS 인덱스 (검색에 사용)
        self._data = None                   # 곡 DataFrame (메타데이터 참조)
        self._feature_stats = ""            # DB 통계 문자열 (프롬프트에 주입)
        self._model = model                 # GPT 모델명
        self._temperature = temperature     # GPT 응답 다양성 (0=결정적, 1=창의적)
        self._graph = None                  # 컴파일된 LangGraph 워크플로우
        self._llm = None                    # ChatOpenAI 인스턴스
```

**`temperature=0.3` 인 이유:**

- 너무 낮으면(0) 매번 같은 벡터 → 추천 다양성 부족
- 너무 높으면(1) 수치가 랜덤해져서 이상한 추천
- 0.3은 적당히 일관되면서도 약간의 변동을 주는 값

### 5.3 `fit()` — 데이터 로드 + 그래프 빌드

```python
def fit(self, data: pd.DataFrame) -> None:
    self._data = data
    self._feature_stats = self._compute_feature_stats(data)  # ①
    self._build_graph()                                       # ②
    self.is_fitted = True
```

**① `_compute_feature_stats()`가 하는 일:**

DB에 저장된 모든 곡의 오디오 특성 통계를 계산합니다. 이 통계는 GPT 시스템 프롬프트에 주입되어, GPT가 **현실적인 범위 안에서** 벡터를 생성하도록 유도합니다.

```
예시 출력:
  mfcc_1: min=-423.1234, max=127.5678, mean=-102.3456, std=89.1234
  mfcc_2: min=-78.9012, max=156.7890, mean=56.7890, std=34.5678
  ...
  bpm: min=60.1234, max=199.5678, mean=117.8901, std=28.4567
  energy: min=0.0012, max=0.2345, mean=0.0567, std=0.0345
  spectral_centroid: min=456.7890, max=6789.0123, mean=2345.6789, std=1023.4567
  zcr: min=0.0123, max=0.2567, mean=0.0890, std=0.0456
```

**왜 통계를 주입하는가?**
GPT는 오디오 분석 데이터의 실제 분포를 모릅니다. 우리 DB의 곡들이 BPM 60~200 범위인지, energy가 0.001~0.3 범위인지 알려줘야 **검색에 의미 있는 벡터**를 생성할 수 있습니다.

**② `_build_graph()`가 하는 일:**

```python
def _build_graph(self) -> None:
    from utils.config import OPENAI_API_KEY

    if not OPENAI_API_KEY:           # API 키 없으면 조기 종료
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    self._llm = ChatOpenAI(          # LangChain GPT 클라이언트
        model=self._model,
        temperature=self._temperature,
        api_key=OPENAI_API_KEY,
    )

    workflow = StateGraph(GraphState)
    workflow.add_node("generate_vector", self._node_generate_vector)
    workflow.add_node("search_faiss", self._node_search_faiss)
    workflow.set_entry_point("generate_vector")
    workflow.add_edge("generate_vector", "search_faiss")
    workflow.add_edge("search_faiss", END)

    self._graph = workflow.compile()  # 실행 가능한 그래프로 컴파일
```

### 5.4 Node 1 — `_node_generate_vector()` (GPT 벡터 생성)

이 노드가 이 알고리즘의 **핵심**입니다.

```python
def _node_generate_vector(self, state: GraphState) -> dict[str, Any]:
    query = state["query"]              # "새벽에 공부할 때 좋은 잔잔한 음악"
    feature_stats = state["feature_stats"]  # DB 통계 문자열

    # ① 프롬프트 조합
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),      # 음악 오디오 전문가 역할 + DB 통계
        ("human", HUMAN_PROMPT),        # 사용자 요청
    ])

    # ② JSON 파서 (AudioFeatureVector 스키마로 출력 강제)
    parser = JsonOutputParser(pydantic_object=AudioFeatureVector)

    # ③ LangChain 체인: 프롬프트 → GPT → JSON 파싱
    chain = prompt | self._llm | parser

    # ④ 실행
    result = chain.invoke({
        "query": query,
        "feature_stats": feature_stats,
    })
    # result = {"mfcc_1": -8.2, "mfcc_2": 3.1, ..., "bpm": 72, "zcr": 0.04}

    # ⑤ FEATURE_COLS 순서대로 벡터 변환
    vector = [float(result.get(col, 0.0)) for col in FEATURE_COLS]
    # vector = [-8.2, 3.1, ..., 72.0, 0.02, 1100.0, 0.04]

    return {"vector": vector, "error": None}
```

**LangChain 체인(`|`) 파이프라인 동작:**

```
prompt.invoke({"query": "...", "feature_stats": "..."})
    → ChatPromptValue (시스템 + 사용자 메시지)
        │
        ▼
self._llm.invoke(messages)
    → AIMessage (GPT의 JSON 텍스트 응답)
        │
        ▼
parser.invoke(ai_message)
    → dict (파이썬 딕셔너리로 파싱됨)
```

### 5.5 Node 2 — `_node_search_faiss()` (FAISS 유사도 검색)

```python
def _node_search_faiss(self, state: GraphState) -> dict[str, Any]:
    vector = state.get("vector")
    if vector is None:
        return {"results": {}, "error": "벡터 생성 실패"}

    # numpy 배열로 변환
    vec_np = np.array(vector, dtype=np.float32)

    # MusicFaissIndex.search_by_vector() 호출
    # 내부에서 L2 정규화 → IndexFlatIP 검색 (코사인 유사도)
    results = self._index.search_by_vector(vec_np, top_k=50)

    return {"results": results, "error": None}
```

**왜 50개를 검색할까?**
`search_by_query`에서 `top_k=3`으로 요청해도, 내부적으로 50개를 먼저 가져온 뒤 `_run_graph`에서 top_k로 자릅니다. 이렇게 하면 `recommend()`나 `recommend_for_user()`에서 자기 자신이나 좋아요 곡을 제외한 후에도 충분한 결과를 확보할 수 있습니다.

### 5.6 세 가지 추천 모드

#### 모드 1: `search_by_query()` — 자연어 검색 (핵심)

```python
def search_by_query(self, query: str, top_k: int = 10):
    return self._run_graph(query.strip(), top_k)
```

가장 단순합니다. 사용자 자연어를 그대로 LangGraph에 넘깁니다.

```
입력: "비 오는 날 카페에서 듣기 좋은 재즈"
동작: GPT → 재즈 특성 벡터 → FAISS 검색
출력: {song_id: score} (재즈/로파이 계열 곡)
```

#### 모드 2: `recommend()` — 곡 기반 추천

```python
def recommend(self, song_id: str, top_k: int = 10):
    # song_id에서 메타데이터 추출
    row = self._data[self._data["song_id"] == song_id]
    r = row.iloc[0]
    title, artist, genre = r["title"], r["artist"], r["genre"]

    # 메타데이터 → 자연어 질의로 변환
    query = f"'{title}' by {artist} ({genre}) 와 비슷한 분위기의 음악"

    results = self._run_graph(query, top_k + 1)
    results.pop(song_id, None)  # 자기 자신 제외
    return dict(sorted(results.items(), ...)[:top_k])
```

```
입력: song_id = "abc123" (IU - 밤편지 - Ballad)
변환: "'밤편지' by IU (Ballad) 와 비슷한 분위기의 음악"
동작: GPT → 발라드 특성 벡터 → FAISS 검색
출력: 비슷한 분위기의 발라드 곡들
```

**`faiss_cbf`와의 차이:**

- `faiss_cbf`: 밤편지의 오디오 수치(MFCC, BPM 등)를 직접 비교
- `llm_inference`: "밤편지 같은 분위기"를 GPT가 해석 → 더 넓은 의미의 유사곡 발견 가능

#### 모드 3: `recommend_for_user()` — 유저 기반 추천

```python
def recommend_for_user(self, user_id: int, top_k: int = 10):
    likes = get_user_liked_song_ids(user_id)

    # 좋아요한 곡 최대 5개의 정보를 종합
    liked_descriptions = []
    for sid in likes[:5]:
        row = self._data[self._data["song_id"] == sid]
        liked_descriptions.append(f"'{title}' - {artist}")

    songs_text = ", ".join(liked_descriptions)
    query = f"다음 곡들을 좋아하는 사용자의 취향에 맞는 음악: {songs_text}"

    results = self._run_graph(query, top_k + len(likes))
    for sid in likes:
        results.pop(sid, None)  # 이미 좋아요한 곡 제외
    return dict(sorted(results.items(), ...)[:top_k])
```

```
입력: user_id = 42 (좋아요: 밤편지, 봄날, Blinding Lights)
변환: "다음 곡들을 좋아하는 사용자의 취향에 맞는 음악: '밤편지' - IU, '봄날' - BTS, 'Blinding Lights' - The Weeknd"
동작: GPT → 이 3곡의 공통 분위기를 종합한 벡터 → FAISS 검색
출력: 세 곡의 교집합 느낌에 맞는 추천
```

**`faiss_cbf`와의 차이:**

- `faiss_cbf`: 좋아요 곡 중 **첫 1개**만 사용
- `llm_inference`: 좋아요 곡 **최대 5개를 종합**하여 전체 취향 파악

---

## 6. 프로젝트 연동 구조

### 6.1 변경된 파일 목록

| 파일                          | 변경 내용                                           |
| ----------------------------- | --------------------------------------------------- |
| `algorithms/llm_inference.py` | 🆕 신규 생성 — LLM 추천 알고리즘 전체               |
| `utils/config.py`             | `OPENAI_API_KEY` 환경변수 추가 (1줄)                |
| `api/main.py`                 | `startup()`에 `llm_inference` 등록 블록 추가 (~8줄) |
| `requirements.txt`            | `langchain-openai>=0.1.0` 추가 (1줄)                |

### 6.2 `api/main.py` 등록 코드

```python
# api/main.py → startup() 함수 안

# ── LLM Inference 추천기 (GPT + LangGraph → FAISS 검색) ──
if faiss_index is not None and getattr(faiss_index, "is_built", False):
    try:
        from algorithms.llm_inference import LLMInferenceRecommender

        recommenders["llm_inference"] = LLMInferenceRecommender(faiss_index)
        recommenders["llm_inference"].fit(song_df)
        print("[API] LLM 알고리즘 등록: llm_inference")
    except Exception as e:
        print(f"[API] LLM 추천기 등록 실패 (OPENAI_API_KEY 확인): {e}")
```

**등록 조건:**

- FAISS 인덱스가 정상 구축되어 있어야 함 (`is_built = True`)
- `OPENAI_API_KEY`가 없어도 등록 자체는 시도하지만, `_build_graph()`에서 경고 후 `_graph = None`
- 이후 `search_by_query()` 호출 시 `_graph is None`이면 빈 결과 반환 (서버 크래시 없음)

### 6.3 의존성 관계 다이어그램

```
utils/config.py
    │ OPENAI_API_KEY
    ▼
algorithms/llm_inference.py
    │
    ├── algorithms/base.py         ← BaseRecommender 상속
    ├── data/faiss_index.py        ← FEATURE_COLS, MusicFaissIndex.search_by_vector()
    ├── data/database.py           ← get_user_liked_song_ids()
    ├── langchain_openai           ← ChatOpenAI
    ├── langchain_core             ← ChatPromptTemplate, JsonOutputParser
    ├── pydantic                   ← AudioFeatureVector 스키마
    └── langgraph                  ← StateGraph, END
    │
    ▼
api/main.py
    │ recommenders["llm_inference"] = LLMInferenceRecommender(faiss_index)
    ▼
웹 UI (static/app.js)
    │ 알고리즘 셀렉트에서 "llm_inference" 선택 가능
```

---

## 7. 환경 설정 및 실행 방법

### Step 1: OpenAI API 키 발급

1. https://platform.openai.com 접속 → 로그인
2. API Keys 메뉴 → "Create new secret key"
3. 생성된 키 복사 (`sk-...` 형태)

### Step 2: `.env` 파일 설정

프로젝트 루트의 `.env` 파일에 추가:

```dotenv
# 기존 설정 유지
DB_PATH=./music_rec.db
JWT_SECRET=your-secret-key
JWT_EXPIRE_HOURS=24

# 🆕 OpenAI API 키 추가
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
```

### Step 3: 패키지 설치

```bash
pip install langchain-openai
```

> `langchain`, `langgraph`, `faiss-cpu`는 이미 `requirements.txt`에 포함되어 있습니다.

### Step 4: 서버 실행

```bash
uvicorn api.main:app --reload
```

정상 등록 시 콘솔 출력:

```
[DB] 초기화 완료: ./music_rec.db
[API] 352곡 로드 완료
[FAISS] 캐시 로드: 352곡
[API] 샘플 알고리즘 등록: faiss_cbf
[LLM-Inference] fit 완료 (곡 수: 352, 모델: gpt-4o-mini)
[LLM-Inference] LangGraph 워크플로우 빌드 완료
[API] LLM 알고리즘 등록: llm_inference
```

### Step 5: 웹 UI에서 사용

1. `http://localhost:8000` 접속 → 로그인
2. 사이드바 "알고리즘" 셀렉트에서 `llm_inference` 선택
3. 검색바에 자연어 입력: "신나는 여름 댄스 음악"
4. 추천 결과 확인

---

## 8. 사용자 시나리오별 동작 흐름

### 시나리오 1: 자연어 검색

```
[사용자] 검색바에 "비 오는 날 카페에서 듣기 좋은 재즈" 입력
    │
    ▼ POST /api/search
[FastAPI] search_by_query() → recommenders["llm_inference"].search_by_query()
    │
    ▼ _run_graph("비 오는 날 카페에서 듣기 좋은 재즈", top_k=3)
[Node 1] GPT에게 프롬프트 전송:
    시스템: "당신은 음악 오디오 분석 전문가입니다..."
    사용자: "사용자 요청: 비 오는 날 카페에서 듣기 좋은 재즈"
    │
    ▼ GPT 응답 (JSON):
    {
      "mfcc_1": -95.5, "mfcc_2": 45.2, ...,
      "bpm": 85.0,
      "energy": 0.035,
      "spectral_centroid": 1800.0,
      "zcr": 0.065
    }
    │
    ▼ FEATURE_COLS 순서 벡터: [-95.5, 45.2, ..., 85.0, 0.035, 1800.0, 0.065]
    │
[Node 2] FAISS search_by_vector(벡터, top_k=50)
    │ L2 정규화 → IndexFlatIP 내적 검색
    ▼
    {"jazzSong1": 0.92, "chillSong2": 0.88, "lofiSong3": 0.85, ...}
    │
    ▼ 상위 3개만 잘라서 반환
[_format()] 곡 메타데이터 결합 → JSON 응답
[웹 UI] 카드 3개 렌더링 (썸네일, 제목, 가수, 유사도 점수)
```

### 시나리오 2: 곡 클릭 → 유사곡 추천

```
[사용자] 곡 카드 클릭 (IU - 밤편지)
    │
    ▼ POST /api/recommend { song_id: "abc123", algorithm: "llm_inference" }
[recommend()] song_id → 메타데이터 조회
    query = "'밤편지' by IU (Ballad) 와 비슷한 분위기의 음악"
    │
    ▼ _run_graph(query, top_k=4)  // top_k+1 (자기 자신 제외용)
    ... (Node 1 → Node 2 동일 흐름)
    │
    ▼ results.pop("abc123")  // 자기 자신 제외
    ▼ 상위 3개 반환
```

### 시나리오 3: "내 취향 추천 받기" 버튼

```
[사용자] "내 취향 추천 받기" 클릭
    │
    ▼ POST /api/recommend/user { algorithm: "llm_inference" }
[recommend_for_user()] DB에서 좋아요 곡 조회
    likes = ["abc123", "def456", "ghi789"]
    │
    ▼ 최대 5곡의 제목+가수 종합
    query = "다음 곡들을 좋아하는 사용자의 취향에 맞는 음악:
             '밤편지' - IU, '봄날' - BTS, 'Dynamite' - BTS"
    │
    ▼ _run_graph(query, top_k=6)  // top_k + len(likes)
    ... (Node 1 → Node 2 동일 흐름)
    │
    ▼ 좋아요한 3곡 제외 → 상위 3개 반환
```

---

## 9. 핵심 기술 개념 설명

### 9.1 LangChain이란?

> LLM(대형 언어 모델)을 애플리케이션에 쉽게 통합하기 위한 프레임워크

이 프로젝트에서 사용하는 LangChain 컴포넌트:

| 컴포넌트             | 역할                        | 사용 위치                 |
| -------------------- | --------------------------- | ------------------------- |
| `ChatOpenAI`         | OpenAI GPT API 클라이언트   | `_build_graph()`          |
| `ChatPromptTemplate` | 시스템/사용자 메시지 템플릿 | `_node_generate_vector()` |
| `JsonOutputParser`   | GPT 응답을 JSON dict로 파싱 | `_node_generate_vector()` |

**LangChain 체인 파이프라인(`|` 연산자):**

```python
chain = prompt | self._llm | parser
result = chain.invoke({"query": "...", "feature_stats": "..."})
```

`prompt → LLM → parser` 순서로 데이터가 흐릅니다. 유닉스 파이프(`|`)와 같은 개념입니다.

### 9.2 LangGraph란?

> LangChain 위에 구축된 **상태 기반 워크플로우 프레임워크**

LangChain 체인은 단방향(linear)이지만, LangGraph는 **분기, 루프, 조건부 실행**이 가능합니다.

```
LangChain 체인:   A → B → C  (일직선)

LangGraph 그래프: A → B → C  (현재 사용)
                  A → B ─┬→ C  (조건 분기 가능)
                         └→ D
                  A → B → A  (루프도 가능)
```

현재는 선형 구조(`generate_vector → search_faiss`)이지만, LangGraph를 사용한 덕분에 향후 확장이 용이합니다.

### 9.3 Pydantic Structured Output

```python
class AudioFeatureVector(PydanticBaseModel):
    mfcc_1: float = Field(description="MFCC 1번째 계수 (보통 -30 ~ 30)")
    ...
    zcr: float = Field(description="제로 크로싱 레이트. 클린/보컬=0.03~0.07")
```

이 스키마를 `JsonOutputParser`에 전달하면:

1. GPT에게 "이 형태의 JSON으로 응답하라"는 지시가 프롬프트에 자동 추가됨
2. GPT 응답을 파싱할 때 필드 누락이나 타입 오류를 잡아줌
3. 결과가 항상 `dict[str, float]` 형태로 보장됨

### 9.4 FAISS `search_by_vector()` 동작 원리

```python
# data/faiss_index.py의 search_by_vector()

def search_by_vector(self, vector, top_k=10):
    vec = vector.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)          # ① L2 정규화 (벡터 길이 = 1)

    scores, indices = self.index.search(vec, top_k)  # ② FAISS 검색

    results = {}
    for score, i in zip(scores[0], indices[0]):
        results[self.id_map[i]] = float(score)  # ③ 점수 매핑

    return results
```

**① L2 정규화:**

```
원본 벡터: [-8.2, 3.1, ..., 72.0, 0.02, 1100.0, 0.04]
정규화 후: [-0.007, 0.003, ..., 0.065, 0.00002, 0.998, 0.00004]
(벡터 길이가 1이 됨 → 내적 = 코사인 유사도)
```

**② 코사인 유사도 의미:**

- 1.0 = 완벽히 같은 방향 (동일한 음악적 특성)
- 0.0 = 직교 (관련 없음)
- -1.0 = 정반대 방향

---

## 10. 프롬프트 엔지니어링 설계

### 10.1 시스템 프롬프트 구조

```
┌─────────────────────────────────────────────────────┐
│  역할 부여: "당신은 음악 오디오 분석 전문가입니다"     │
├─────────────────────────────────────────────────────┤
│  벡터 구성 설명 (17차원 각각의 의미)                  │
│  - mfcc_1~13: 음색 (각 계수의 역할)                  │
│  - bpm: 템포 (숫자 범위)                              │
│  - energy: 에너지 (숫자 범위)                         │
│  - spectral_centroid: 밝기 (숫자 범위)                │
│  - zcr: 소리 질감 (숫자 범위)                         │
├─────────────────────────────────────────────────────┤
│  DB 실제 통계 (동적 주입)                             │
│  "{feature_stats}"  ← fit() 시 계산된 min/max/mean   │
├─────────────────────────────────────────────────────┤
│  장르별 특성 가이드라인 (9개 장르)                     │
│  - 발라드: 낮은 bpm, 낮은 energy, ...                │
│  - K-POP: 중간~높은 bpm, 중간 energy, ...            │
│  - EDM: 높은 bpm, 높은 energy, ...                   │
│  - ...                                               │
├─────────────────────────────────────────────────────┤
│  출력 규칙                                            │
│  - 반드시 JSON 형식                                   │
│  - 17개 필드 모두 포함                                │
│  - DB 통계 범위 안에서 값 설정                        │
└─────────────────────────────────────────────────────┘
```

### 10.2 왜 이렇게 설계했는가

| 설계 결정         | 이유                                                                 |
| ----------------- | -------------------------------------------------------------------- |
| DB 통계 주입      | GPT가 우리 데이터의 실제 수치 범위를 모르면 비현실적인 벡터를 생성함 |
| 장르별 가이드라인 | "잔잔한"이 bpm=60~80이라는 매핑을 GPT가 일관되게 수행하도록 유도     |
| MFCC 범위 명시    | MFCC는 일반인이 이해하기 어렵기에, GPT에게도 범위를 제시해야 함      |
| JSON 출력 강제    | `JsonOutputParser` + 프롬프트 규칙으로 이중 안전장치                 |
| temperature=0.3   | 수치 생성 작업이므로 너무 창의적이면 안 됨. 약간의 변동만 허용       |

### 10.3 프롬프트 예시 (실제 GPT에게 전달되는 내용)

```
[System]
당신은 음악 오디오 분석 전문가입니다.
사용자가 원하는 음악의 분위기, 감정, 상황을 설명하면,
해당 음악의 오디오 특성을 17차원 수치 벡터로 추정해야 합니다.

## 벡터 구성 (총 17차원)
1. mfcc_1 ~ mfcc_13: Mel-Frequency Cepstral Coefficients
   ...
## 현재 DB의 오디오 특성 통계 (참고용)
  mfcc_1: min=-423.1234, max=127.5678, mean=-102.3456, std=89.1234
  ...
  bpm: min=60.1234, max=199.5678, mean=117.8901, std=28.4567
  ...
## 장르별 특성 가이드라인
- 잔잔한 발라드/피아노: 낮은 bpm(60-85), ...
  ...

[Human]
사용자 요청: 새벽에 공부할 때 듣기 좋은 잔잔한 피아노 음악

위 요청에 맞는 음악의 오디오 특성 벡터를 JSON으로 반환해주세요.
```

---

## 11. 한계점 및 개선 방향

### 현재 한계점

| 한계               | 설명                                                                          |
| ------------------ | ----------------------------------------------------------------------------- |
| **속도**           | GPT API 호출로 1~3초 지연. faiss_cbf는 밀리초 단위                            |
| **비용**           | 요청당 ~100 토큰 소비. gpt-4o-mini 기준 매우 저렴하지만 0은 아님              |
| **MFCC 추정 한계** | GPT가 MFCC 값을 정확히 추정하기 어려움. bpm/energy/centroid/zcr는 비교적 정확 |
| **오프라인 불가**  | 인터넷 연결 + OpenAI API 접근 필요                                            |
| **캐싱 없음**      | 동일 질의를 반복해도 매번 GPT 호출                                            |

### 개선 방향

#### 1. 벡터 캐싱 추가

```python
# 같은 질의에 대해 캐시된 벡터 반환
self._vector_cache = {}  # {query_hash: vector}

def _node_generate_vector(self, state):
    cache_key = hashlib.md5(state["query"].encode()).hexdigest()
    if cache_key in self._vector_cache:
        return {"vector": self._vector_cache[cache_key]}
    ...
    self._vector_cache[cache_key] = vector
```

#### 2. LangGraph 노드 확장

```
현재: generate_vector → search_faiss

개선안:
  generate_vector → validate_vector → search_faiss → rerank → explain
                         │
                         └── (범위 초과 시) → clamp_vector → search_faiss
```

- **validate_vector**: 생성된 벡터가 DB 통계 범위 내인지 검증
- **rerank**: GPT에게 검색 결과를 다시 보여주고 최종 순위 조정
- **explain**: "이 곡을 추천한 이유" 자연어 설명 생성

#### 3. MFCC 의존도 줄이기

```python
# MFCC 가중치를 낮추고 bpm/energy/centroid/zcr 가중치를 높이는 방식
weighted_vector = vector * weights  # weights: MFCC는 0.5, 나머지는 2.0
```

#### 4. Few-shot 프롬프트

```python
# DB에서 실제 곡 예시를 프롬프트에 포함
examples = """
예시 1: "아이유 - 밤편지" (Ballad)
→ {"mfcc_1": -105.3, ..., "bpm": 72, "energy": 0.025, "spectral_centroid": 1200, "zcr": 0.045}

예시 2: "BTS - Dynamite" (K-POP)
→ {"mfcc_1": -85.7, ..., "bpm": 114, "energy": 0.065, "spectral_centroid": 2800, "zcr": 0.09}
"""
```

---

## 12. 트러블슈팅 FAQ

### Q1: 서버 시작 시 "OPENAI_API_KEY가 설정되지 않았습니다" 경고

```
[LLM-Inference] ⚠️ OPENAI_API_KEY가 설정되지 않았습니다.
[LLM-Inference]    .env 파일에 OPENAI_API_KEY=sk-... 를 추가하세요.
```

**해결**: `.env` 파일에 `OPENAI_API_KEY=sk-...` 추가 후 서버 재시작

### Q2: 알고리즘 셀렉트에 `llm_inference`가 안 나와요

**원인 1**: FAISS 인덱스가 구축되지 않음 (audio_features 데이터 없음)
→ `python data/melon_pipeline.py`로 데이터 수집 먼저

**원인 2**: `langchain-openai` 패키지 미설치
→ `pip install langchain-openai`

**원인 3**: `.env`의 `OPENAI_API_KEY`가 비어 있음
→ API 키 설정 확인

### Q3: 검색하면 결과가 없어요

**원인 1**: GPT API 호출 실패 (네트워크 오류, API 키 만료 등)
→ 서버 콘솔에서 `[LLM-Inference] GPT 벡터 생성 실패:` 로그 확인

**원인 2**: FAISS 인덱스에 곡이 적음
→ 데이터 수집 파이프라인으로 더 많은 곡 확보

### Q4: 추천 결과가 이상해요 (관련 없는 곡이 나옴)

**가능한 원인:**

- GPT가 생성한 MFCC 값이 비현실적 → temperature를 0.1로 낮추기
- DB에 해당 분위기의 곡이 없음 → 데이터 다양성 확보 필요
- 프롬프트 개선 필요 → SYSTEM_PROMPT에 더 구체적인 가이드 추가

### Q5: API 비용이 걱정돼요

**현재 비용 추정 (gpt-4o-mini 기준):**

- 요청당 ~1,000 토큰 (프롬프트 ~900 + 응답 ~100)
- gpt-4o-mini: $0.15 / 1M input tokens
- 1,000회 검색 ≈ $0.15 (매우 저렴)

**비용 절감 방법:**

- 벡터 캐싱 구현 (동일 질의 재사용)
- temperature=0으로 설정 시 동일 질의 → 동일 결과 (캐싱 효과 극대화)

### Q6: `gpt-4o-mini` 대신 다른 모델을 쓰고 싶어요

```python
# api/main.py → startup()에서 model 파라미터 변경
recommenders["llm_inference"] = LLMInferenceRecommender(
    faiss_index,
    model="gpt-4o",         # 더 정확하지만 비용 높음
    temperature=0.2,
)
```

사용 가능한 모델:

- `gpt-4o-mini`: 빠르고 저렴 (기본값, 추천)
- `gpt-4o`: 더 정확하지만 ~10배 비용
- `gpt-3.5-turbo`: 가장 저렴하지만 정확도 낮음

---

> 📝 이 알고리즘은 "자연어 → 음악 추천"이라는 **새로운 UX**를 제공합니다.
> 기존 `faiss_cbf`가 "이 곡과 비슷한 곡"을 찾는다면,
> `llm_inference`는 "이런 **느낌**의 곡"을 찾아줍니다. 🎵
