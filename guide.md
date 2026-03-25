# 🎵 Music-Rec 프로젝트 코드 리뷰 가이드

> **작성 목적**: 신입 개발자가 프로젝트 전체를 빠르게 이해하고, 코드를 수정·확장할 수 있도록 돕는 종합 가이드입니다.
>
> **프로젝트 한줄 요약**: 멜론 차트에서 곡을 크롤링하고, YouTube에서 오디오를 받아 분석한 뒤, FAISS 벡터 검색으로 유사 곡을 추천하는 **AI 음악 추천 시스템**입니다.

---

## 📖 목차

1. [프로젝트 전체 구조](#1-프로젝트-전체-구조)
2. [데이터 흐름 한눈에 보기](#2-데이터-흐름-한눈에-보기)
3. [폴더별 상세 분석](#3-폴더별-상세-분석)
   - [utils/ — 설정 관리](#31-utils--설정-관리)
   - [data/ — 데이터 수집·저장·인덱싱](#32-data--데이터-수집저장인덱싱)
   - [algorithms/ — 추천 알고리즘](#33-algorithms--추천-알고리즘)
   - [evaluation/ — 추천 성능 평가](#34-evaluation--추천-성능-평가)
   - [api/ — FastAPI 백엔드](#35-api--fastapi-백엔드)
   - [static/ — 프론트엔드 (웹 UI)](#36-static--프론트엔드-웹-ui)
4. [핵심 기술 개념 설명](#4-핵심-기술-개념-설명)
5. [실행 순서 가이드](#5-실행-순서-가이드)
6. [새 추천 알고리즘 추가하는 방법](#6-새-추천-알고리즘-추가하는-방법)
7. [코드 품질 리뷰 및 개선 포인트](#7-코드-품질-리뷰-및-개선-포인트)
8. [자주 묻는 질문 (FAQ)](#8-자주-묻는-질문-faq)

---

## 1. 프로젝트 전체 구조

```
music-rec/
├── api/
│   └── main.py                 # FastAPI 서버 (인증, 추천, 검색, 인터랙션 API)
├── data/
│   ├── melon_pipeline.py       # 멜론 크롤링 → YouTube 매칭 → 오디오 특성 추출
│   ├── database.py             # SQLite DB 초기화/CRUD (곡, 유저, 인터랙션)
│   └── faiss_index.py          # FAISS 벡터 인덱스 생성 및 유사도 검색
├── algorithms/
│   ├── base.py                 # 추천기 추상 클래스 (인터페이스 정의)
│   └── faiss_cbf.py            # FAISS 기반 콘텐츠 필터링 추천기 (샘플 구현)
├── evaluation/
│   └── metrics.py              # 추천 성능 평가 지표 (Precision, Recall, NDCG 등)
├── static/
│   ├── index.html              # 웹 UI HTML
│   ├── style.css               # 스타일시트 (Spotify 스타일 다크 테마)
│   └── app.js                  # 프론트엔드 JavaScript (API 통신, UI 렌더링)
├── utils/
│   └── config.py               # 환경변수 로드, 전역 설정값 관리
├── requirements.txt            # Python 패키지 의존성 목록
└── README.md                   # 프로젝트 소개 및 설치 가이드
```

### 각 폴더의 역할 요약

| 폴더          | 역할                           | 비유                                 |
| ------------- | ------------------------------ | ------------------------------------ |
| `utils/`      | 프로젝트 전체 설정값 보관      | 건물의 **설계도면**                  |
| `data/`       | 데이터 수집, 저장, 벡터 인덱싱 | 건물의 **기초 공사 + 자재 창고**     |
| `algorithms/` | 추천 로직 구현                 | 건물의 **핵심 엔진**                 |
| `evaluation/` | 추천 품질 측정                 | 건물의 **품질 검사팀**               |
| `api/`        | HTTP API 서버                  | 건물의 **안내 데스크** (외부와 소통) |
| `static/`     | 사용자가 보는 화면             | 건물의 **로비 인테리어**             |

---

## 2. 데이터 흐름 한눈에 보기

전체 시스템은 아래 순서로 동작합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: 데이터 수집 (melon_pipeline.py 실행)                     │
│                                                                   │
│  멜론 차트 크롤링 ──→ YouTube 검색 ──→ 오디오 다운로드               │
│        │                   │                   │                  │
│        ▼                   ▼                   ▼                  │
│  (title, artist)     (video_id, url)    librosa 오디오 분석        │
│                                          (MFCC, BPM, energy...)   │
│        └──────────────┼──────────────────────┘                    │
│                       ▼                                           │
│              SQLite DB 저장 (songs + audio_features 테이블)        │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: 서버 시작 (uvicorn api.main:app --reload)               │
│                                                                   │
│  DB에서 곡 로드 ──→ FAISS 벡터 인덱스 구축 ──→ 추천기 등록          │
│                                                                   │
│  startup() 함수에서 자동 실행됨                                     │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: 사용자 요청 처리                                         │
│                                                                   │
│  웹 UI (static/) ──HTTP──→ FastAPI (api/main.py)                  │
│       │                            │                              │
│       │  "이 곡과 비슷한 곡 추천해줘"  │                              │
│       │                            ▼                              │
│       │                   algorithms/faiss_cbf.py                 │
│       │                            │                              │
│       │                     FAISS 유사도 검색                      │
│       │                            │                              │
│       ◀────── JSON 응답 ◀──────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 폴더별 상세 분석

### 3.1 `utils/` — 설정 관리

#### `utils/config.py`

**역할**: `.env` 파일에서 환경변수를 읽어오고, 프로젝트 전역에서 사용하는 설정값을 한곳에 모아둡니다.

```python
# 핵심 구조 해설

load_dotenv()  # .env 파일의 환경변수를 os.environ에 로드

# API 키 (외부 서비스 연동용)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")  # 없으면 빈 문자열
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")

# DB & 인증
DB_PATH          = os.getenv("DB_PATH", "./music_rec.db")     # SQLite 파일 위치
JWT_SECRET       = os.getenv("JWT_SECRET", "dev-change-this-secret")  # ⚠️ 배포 시 변경 필수!
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))   # 토큰 만료 시간

# 멜론 크롤링 설정
MELON_CHART_URLS = [...]               # 크롤링할 차트 URL 목록
MELON_MAX_SONGS_PER_CHART = 100        # 차트당 최대 곡 수
MELON_GENRE_CODES = ["GN0100", ...]    # 장르 코드 (한국대중, 해외POP 등)
MELON_MAX_SONGS_PER_GENRE = 300        # 장르당 최대 곡 수

# 추천 설정
TOP_K            = 10    # 추천 결과 개수
COLD_START_LIMIT = 3     # 인터랙션이 이 수보다 적으면 "콜드 스타트" 상태
```

**신입 개발자가 알아야 할 것**:

- `os.getenv("KEY", "기본값")` 패턴: 환경변수가 없으면 기본값을 사용합니다
- 민감한 정보(JWT_SECRET, API_KEY)는 코드에 직접 쓰지 않고 `.env` 파일로 분리합니다
- 이 파일을 수정하면 **크롤링 범위, 추천 개수 등 프로젝트 전체 동작**이 바뀝니다

---

### 3.2 `data/` — 데이터 수집·저장·인덱싱

이 폴더는 프로젝트의 **데이터 계층**입니다. 3개 파일이 각각 다른 역할을 합니다.

#### 3.2.1 `data/database.py` — SQLite DB 관리

**역할**: SQLite 데이터베이스의 테이블 생성, 곡·유저·인터랙션 데이터의 CRUD(생성·읽기·수정·삭제)를 담당합니다.

##### 테이블 구조

```
┌──────────────────┐     ┌───────────────────────┐
│      songs       │     │    audio_features     │
├──────────────────┤     ├───────────────────────┤
│ song_id (PK)     │◀───▶│ song_id (PK, FK)      │
│ title            │     │ mfcc_1 ~ mfcc_13      │  ← 음색(톤) 특성
│ artist           │     │ bpm                   │  ← 빠르기
│ youtube_url      │     │ energy                │  ← 에너지(음량)
│ thumbnail_url    │     │ spectral_centroid     │  ← 밝기(주파수 무게중심)
│ genre            │     │ zcr                   │  ← 제로 크로싱률(소리 질감)
│ duration         │     └───────────────────────┘
│ collected_at     │
└──────────────────┘

┌──────────────────┐     ┌───────────────────────┐
│      users       │     │    interactions       │
├──────────────────┤     ├───────────────────────┤
│ user_id (PK)     │◀────│ user_id (FK)          │
│ username         │     │ song_id (FK)          │──▶ songs.song_id
│ password_hash    │     │ action                │  ← "play"/"like"/"skip"/"unlike"
│ created_at       │     │ play_seconds          │  ← 재생한 초
└──────────────────┘     │ timestamp             │
                         └───────────────────────┘
```

##### 주요 함수 설명

| 함수                                             | 역할                                         | 호출 시점                  |
| ------------------------------------------------ | -------------------------------------------- | -------------------------- |
| `init_db()`                                      | 테이블 생성 (없으면 만들기)                  | 서버 시작, 파이프라인 시작 |
| `upsert_song(song)`                              | 곡 정보 INSERT 또는 UPDATE                   | 데이터 수집 시             |
| `upsert_audio_features(features)`                | 오디오 특성 INSERT 또는 UPDATE               | 데이터 수집 시             |
| `get_all_songs()`                                | songs + audio_features JOIN한 DataFrame 반환 | 서버 시작 시 전체 곡 로드  |
| `get_song(song_id)`                              | 특정 곡 1건 조회                             | API에서 곡 상세 조회       |
| `create_user(username, password)`                | 회원가입 (bcrypt 해시)                       | 회원가입 API               |
| `authenticate_user(username, password)`          | 로그인 검증                                  | 로그인 API                 |
| `log_interaction(user_id, song_id, action, ...)` | 사용자 행동 기록                             | 좋아요/재생/스킵 시        |
| `get_user_liked_song_ids(user_id)`               | 좋아요한 곡 ID 목록 (unlike 고려)            | 추천 시                    |
| `get_user_liked_songs(user_id)`                  | 좋아요한 곡의 메타데이터                     | 좋아요 목록 API            |

##### 비밀번호 처리 흐름 (중요!)

```
회원가입 시:
  평문 비밀번호 → bcrypt.hashpw() → "$2b$..." 형태의 해시 → DB 저장

로그인 시:
  1. DB에서 password_hash 조회
  2. "$2"로 시작? → bcrypt 방식으로 검증 (bcrypt.checkpw)
  3. 아니면? → SHA-256 레거시 방식으로 검증
     └─ 레거시 성공 시 → bcrypt로 자동 업그레이드 (보안 강화)
```

> 💡 **왜 두 가지 방식이 있나요?**
> 초기 버전에서 SHA-256을 썼다가 보안이 약해서 bcrypt로 변경했습니다.
> 기존 사용자를 위해 SHA-256 → bcrypt 자동 마이그레이션 로직이 포함되어 있습니다.

##### `get_user_liked_song_ids()` — Window Function 활용

```sql
-- 곡별로 가장 최근 인터랙션이 'like'인 것만 선택
WITH last AS (
    SELECT song_id, action,
           ROW_NUMBER() OVER (
               PARTITION BY song_id        -- 곡별로 그룹핑
               ORDER BY timestamp DESC     -- 최신 순 정렬
           ) AS rn
    FROM interactions
    WHERE user_id = ? AND action IN ('like', 'unlike')
)
SELECT song_id FROM last WHERE rn = 1 AND action = 'like'
```

> 💡 이 쿼리는 "좋아요 → 좋아요 취소 → 좋아요"같은 토글 상황에서
> **가장 마지막 상태가 'like'인 곡만** 정확히 뽑아냅니다.

---

#### 3.2.2 `data/melon_pipeline.py` — 데이터 수집 파이프라인

**역할**: 멜론 차트에서 곡 목록을 크롤링하고, YouTube에서 오디오를 다운로드해서, librosa로 오디오 특성을 추출하는 **데이터 수집 전용 스크립트**입니다.

##### 전체 실행 흐름

```
python data/melon_pipeline.py 실행
        │
        ▼
┌─ gather_all_melon_tracks() ──────────────────────────────┐
│                                                           │
│  1) 멜론 차트 크롤링 (scrape_chart_url)                    │
│     - TOP100, HOT100, 일간, 주간, 월간 차트                │
│     - HTML에서 <tr class="lst50"> 파싱                     │
│                                                           │
│  2) 장르별 곡 수집 (scrape_genre_code)                     │
│     - GN0100(한국대중), GN0900(해외POP), GN1500            │
│     - 페이지네이션: 50곡/페이지씩 반복 요청                  │
│                                                           │
│  3) 추가 URL 수집 (scrape_extra_page) — 선택사항           │
│                                                           │
│  4) dedupe_tracks() — (title, artist) 기준 중복 제거        │
└──────────────────────────────────────────────────────────┘
        │
        ▼  고유 곡 목록 [(title, artist), ...]
        │
┌─ 곡마다 반복 처리 ──────────────────────────────────────┐
│                                                           │
│  5) search_youtube_ytdlp(title, artist)                   │
│     - yt-dlp로 YouTube 검색 → video_id, URL 획득           │
│                                                           │
│  6) upsert_song() → DB songs 테이블 저장                   │
│                                                           │
│  7) download_audio(youtube_url) → MP3 다운로드              │
│     - 임시 디렉토리에 저장 (처리 후 자동 삭제)               │
│                                                           │
│  8) extract_features(mp3_path) → librosa 오디오 분석        │
│     - MFCC (13차원), BPM, Energy, Spectral Centroid, ZCR   │
│                                                           │
│  9) upsert_audio_features() → DB audio_features 저장       │
└──────────────────────────────────────────────────────────┘
```

##### 주요 함수 상세

**`scrape_chart_url(url, max_results)`**

- 멜론 차트 페이지 HTML을 가져와서 `<tr class="lst50">` 또는 `<tr class="lst100">` 태그에서 순위, 제목, 가수를 추출합니다.
- `HEADERS` 딕셔너리로 브라우저인 척 위장합니다 (User-Agent 설정). 이게 없으면 멜론이 요청을 차단합니다.

**`scrape_genre_code(gnr_code, max_total)`**

- 장르별 곡 목록 페이지(song_listPaging.htm)를 페이징하며 크롤링합니다.
- `pageIndex`가 1 → 51 → 101 → ... 으로 증가합니다 (멜론의 페이징 규칙).

**`search_youtube_ytdlp(title, artist)`**

- `yt-dlp` 라이브러리로 `"{title} {artist} official"` 검색어를 YouTube에서 검색합니다.
- `extract_flat=False`로 설정하여 **영상 메타데이터까지 함께 가져옵니다**.
- 반환: `{song_id, youtube_url, thumbnail_url, genre}`

**`download_audio(youtube_url, output_path)`**

- YouTube 영상의 오디오를 MP3로 다운로드합니다.
- FFmpeg 후처리로 `bestaudio` → MP3 128kbps 변환합니다.
- `tempfile.TemporaryDirectory()`를 사용해서 **처리 후 자동 삭제**됩니다 (디스크 공간 절약).

**`extract_features(audio_path, duration=30.0)`**

- librosa로 오디오 파일의 첫 30초만 분석합니다 (속도 최적화).
- 추출하는 특성:

| 특성                                       | 설명                       | 차원 수         |
| ------------------------------------------ | -------------------------- | --------------- |
| MFCC (Mel-Frequency Cepstral Coefficients) | 음색/톤의 수학적 표현      | 13개            |
| BPM (Beats Per Minute)                     | 템포/빠르기                | 1개             |
| Energy (RMS)                               | 음량/에너지 크기           | 1개             |
| Spectral Centroid                          | 밝기 (고음 vs 저음 비율)   | 1개             |
| ZCR (Zero-Crossing Rate)                   | 소리 질감 (노이즈 vs 클린) | 1개             |
| **합계**                                   |                            | **17차원 벡터** |

**`run()`** — 메인 실행 함수

- 이미 특성이 있는 곡은 스킵합니다 (중복 수집 방지).
- tqdm으로 진행 상황을 프로그레스 바로 보여줍니다.
- 각 곡 처리 사이에 `time.sleep(0.3)` — 서버 부하를 줄이기 위한 쓰로틀링입니다.

---

#### 3.2.3 `data/faiss_index.py` — FAISS 벡터 인덱스

**역할**: 곡의 오디오 특성(17차원 벡터)을 FAISS 인덱스에 넣어서, "이 곡과 비슷한 곡"을 **밀리초 단위로 빠르게 검색**할 수 있게 합니다.

##### FAISS란?

> Facebook AI Similarity Search의 약자로, **대규모 벡터의 유사도 검색**을 매우 빠르게 수행하는 라이브러리입니다.
> 예: 100만 곡 중에서 "이 곡과 가장 비슷한 10곡"을 찾는 데 수 밀리초밖에 안 걸립니다.

##### 핵심 동작 원리

```
1. 원본 데이터 (songs + audio_features JOIN)
   ┌─────────────────────────────────────────┐
   │ song_id │ mfcc_1 │ ... │ bpm │ energy │ │
   │  abc123 │  -5.2  │ ... │ 120 │  0.03  │ │  ← 각 행이 17차원 벡터
   │  def456 │  -3.1  │ ... │  95 │  0.05  │ │
   └─────────────────────────────────────────┘

2. L2 정규화 (각 벡터의 길이를 1로 맞춤)
   → 내적(Inner Product) = 코사인 유사도가 됨

3. FAISS IndexFlatIP에 등록
   → 정확한 코사인 유사도 기반 검색 가능

4. 검색 시:
   query_vec = 기준 곡의 벡터
   scores, indices = index.search(query_vec, top_k + 1)
   → 자기 자신 제외 후 상위 top_k개 반환
```

##### 주요 메서드 설명

| 메서드                            | 역할                            | 반환값                   |
| --------------------------------- | ------------------------------- | ------------------------ |
| `build(data, use_cache)`          | DataFrame으로 FAISS 인덱스 구축 | 없음 (내부 상태 변경)    |
| `search_by_id(song_id, top_k)`    | 곡 ID 기반 유사곡 검색          | `{song_id: score}`       |
| `search_by_vector(vector, top_k)` | 임의 벡터로 유사곡 검색         | `{song_id: score}`       |
| `get_vector(song_id)`             | 특정 곡의 특성 벡터 반환        | `np.ndarray` 또는 `None` |

##### 캐시 시스템

```
build() 호출 시:
  1. 캐시 파일이 있으면 → 바로 로드 (빠름)
  2. 없으면 → 새로 구축 → 캐시 파일 저장

캐시 파일:
  - data/faiss.index     ← FAISS 인덱스 바이너리
  - data/faiss_id_map.npy ← song_id 매핑 배열
```

##### `id_map`과 `id_to_idx`의 역할

```
FAISS는 내부적으로 0, 1, 2, ... 같은 정수 인덱스를 사용합니다.
하지만 우리 곡은 "abc123" 같은 YouTube video_id가 식별자입니다.

id_map    = ["abc123", "def456", "ghi789"]  ← 인덱스 번호 → song_id
id_to_idx = {"abc123": 0, "def456": 1, ...}  ← song_id → 인덱스 번호

이 두 딕셔너리가 "정수 인덱스 ↔ 문자열 song_id" 변환을 담당합니다.
```

---

### 3.3 `algorithms/` — 추천 알고리즘

#### 3.3.1 `algorithms/base.py` — 추천기 인터페이스 (추상 클래스)

**역할**: 모든 추천 알고리즘이 반드시 구현해야 하는 **공통 규격(인터페이스)**을 정의합니다.

```python
class BaseRecommender(ABC):  # ABC = Abstract Base Class (추상 클래스)

    # ✅ 반드시 구현해야 하는 메서드 (@abstractmethod)
    def fit(self, data: pd.DataFrame) -> None:
        """모델 학습. 구현 마지막에 self.is_fitted = True 필수!"""

    def recommend(self, song_id: str, top_k: int = 10) -> dict[str, float]:
        """곡 기반 추천. {song_id: score} 형태로 반환"""

    # 📌 선택적으로 오버라이드 가능한 메서드
    def recommend_for_user(self, user_id, top_k) -> dict[str, float]:
        """유저 기반 추천. 기본값은 빈 딕셔너리"""

    # 🛡️ 내부 유틸
    def _check_fitted(self):
        """fit() 호출 여부 확인. 안 했으면 에러 발생"""
```

> 💡 **왜 추상 클래스를 쓰나요?**
> 팀원마다 다른 추천 알고리즘(CF, RL, BERT 등)을 만들어도,
> API 서버 코드를 수정하지 않고 **통일된 방식으로 등록하고 호출**할 수 있기 때문입니다.
> 이것을 **전략 패턴(Strategy Pattern)**이라고 합니다.

#### 3.3.2 `algorithms/faiss_cbf.py` — 샘플 추천기

**역할**: `BaseRecommender`를 상속받아 구현한 **샘플 추천 알고리즘**입니다.
FAISS 인덱스를 활용한 콘텐츠 기반 필터링(CBF: Content-Based Filtering)입니다.

```python
class FaissContentRecommender(BaseRecommender):

    def __init__(self, faiss_index):
        super().__init__("faiss_cbf")  # 알고리즘 이름 등록
        self._index = faiss_index       # FAISS 인덱스 객체 주입

    def fit(self, data):
        self._data = data
        self.is_fitted = True  # 실제 학습은 FAISS 쪽에서 처리됨

    def recommend(self, song_id, top_k=10):
        # FAISS 인덱스에서 코사인 유사도 기반 검색
        return self._index.search_by_id(song_id, top_k)

    def recommend_for_user(self, user_id, top_k=10):
        # 유저가 좋아요한 곡 중 첫 번째 곡으로 추천
        likes = get_user_liked_song_ids(user_id)
        for sid in likes:
            out = self._index.search_by_id(sid, top_k)
            if out:
                return out
        return {}

    def search_by_query(self, query, top_k=10):
        # 단순 키워드 매칭 (제목/가수에 검색어 포함 여부)
        # 각 토큰이 텍스트에 포함되면 hit += 1
        # 최종 점수 = hit / 총 토큰 수
```

> 💡 `search_by_query`는 간단한 문자열 매칭이라 검색 품질이 낮습니다.
> BERT나 LangChain 기반의 시맨틱 검색으로 교체하면 훨씬 나아집니다.

---

### 3.4 `evaluation/` — 추천 성능 평가

#### `evaluation/metrics.py`

**역할**: 추천 시스템의 품질을 수치로 측정하는 **평가 지표(metrics)**를 제공합니다.

##### 제공하는 지표

| 지표                | 의미                                | 수식 (간략)                        |
| ------------------- | ----------------------------------- | ---------------------------------- |
| **Precision@K**     | 추천 K개 중 실제 관련 곡의 비율     | `관련 곡 ∩ 추천 곡 / K`            |
| **Recall@K**        | 전체 관련 곡 중 추천에 포함된 비율  | `관련 곡 ∩ 추천 곡 / 전체 관련 곡` |
| **NDCG@K**          | 관련 곡이 상위에 있을수록 높은 점수 | DCG / IDCG (순위 보정)             |
| **Skip Rate**       | 전체 재생 중 스킵 비율              | `skip 수 / 전체 play 수`           |
| **Completion Rate** | 곡을 90% 이상 들은 비율             | `완청 수 / 전체 play 수`           |

##### 함수 사용 예시

```python
from evaluation.metrics import evaluate, evaluate_all

# 단일 평가
results = evaluate(
    recommended=["song_a", "song_b", "song_c"],  # 추천 결과
    relevant=["song_a", "song_d"],                 # 실제 정답
    k_list=[5, 10]
)
# → {"Precision@5": 0.2, "Recall@5": 0.5, "NDCG@5": 1.0, ...}

# 전체 평가 (여러 쿼리의 평균)
avg = evaluate_all(
    recommender=my_algo,
    test_data=[
        ("query_song_1", ["relevant_1", "relevant_2"]),
        ("query_song_2", ["relevant_3"]),
    ]
)
```

> 💡 NDCG가 Precision/Recall보다 중요한 이유:
> 추천에서는 "맞추는 것"뿐만 아니라 "좋은 곡을 상위에 배치하는 것"이 더 중요하기 때문입니다.

---

### 3.5 `api/` — FastAPI 백엔드

#### `api/main.py`

**역할**: 프론트엔드와 추천 엔진을 연결하는 **HTTP API 서버**입니다.
인증(JWT), 곡 조회, 추천, 검색, 인터랙션 기록 등 모든 API 엔드포인트를 제공합니다.

##### API 엔드포인트 목록

| 메서드 | 경로                   | 설명                                  | 인증 필요 |
| ------ | ---------------------- | ------------------------------------- | --------- |
| GET    | `/`                    | 웹 UI (index.html) 서빙               | ❌        |
| GET    | `/api/songs`           | 곡 목록 조회 (limit, genre 필터)      | ❌        |
| GET    | `/api/songs/{song_id}` | 곡 상세 조회                          | ❌        |
| POST   | `/api/signup`          | 회원가입                              | ❌        |
| POST   | `/api/login`           | 로그인 (JWT 토큰 발급)                | ❌        |
| GET    | `/api/me`              | 현재 로그인 사용자 정보               | ✅        |
| GET    | `/api/likes`           | 좋아요한 곡 목록                      | ✅        |
| POST   | `/api/recommend`       | 곡 기반 추천                          | ❌        |
| POST   | `/api/recommend/user`  | 유저 기반 추천                        | ✅        |
| POST   | `/api/search`          | 자연어 검색                           | ❌        |
| POST   | `/api/interact`        | 인터랙션 기록 (play/like/skip/unlike) | ✅        |
| GET    | `/api/algorithms`      | 등록된 알고리즘 목록                  | ❌        |

##### 서버 시작 시 (`startup()`)

```python
@app.on_event("startup")
async def startup():
    # 1. DB 초기화 (테이블 없으면 생성)
    init_db()

    # 2. 전체 곡 데이터 로드 (songs + audio_features JOIN)
    song_df = get_all_songs()

    # 3. FAISS 인덱스 구축 (오디오 특성 벡터 → 유사도 검색 준비)
    faiss_index = MusicFaissIndex()
    faiss_index.build(song_df)

    # 4. 추천기 등록 (recommenders 딕셔너리에 추가)
    recommenders["faiss_cbf"] = FaissContentRecommender(faiss_index)
    recommenders["faiss_cbf"].fit(song_df)
```

> 💡 **`recommenders` 딕셔너리가 핵심!**
> 모든 추천 API는 `recommenders[algorithm_name]`으로 추천기를 찾아서 호출합니다.
> 새 알고리즘을 추가하려면 이 딕셔너리에 등록만 하면 됩니다.

##### JWT 인증 흐름

```
1. 로그인/회원가입 → 서버가 JWT 토큰 발급
   토큰 내용: {"sub": "username", "exp": "만료시간"}
   서명키: JWT_SECRET

2. 프론트에서 localStorage에 저장

3. 인증이 필요한 API 호출 시:
   Headers: { Authorization: "Bearer <JWT 토큰>" }

4. 서버에서 토큰 검증:
   _get_current_username(authorization)
   → jwt.decode()로 디코딩
   → 만료됐으면 401 에러
   → 유효하면 username 반환
```

##### `_format()` — 추천 결과 가공 함수

```python
def _format(rec_dict: dict, algorithm: str) -> dict:
    # {song_id: score} → [{ song_id, title, artist, youtube_url, ... score }]
    # song_df에서 곡 메타데이터를 붙여서 프론트가 바로 렌더링할 수 있는 형태로 변환
```

---

### 3.6 `static/` — 프론트엔드 (웹 UI)

#### 3.6.1 `static/index.html` — HTML 구조

UI는 크게 4개 영역으로 구성됩니다:

```
┌──────────────────────────────────────────────────────┐
│  [auth-gate] 로그인/회원가입 화면 (인증 전에만 보임)    │
│  ┌────────────────────────────────────┐               │
│  │  아이디 입력 / 비밀번호 입력        │               │
│  │  [회원가입] [로그인]                │               │
│  └────────────────────────────────────┘               │
└──────────────────────────────────────────────────────┘

인증 후 ↓

┌─────────┬────────────────────────────────────────────┐
│ sidebar │              main (뷰 전환)                 │
│         │                                             │
│ 🎵로고   │  [view-home]     추천 음악                  │
│ [홈]     │  ┌──────────────────────────────────┐      │
│ [좋아요]  │  │  🔍 검색 바                       │      │
│          │  │  🃏 곡 카드 3개 (그리드)            │      │
│ 알고리즘  │  │  📋 검색 결과 (접이식)             │      │
│ [선택]    │  └──────────────────────────────────┘      │
│          │                                             │
│ 장르 목록 │  [view-liked]   좋아요한 곡 목록            │
│          │                                             │
│ 유저 정보 │                                             │
│ [로그아웃]│                                             │
├─────────┴────────────────────────────────────────────┤
│ [player] 하단 플레이어 바                               │
│ 💿 곡 정보 | ▶ YouTube에서 듣기 | 알고리즘 배지          │
└──────────────────────────────────────────────────────┘
```

**`<template id="song-card-tpl">`** — 곡 카드 템플릿

- `<template>` 태그는 화면에 렌더링되지 않고, JS에서 `cloneNode`로 복제해서 사용합니다.
- 카드 구성: 썸네일 + 재생 버튼 + 유사도 점수 뱃지 + 제목/가수 + 좋아요/스킵 버튼

#### 3.6.2 `static/app.js` — 프론트엔드 로직

##### 전역 상태

```javascript
let currentSong = null; // 현재 선택/재생 중인 곡
let authUser = null; // 로그인한 사용자 정보
let authToken = null; // JWT 토큰
let likedSongs = new Set(); // 좋아요한 곡 ID 집합
let songCache = new Map(); // song_id → 곡 객체 캐시
```

##### 앱 초기화 흐름 (`DOMContentLoaded`)

```
1. bindAuthEvents()       → 로그인/회원가입 버튼에 이벤트 연결
2. bootstrapAuth()        → localStorage에 저장된 토큰으로 자동 로그인 시도
3. setupNav()             → 사이드바 네비게이션 (홈/좋아요) 전환
4. loadAlgorithms()       → /api/algorithms → 알고리즘 셀렉트 박스 채우기
5. loadGenres()           → /api/songs → 장르 목록 추출
6. loadSongs()            → /api/songs → 곡 그리드 표시
7. 이벤트 리스너 등록      → 검색, 유튜브 열기, 좋아요, 추천 받기 등
```

##### 핵심 함수 흐름

```
사용자가 곡 카드 클릭
    │
    ├─ updatePlayer(song)          → 하단 플레이어 UI 업데이트
    ├─ logInteraction(song_id, "play")  → POST /api/interact
    └─ fetchRecommendations(song_id)
           │
           └─ POST /api/recommend
              { song_id, algorithm, top_k: 3 }
                   │
                   ▼
              renderGrid(recommendations, "song-grid", true)
              → 추천 결과 3개를 카드로 표시
```

##### `apiFetch()` — API 통신 유틸

```javascript
async function apiFetch(url, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (authToken) {
    opts.headers.Authorization = `Bearer ${authToken}`; // JWT 토큰 자동 첨부
  }
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + url, opts);
  if (!res.ok) return null; // 에러 시 null 반환 (예외 안 던짐)
  return await res.json();
}
```

> 💡 모든 API 호출이 이 함수를 거칩니다. 인증 토큰을 자동으로 붙여주므로,
> 개별 API 호출 시 토큰 관리를 신경 쓸 필요가 없습니다.

#### 3.6.3 `static/style.css` — 스타일

- **Spotify 스타일 다크 테마**: 검정 배경(`#0a0a0a`) + 초록 액센트(`#1db954`)
- CSS Grid 기반 레이아웃: `grid-template-columns: var(--sidebar-w) 1fr`
- CSS Custom Properties(변수)를 `--bg`, `--accent` 등으로 정의하여 테마 변경이 쉽습니다
- 반응형 처리: `@media (max-width: 900px)` — 모바일에서 3열 → 1열

---

## 4. 핵심 기술 개념 설명

### 4.1 CBF (Content-Based Filtering) — 콘텐츠 기반 필터링

> "이 곡의 **음악적 특성(MFCC, BPM 등)**이 비슷한 다른 곡을 추천한다"

```
곡 A: [mfcc=-5.2, bpm=120, energy=0.03, ...]  ← 17차원 벡터
곡 B: [mfcc=-5.1, bpm=118, energy=0.04, ...]  ← 벡터가 가까움 → 유사!
곡 C: [mfcc= 3.8, bpm= 70, energy=0.01, ...]  ← 벡터가 멀음 → 다른 곡
```

장점: 새 곡이 추가되어도 바로 추천 가능 (유저 데이터 불필요)
단점: 사용자 취향 반영 불가, "음악적으로 비슷함 ≠ 사용자가 좋아함"

### 4.2 코사인 유사도 (Cosine Similarity)

> 두 벡터가 이루는 **각도**로 유사도를 측정합니다.
> 벡터의 크기(길이)는 무시하고, **방향(패턴)**만 비교합니다.

```
코사인 유사도 = cos(θ) = (A · B) / (|A| × |B|)

L2 정규화 후: |A| = |B| = 1
→ 코사인 유사도 = A · B (단순 내적!)

이것이 FAISS IndexFlatIP(Inner Product)를 사용하는 이유입니다.
```

### 4.3 JWT (JSON Web Token) 인증

```
토큰 구조: header.payload.signature

payload = {
  "sub": "username",      ← 사용자 식별자
  "exp": 1711411200       ← 만료 시간 (Unix timestamp)
}

서명 = HMAC-SHA256(header + "." + payload, JWT_SECRET)

검증: 서버가 같은 JWT_SECRET으로 다시 서명해서 일치하는지 확인
→ 토큰 위변조 방지 (DB 조회 없이 인증 가능)
```

### 4.4 전략 패턴 (Strategy Pattern)

이 프로젝트의 알고리즘 구조가 **전략 패턴**입니다:

```
BaseRecommender (인터페이스)
    ├── FaissContentRecommender (FAISS CBF)
    ├── MyBertRecommender       (팀원 A가 구현)
    └── MyRLRecommender         (팀원 B가 구현)

API 서버는 어떤 알고리즘이든:
    recommenders[algorithm_name].recommend(song_id, top_k)
→ 호출 방법이 동일! 새 알고리즘 추가해도 API 코드 변경 불필요
```

---

## 5. 실행 순서 가이드

### STEP 1: 환경 준비

```bash
# 1. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. 패키지 설치
pip install -U pip setuptools wheel
pip install -r requirements.txt

# 3. ffmpeg 설치 (오디오 처리에 필요)
# Windows: https://www.gyan.dev/ffmpeg/builds/ 에서 다운로드 후 PATH에 추가
# macOS: brew install ffmpeg

# 4. 환경변수 설정
# .env 파일 생성 후 아래 내용 작성:
# DB_PATH=./music_rec.db
# JWT_SECRET=강한_랜덤_문자열
# JWT_EXPIRE_HOURS=24
```

### STEP 2: 데이터 수집

```bash
python data/melon_pipeline.py
# 멜론 크롤링 → YouTube 매칭 → 오디오 다운로드 → 특성 추출 → DB 저장
# ⚠️ 시간이 오래 걸립니다 (곡 수에 따라 수십 분~수 시간)
```

### STEP 3: 서버 실행

```bash
uvicorn api.main:app --reload
# http://127.0.0.1:8000 에서 웹 UI 접속
```

### STEP 4: 웹 UI 사용

1. 회원가입 → 로그인
2. 사이드바에서 알고리즘 선택 (faiss_cbf)
3. 곡 카드 클릭 → 하단 플레이어 업데이트 + 유사곡 추천
4. 검색바에 키워드 입력 → 검색 결과 표시
5. 좋아요/스킵으로 인터랙션 기록
6. "내 취향 추천 받기" → 좋아요 기반 유저 추천

---

## 6. 새 추천 알고리즘 추가하는 방법

### Step 1: 알고리즘 파일 생성

`algorithms/` 폴더에 새 파일을 만듭니다:

```python
# algorithms/my_algo.py

from algorithms.base import BaseRecommender
import pandas as pd

class MyRecommender(BaseRecommender):

    def __init__(self, faiss_index=None):
        super().__init__("my_algo")    # ← 알고리즘 이름 (API에서 사용)
        self._index = faiss_index

    def fit(self, data: pd.DataFrame) -> None:
        """
        여기에 모델 학습 로직을 구현합니다.
        data는 songs + audio_features가 JOIN된 DataFrame입니다.
        """
        # 예: self._model = train_model(data)
        self.is_fitted = True          # ← 이 줄 필수!

    def recommend(self, song_id: str, top_k: int = 10) -> dict[str, float]:
        """
        곡 기반 추천을 구현합니다.
        반드시 {song_id: score} 딕셔너리를 반환하세요.
        score가 높을수록 추천 우선순위가 높습니다.
        """
        self._check_fitted()
        # 예: return {"abc123": 0.95, "def456": 0.88, ...}
        return {}
```

### Step 2: 서버에 등록

`api/main.py`의 `startup()` 함수에 추가합니다:

```python
# api/main.py → startup() 안에 추가
from algorithms.my_algo import MyRecommender

recommenders["my_algo"] = MyRecommender(faiss_index)
recommenders["my_algo"].fit(song_df)
print("[API] 알고리즘 등록: my_algo")
```

### Step 3: 확인

서버를 재시작하면 웹 UI의 알고리즘 셀렉트 박스에 `my_algo`가 나타납니다.

---

## 7. 코드 품질 리뷰 및 개선 포인트

### ✅ 잘 된 점

| 항목                 | 설명                                                                 |
| -------------------- | -------------------------------------------------------------------- |
| **추상 클래스 활용** | `BaseRecommender`로 알고리즘 인터페이스를 통일하여 확장성이 높음     |
| **비밀번호 보안**    | bcrypt 해싱 + SHA-256 레거시 자동 마이그레이션                       |
| **FAISS 캐시**       | 매번 인덱스를 새로 구축하지 않고 파일로 캐시하여 서버 시작 속도 개선 |
| **임시 파일 관리**   | `tempfile.TemporaryDirectory()`로 오디오 파일 자동 삭제              |
| **중복 수집 방지**   | 이미 특성이 있는 곡은 자동 스킵                                      |
| **좋아요 토글 로직** | Window Function으로 최종 상태만 정확히 추출                          |
| **프론트 캐싱**      | `songCache` Map으로 중복 API 호출 방지                               |

### ⚠️ 개선이 필요한 점

#### 1. DB 커넥션 관리 — 매번 열고 닫는 패턴

```python
# 현재 방식 (database.py의 모든 함수)
def some_func():
    conn = get_conn()     # 매번 새 연결
    # ... 작업 ...
    conn.close()          # 매번 닫기
```

**문제**: 매 API 요청마다 DB 연결을 새로 만들고 닫으므로 오버헤드가 발생합니다.
**개선안**: 커넥션 풀(Connection Pool)이나 FastAPI의 의존성 주입(Depends)을 사용하세요.

```python
# 개선 예시
from contextlib import contextmanager

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# 사용
with get_conn() as conn:
    conn.execute(...)
```

#### 2. 전역 변수 의존 (`song_df`, `recommenders`)

```python
# api/main.py
song_df = None          # 전역 변수
recommenders = {}       # 전역 변수
```

**문제**: 전역 상태에 의존하면 테스트가 어렵고, 코드 추적이 복잡해집니다.
**개선안**: FastAPI의 `app.state`나 의존성 주입 패턴을 사용하세요.

```python
# 개선 예시
app.state.song_df = get_all_songs()
app.state.recommenders = {}
```

#### 3. `upsert_audio_features`의 SQL 인젝션 위험

```python
# 현재 코드
cols = ", ".join(features.keys())      # 컬럼명을 직접 문자열로 조합
vals = ", ".join(f":{k}" for k in features.keys())
conn.execute(f"INSERT INTO audio_features ({cols}) VALUES ({vals}) ...")
```

**문제**: `features`의 key에 악성 문자열이 들어오면 SQL 인젝션이 가능합니다.
**개선안**: 허용된 컬럼명만 화이트리스트로 검증하세요.

```python
ALLOWED_COLS = {"song_id", "mfcc_1", ..., "zcr"}
for key in features.keys():
    if key not in ALLOWED_COLS:
        raise ValueError(f"허용되지 않은 컬럼: {key}")
```

#### 4. `_hash_password` 함수가 미사용 (데드 코드)

```python
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()
```

**상태**: `authenticate_user`의 레거시 호환 로직에서만 내부적으로 사용됩니다.
새 회원가입은 모두 bcrypt를 사용하므로, 충분한 시간이 지나면 레거시 코드와 함께 제거해도 됩니다.

#### 5. 에러 처리 강화 필요

```python
# melon_pipeline.py — 네트워크 에러 시 전체 파이프라인이 중단될 수 있음
def _fetch(url, referer=None):
    try:
        res = requests.get(url, headers=h, timeout=20)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"  ⚠️ 요청 실패: ...")
        return None
```

**개선안**: 재시도(retry) 로직을 추가하면 네트워크 불안정 시에도 안정적으로 동작합니다.

```python
import time

def _fetch_with_retry(url, max_retries=3, delay=2):
    for attempt in range(max_retries):
        result = _fetch(url)
        if result is not None:
            return result
        time.sleep(delay * (attempt + 1))  # 점점 오래 대기
    return None
```

#### 6. 프론트엔드 — XSS 방지 부족

```javascript
// app.js — song.title을 textContent로 설정 (이건 안전!)
tpl.querySelector(".card-title").textContent = song.title; // ✅ 안전

// 하지만 innerHTML을 사용하는 부분도 있음:
container.innerHTML = "<p class='placeholder'>곡이 없습니다.</p>"; // ⚠️ 정적 문자열이라 안전
```

현재는 큰 문제가 없지만, 향후 사용자 입력을 HTML에 넣을 때는 반드시 `textContent`를 사용하세요.

#### 7. `@app.on_event("startup")` 지원 중단 예정

```python
@app.on_event("startup")   # ⚠️ FastAPI에서 deprecated 예정
async def startup():
```

**개선안**: `lifespan` 패턴으로 변경하세요.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup 로직
    init_db()
    app.state.song_df = get_all_songs()
    yield
    # shutdown 로직 (필요 시)

app = FastAPI(lifespan=lifespan)
```

---

## 8. 자주 묻는 질문 (FAQ)

### Q1: 서버를 실행했는데 "알고리즘이 없다"고 나와요

**A**: 데이터가 없어서 FAISS 인덱스가 구축되지 않은 것입니다.
먼저 `python data/melon_pipeline.py`로 데이터를 수집하세요.

### Q2: 멜론 크롤링이 안 돼요

**A**: 멜론이 User-Agent 없이는 요청을 차단합니다.
`melon_pipeline.py`의 `HEADERS`에 User-Agent가 설정되어 있는지 확인하세요.
또한 멜론 사이트 구조가 변경되면 파서도 수정이 필요합니다.

### Q3: ffmpeg가 없다고 나와요

**A**: `yt-dlp`가 오디오 변환에 ffmpeg를 사용합니다.

- Windows: [gyan.dev/ffmpeg](https://www.gyan.dev/ffmpeg/builds/)에서 다운로드 → 환경변수 PATH에 추가
- macOS: `brew install ffmpeg`

### Q4: 새 알고리즘을 추가했는데 API에 안 나와요

**A**: `api/main.py`의 `startup()` 함수에서 `recommenders` 딕셔너리에 등록했는지 확인하세요.
서버 재시작이 필요합니다 (`--reload` 옵션 사용 시 자동 재시작).

### Q5: `recommend()`의 반환값 형태가 뭔가요?

**A**: `{song_id: score}` 형태의 딕셔너리입니다. score가 높을수록 추천 우선순위가 높습니다.

```python
{"abc123": 0.95, "def456": 0.88, "ghi789": 0.72}
```

### Q6: DB 파일은 어디 있나요?

**A**: 프로젝트 루트의 `music_rec.db` 파일입니다 (config.py의 DB_PATH 설정).
SQLite 뷰어(DB Browser for SQLite 등)로 직접 열어볼 수 있습니다.

### Q7: 프론트엔드를 수정하면 바로 반영되나요?

**A**: 네. `static/` 폴더는 FastAPI가 정적 파일로 서빙하므로,
HTML/CSS/JS를 수정하고 브라우저를 새로고침하면 바로 반영됩니다.

---

## 📎 부록: 파일 간 의존성 맵

```
utils/config.py
    ▲
    │ (설정값 import)
    │
    ├── data/database.py     ← DB_PATH
    ├── data/melon_pipeline.py ← MELON_* 설정값
    ├── api/main.py          ← JWT_SECRET, JWT_EXPIRE_HOURS
    └── evaluation/metrics.py ← EVAL_K_LIST

data/database.py
    ▲
    │ (DB 함수 import)
    │
    ├── data/melon_pipeline.py ← upsert_song, upsert_audio_features, get_all_songs
    ├── algorithms/faiss_cbf.py ← get_user_liked_song_ids
    └── api/main.py           ← init_db, get_all_songs, create_user, authenticate_user, ...

data/faiss_index.py
    ▲
    │ (FAISS 인덱스 import)
    │
    ├── algorithms/faiss_cbf.py ← MusicFaissIndex 인스턴스 사용
    └── api/main.py            ← 인덱스 구축 및 알고리즘에 전달

algorithms/base.py
    ▲
    │ (추상 클래스 상속)
    │
    └── algorithms/faiss_cbf.py ← BaseRecommender 상속

algorithms/faiss_cbf.py
    ▲
    │ (추천기 import)
    │
    └── api/main.py ← FaissContentRecommender 등록

static/ (app.js → api/main.py의 HTTP 엔드포인트 호출)
```

---

> 📝 **마지막으로**: 이 프로젝트는 "음악 추천 시스템의 뼈대"를 제공하는 **베이스 프로젝트**입니다.
> FAISS CBF 추천기는 샘플이므로, 이를 참고하여 **협업 필터링(CF), 강화학습(RL), BERT 기반 검색** 등
> 다양한 알고리즘을 `algorithms/` 폴더에 추가해 나가면 됩니다.
> `BaseRecommender`만 상속하면 API 코드 수정 없이 바로 연동됩니다. 🎵
