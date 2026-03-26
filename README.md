# AI Personal Music Recommendation Base

SQLite 기반 곡·오디오 특성 저장, FAISS 유사도 검색, **JWT 인증(bcrypt)**, FastAPI, 정적 웹 UI를 포함한 음악 추천 시스템 베이스입니다.

## 프로젝트 구조

```text
music-rec-base/
├── api/
│   └── main.py                # FastAPI: 인증, 추천/검색/인터랙션/좋아요 API
├── data/
│   ├── melon_pipeline.py      # Melon 크롤링 + YouTube 매칭 + 오디오 특성 추출
│   ├── database.py            # SQLite 초기화/CRUD, interactions·좋아요 조회
│   └── faiss_index.py         # 오디오 특성 기반 FAISS 인덱스
├── algorithms/
│   ├── base.py                # 추천기 인터페이스 베이스
│   └── faiss_cbf.py           # 베이스 샘플: FAISS 오디오 유사도 + 키워드 검색
├── evaluation/
│   └── metrics.py             # 추천 평가 유틸
├── static/                    # 웹 UI (인증 게이트, 홈·검색·좋아요, 플레이어)
├── utils/
│   └── config.py              # 환경변수, JWT, 수집/추천 설정
├── docs/                      # 팀장 로컬 문서 (Git 제외 — `.gitignore`의 `docs/`)
├── requirements.txt
└── README.md
```

## 요구 환경

- Python 3.11 권장
- macOS/Linux 기준 명령어
- `ffmpeg` 설치 권장 (`yt-dlp` 오디오 후처리에 필요)

## 설치

```bash
cd music-rec-base
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

> 참고: `scikit-surprise`는 일부 macOS 환경에서 빌드 이슈가 있어 기본 의존성에서 분리되어 있습니다. 필요 시에만 수동 설치하세요.
>
> ```bash
> pip install scikit-surprise==1.1.4
> ```

## 환경변수 설정

템플릿 **`.env.example`** 을 복사해 `.env` 를 만든 뒤 값을 채우세요.

```bash
cp .env.example .env
```

```dotenv
DB_PATH=./music_rec.db
JWT_SECRET=강한_랜덤_문자열로_변경
JWT_EXPIRE_HOURS=24

# 선택
# GEMINI_API_KEY=
# YOUTUBE_API_KEY=
```

| 변수               | 설명                                                            |
| ------------------ | --------------------------------------------------------------- |
| `DB_PATH`          | SQLite 파일 경로 (기본 `./music_rec.db`)                        |
| `JWT_SECRET`       | JWT 서명 키. **로컬 기본값은 개발용**이므로 배포 전 반드시 변경 |
| `JWT_EXPIRE_HOURS` | 액세스 토큰 만료 시간(시간 단위)                                |
| `GEMINI_API_KEY`   | LangChain/LangGraph·Gemini 연동 시                              |
| `YOUTUBE_API_KEY`  | YouTube Data API 사용 시                                        |

## 데이터 수집 파이프라인

### Melon 기반 통합 파이프라인

```bash
python data/melon_pipeline.py
```

- **멜론 여러 차트** (기본: TOP100, HOT100, 일간, 주간) + **장르별 곡 목록** (페이징, 곡 수는 `utils/config.py`에서 조절)
- (선택) `**MELON_EXTRA_SONG_PAGE_URLS`\*\* — 플레이리스트 등 곡 리스트 URL. HTML이 차트형(`lst50`)이거나 장르형(`wrap_song_info`)일 때만 파싱됩니다 (JS 전용 페이지는 실패할 수 있음).
- 소스를 합친 뒤 **제목+가수 기준 중복 제거** → `yt-dlp`로 YouTube 매칭
- 오디오 다운로드 후 `librosa` 특성(MFCC, BPM, energy 등) 17개 컬럼 추출
- `songs` + `audio_features` 동시 저장, 임시 오디오 파일 즉시 삭제

곡 수·소스는 `utils/config.py`의 `MELON_CHART_URLS`, `MELON_MAX_SONGS_PER_CHART`, `MELON_GENRE_CODES`, `MELON_MAX_SONGS_PER_GENRE`로 조정합니다.

YouTube **Data API v3**로 메타만 모으는 모듈이 필요하면 팀에서 별도 파일을 두면 됩니다. `requirements.txt`의 `google-api-python-client`는 그런 확장용입니다.

## API 실행

```bash
uvicorn api.main:app --reload
```

브라우저에서 `http://localhost:8000` 접속.

앱 시작 시 동작:

- DB 초기화 및 곡 데이터 로드
- `MusicFaissIndex` 인덱스 캐시 로드 또는 재구축 (실패해도 곡 API 등은 기동 유지)
- FAISS가 정상 빌드되면 `**faiss_cbf**` 샘플 알고리즘이 `recommenders`에 자동 등록됩니다.

### `startup()`에서 FAISS 넘기기

`MusicFaissIndex` 인스턴스는 `**startup()`의 `try` 블록 안**에서만 생성됩니다. 팀 알고리즘을 등록할 때는 **같은 함수 안\*\*에서 방금 만든 `faiss_index` 변수를 생성자에 넘기면 됩니다. 로직을 다른 함수로 빼면 인자로 전달하거나, 모듈 전역(예: `app.state`)에 보관하는 식으로 참조를 유지하세요.

## 주요 API

| 메서드 | 경로                   | 인증   | 설명                                                            |
| ------ | ---------------------- | ------ | --------------------------------------------------------------- |
| POST   | `/api/signup`          | —      | 회원가입 → `access_token`                                       |
| POST   | `/api/login`           | —      | 로그인 → `access_token`                                         |
| GET    | `/api/me`              | Bearer | 현재 사용자명                                                   |
| GET    | `/api/likes`           | Bearer | DB 기준 좋아요 곡 목록                                          |
| GET    | `/api/songs`           | —      | `?limit=30&genre=...`                                           |
| GET    | `/api/songs/{song_id}` | —      | 곡 상세                                                         |
| POST   | `/api/recommend`       | —      | 곡 기반 추천                                                    |
| POST   | `/api/recommend/user`  | Bearer | 유저 기반 추천                                                  |
| POST   | `/api/search`          | —      | 자연어 검색 (`search_by_query` 구현 시)                         |
| POST   | `/api/interact`        | Bearer | `play` / `like` / `skip` / `unlike`                             |
| GET    | `/api/algorithms`      | —      | `{ "algorithms": [...] }` · 비어 있으면 `notice` 안내 문구 포함 |

보호된 엔드포인트는 헤더에 `Authorization: Bearer <access_token>`을 붙입니다.

FAISS가 비어 있거나 초기화에 실패하면 `recommenders`가 비어 있을 수 있습니다. 그때는 사이드바에 안내가 표시되고, `GET /api/algorithms` 응답의 `notice`를 참고하면 됩니다.

## 웹 UI

- 로그인/회원가입 후 토큰은 브라우저 `localStorage`에 저장됩니다.
- **알고리즘 0개**일 때 사이드바에 회색 안내 문구가 나옵니다.
- **홈:** 추천 최대 3곡 그리드 + 같은 화면의 검색창(별도 검색 탭 없음).
- **좋아요:** 서버 `interactions`에 저장되며, 새로고침 시 `GET /api/likes`로 복원됩니다.

## DB 확인

```bash
sqlite3 music_rec.db
```

```sql
.tables
.schema
SELECT * FROM songs LIMIT 10;
```

## 알고리즘 개발 메모

- `algorithms/base.py` 기반으로 추천기 구현 (`fit` 끝에 `self.is_fitted = True`)
- 참고 구현: `algorithms/faiss_cbf.py` (`FaissContentRecommender`)
- `fit(...)`, `recommend(...)` 인터페이스 준수
- 유저 맞춤이 필요하면 `recommend_for_user(user_id, top_k)` 구현
- 자연어 검색이 필요하면 `search_by_query(query, top_k)` 구현
- 반환 포맷은 `{song_id: score}` 딕셔너리 권장
- `api/main.py`의 `startup()`에서 인스턴스 생성·학습·`recommenders["키"]` 등록

## 추가 문서 (로컬)

- `docs/project-overview.md` — 기획·역할 분담
- `docs/team-faq.md` — 실행·코드 Q&A

`docs/` 는 **팀장 로컬 확인용**으로 `.gitignore`에 두어 Git에 포함하지 않습니다. 팀원에게는 필요 시 파일을 따로 공유하세요.
