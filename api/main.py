from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
import jwt
from data.database import (
    init_db, get_all_songs, get_song,
    create_user, authenticate_user,
    get_or_create_user, log_interaction,
    get_user_liked_songs,
)
from utils.config import JWT_SECRET, JWT_EXPIRE_HOURS

app = FastAPI(title="🎵 AI 음악 추천 시스템", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 전역 상태
song_df    = None
recommenders = {}   # {"알고리즘명": recommender 인스턴스}


def _initialize_app_state() -> None:
    """
    서버가 추천에 필요한 전역 상태를 한 번에 준비하는 공통 초기화 함수.

    startup()에서만 초기화에 의존하면, 개발 중 리로드 타이밍이나 첫 요청 시점에
    song_df가 아직 비어 있어서 홈 화면이 비는 경우가 생길 수 있습니다.
    그래서 이 함수로 DB 로드와 추천기 등록을 묶어두고, 필요하면 요청 처리 중에도
    다시 호출할 수 있게 만들어 안정성을 높입니다.
    """
    global song_df, recommenders

    init_db()
    song_df = get_all_songs()
    print(f"[API] {len(song_df)}곡 로드 완료")

    recommenders.clear()

    faiss_index = None
    try:
        from data.faiss_index import MusicFaissIndex

        faiss_index = MusicFaissIndex()
        faiss_index.build(song_df)
    except Exception as e:
        print(f"[API] FAISS 초기화 실패: {e}")

    if faiss_index is not None and getattr(faiss_index, "is_built", False):
        from algorithms.faiss_cbf import FaissContentRecommender
        from algorithms.hybrid_recommender import HybridRecommender

        recommenders["faiss_cbf"] = FaissContentRecommender(faiss_index)
        recommenders["faiss_cbf"].fit(song_df)
        print("[API] 알고리즘 등록: faiss_cbf")

        recommenders["hybrid"] = HybridRecommender(faiss_index)
        recommenders["hybrid"].fit(song_df)
        print("[API] 알고리즘 등록: hybrid")


def _ensure_app_ready() -> None:
    """
    요청 처리 직전에 앱 상태가 준비되어 있는지 확인하고, 비어 있으면 즉시 초기화하는 함수.

    기본 홈 화면은 /api/songs 응답에 의존하므로, startup() 타이밍 이슈가 있어도
    이 함수가 있으면 첫 요청에서 바로 상태를 복구할 수 있습니다.
    """
    global song_df, recommenders
    if song_df is None or song_df.empty or not recommenders:
        _initialize_app_state()


@app.on_event("startup")
async def startup():
    _initialize_app_state()


# ── 웹 UI ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── 곡 API ────────────────────────────────────────────────

@app.get("/api/songs")
def list_songs(limit: int = 30, genre: str = None):
    _ensure_app_ready()
    df = song_df.copy()
    if genre:
        df = df[df["genre"] == genre]
    return df.head(limit)[
        ["song_id", "title", "artist", "youtube_url", "thumbnail_url", "genre"]
    ].to_dict(orient="records")


@app.get("/api/songs/{song_id}")
def get_song_detail(song_id: str):
    _ensure_app_ready()
    song = get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="곡을 찾을 수 없습니다")
    return song


# ── 추천 API ──────────────────────────────────────────────

class RecommendRequest(BaseModel):
    song_id:   str
    algorithm: str = "default"
    top_k:     int = 10
    exclude_song_ids: list[str] = []


class UserRecommendRequest(BaseModel):
    username:  str
    algorithm: str = "default"
    top_k:     int = 10
    exclude_song_ids: list[str] = []


class SearchRequest(BaseModel):
    query:     str
    algorithm: str = "default"
    top_k:     int = 10
    exclude_song_ids: list[str] = []


class AuthRequest(BaseModel):
    username: str
    password: str


def _create_token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {"sub": username, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def _get_current_username(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")


def _exclude_song_ids(rec_dict: dict[str, float], exclude_song_ids: list[str]) -> dict[str, float]:
    """
    추천 결과에서 프론트가 제외 요청한 song_id들을 제거하는 후처리 함수.

    스킵(X) 버튼을 눌렀을 때 같은 곡이 다시 추천되는 문제를 막기 위해,
    프론트가 보낸 제외 목록을 서버 결과에 한 번 더 적용합니다.

    Args:
        rec_dict: 추천기에서 생성한 {song_id: score} 결과
        exclude_song_ids: 응답에서 제외할 곡 ID 목록

    Returns:
        제외 목록이 반영된 추천 결과 딕셔너리
    """
    if not exclude_song_ids:
        return rec_dict
    excluded = set(exclude_song_ids)
    return {sid: score for sid, score in rec_dict.items() if sid not in excluded}


@app.post("/api/signup")
def signup(req: AuthRequest):
    username = req.username.strip()
    if len(username) < 2 or len(req.password) < 4:
        raise HTTPException(status_code=400, detail="아이디 2자 이상, 비밀번호 4자 이상이 필요합니다")
    if not create_user(username, req.password):
        raise HTTPException(status_code=409, detail="이미 존재하는 아이디입니다")
    token = _create_token(username)
    return {"status": "ok", "username": username, "access_token": token}


@app.post("/api/login")
def login(req: AuthRequest):
    username = req.username.strip()
    if not authenticate_user(username, req.password):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다")
    token = _create_token(username)
    return {"status": "ok", "username": username, "access_token": token}


@app.get("/api/me")
def me(authorization: str | None = Header(default=None)):
    username = _get_current_username(authorization)
    return {"username": username}


@app.get("/api/likes")
def list_likes(authorization: str | None = Header(default=None)):
    """현재 로그인 사용자의 좋아요 곡 목록 (DB interactions 기준)."""
    username = _get_current_username(authorization)
    user_id = get_or_create_user(username)
    likes = get_user_liked_songs(user_id)
    return {"likes": likes}


@app.post("/api/recommend")
def recommend_by_song(req: RecommendRequest):
    """곡 기반 추천"""
    _ensure_app_ready()
    if req.algorithm not in recommenders:
        raise HTTPException(
            status_code=400,
            detail=f"알고리즘 '{req.algorithm}' 없음. 사용 가능: {list(recommenders.keys())}",
        )
    rec_dict = recommenders[req.algorithm].recommend(
        req.song_id,
        req.top_k + len(req.exclude_song_ids),
    )
    rec_dict = _exclude_song_ids(rec_dict, req.exclude_song_ids)
    return _format(rec_dict, req.algorithm)


@app.post("/api/recommend/user")
def recommend_by_user(req: UserRecommendRequest, authorization: str | None = Header(default=None)):
    """유저 기반 추천"""
    _ensure_app_ready()
    username = _get_current_username(authorization)
    if req.algorithm not in recommenders:
        raise HTTPException(status_code=400, detail="알고리즘 없음")
    user_id = get_or_create_user(username)
    rec_dict = recommenders[req.algorithm].recommend_for_user(
        user_id,
        req.top_k + len(req.exclude_song_ids),
    )
    rec_dict = _exclude_song_ids(rec_dict, req.exclude_song_ids)
    return _format(rec_dict, req.algorithm)


@app.post("/api/search")
def search_by_query(req: SearchRequest):
    """자연어 검색 (BERT / LangChain 계열에서 구현)"""
    _ensure_app_ready()
    if req.algorithm not in recommenders:
        raise HTTPException(status_code=400, detail="알고리즘 없음")
    algo = recommenders[req.algorithm]
    if not hasattr(algo, "search_by_query"):
        raise HTTPException(status_code=400, detail="이 알고리즘은 자연어 검색을 지원하지 않습니다")
    rec_dict = algo.search_by_query(req.query, req.top_k + len(req.exclude_song_ids))
    rec_dict = _exclude_song_ids(rec_dict, req.exclude_song_ids)
    return _format(rec_dict, req.algorithm)


def _format(rec_dict: dict, algorithm: str) -> dict:
    results = []
    for sid, score in sorted(rec_dict.items(), key=lambda x: x[1], reverse=True):
        row = song_df[song_df["song_id"] == sid]
        if not row.empty:
            r = row.iloc[0]
            results.append({
                "song_id":       sid,
                "title":         r["title"],
                "artist":        r["artist"],
                "youtube_url":   r["youtube_url"],
                "thumbnail_url": r["thumbnail_url"],
                "genre":         r["genre"],
                "score":         round(float(score), 4),
            })
    return {"algorithm": algorithm, "recommendations": results}


# ── 인터랙션 API ───────────────────────────────────────────

class InteractionRequest(BaseModel):
    username:    str
    song_id:     str
    action:      str        # "play" | "like" | "skip" | "unlike"
    play_seconds: int = 0


@app.post("/api/interact")
def interact(req: InteractionRequest, authorization: str | None = Header(default=None)):
    if req.action not in ("play", "like", "skip", "unlike"):
        raise HTTPException(status_code=400, detail="action은 play/like/skip/unlike 중 하나")
    username = _get_current_username(authorization)
    user_id = get_or_create_user(username)
    log_interaction(user_id, req.song_id, req.action, req.play_seconds)
    return {"status": "ok"}


_EMPTY_ALGO_NOTICE = (
    "등록된 추천 알고리즘이 없습니다. "
    "Melon 파이프라인으로 audio_features를 채운 뒤 서버를 재시작하거나, "
    "api/main.py의 startup()에서 recommenders에 알고리즘을 등록하세요."
)


@app.get("/api/algorithms")
def list_algorithms():
    _ensure_app_ready()
    keys = list(recommenders.keys())
    body: dict = {"algorithms": keys}
    if not keys:
        body["notice"] = _EMPTY_ALGO_NOTICE
    return body
