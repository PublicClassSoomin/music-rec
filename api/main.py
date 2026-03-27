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


@app.on_event("startup")
async def startup():
    global song_df
    init_db()
    song_df = get_all_songs()
    print(f"[API] {len(song_df)}곡 로드 완료")

    # FAISS 인덱스: 이 블록 안의 지역 변수 faiss_index를 알고리즘 생성 시 그대로 넘깁니다.
    # 다른 함수로 분리할 경우 faiss_index를 인자로 전달하거나 모듈 전역에 보관하세요.
    faiss_index = None
    try:
        from data.faiss_index import MusicFaissIndex

        faiss_index = MusicFaissIndex()
        faiss_index.build(song_df)
    except Exception as e:
        print(f"[API] FAISS 초기화 실패: {e}")

    if faiss_index is not None and getattr(faiss_index, "is_built", False):
        from algorithms.faiss_cbf import FaissContentRecommender

        recommenders["faiss_cbf"] = FaissContentRecommender(faiss_index)
        recommenders["faiss_cbf"].fit(song_df)
        print("[API] 샘플 알고리즘 등록: faiss_cbf")

    # ── 팀원 추가 등록 예시 (같은 startup() 안에서 faiss_index 사용) ──
    # from algorithms.my_algo import MyRecommender
    # recommenders["my_algo"] = MyRecommender(faiss_index)
    # recommenders["my_algo"].fit(song_df)
    # ────────────────────────────────────────────────────────────────

    # ── LLM Inference 추천기 (GPT + LangGraph → FAISS 검색) ──
    if faiss_index is not None and getattr(faiss_index, "is_built", False):
        try:
            from algorithms.llm_inference import LLMInferenceRecommender

            recommenders["llm_inference"] = LLMInferenceRecommender(faiss_index)
            recommenders["llm_inference"].fit(song_df)
            print("[API] LLM 알고리즘 등록: llm_inference")
        except Exception as e:
            print(f"[API] LLM 추천기 등록 실패 (OPENAI_API_KEY 확인): {e}")


# ── 웹 UI ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── 곡 API ────────────────────────────────────────────────

@app.get("/api/songs")
def list_songs(limit: int = 30, genre: str = None):
    if song_df is None:
        raise HTTPException(status_code=503, detail="서버 초기화 중입니다. 잠시 후 다시 시도하세요.")
    df = song_df.copy()
    if genre:
        df = df[df["genre"] == genre]
    return df.head(limit)[
        ["song_id", "title", "artist", "youtube_url", "thumbnail_url", "genre"]
    ].to_dict(orient="records")


@app.get("/api/songs/{song_id}")
def get_song_detail(song_id: str):
    song = get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="곡을 찾을 수 없습니다")
    return song


# ── 추천 API ──────────────────────────────────────────────

class RecommendRequest(BaseModel):
    song_id:   str
    algorithm: str = "default"
    top_k:     int = 10


class UserRecommendRequest(BaseModel):
    username:  str
    algorithm: str = "default"
    top_k:     int = 10


class SearchRequest(BaseModel):
    query:     str
    algorithm: str = "default"
    top_k:     int = 10


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
    if req.algorithm not in recommenders:
        raise HTTPException(
            status_code=400,
            detail=f"알고리즘 '{req.algorithm}' 없음. 사용 가능: {list(recommenders.keys())}",
        )
    rec_dict = recommenders[req.algorithm].recommend(req.song_id, req.top_k)
    return _format(rec_dict, req.algorithm)


@app.post("/api/recommend/user")
def recommend_by_user(req: UserRecommendRequest, authorization: str | None = Header(default=None)):
    """유저 기반 추천"""
    username = _get_current_username(authorization)
    if req.algorithm not in recommenders:
        raise HTTPException(status_code=400, detail="알고리즘 없음")
    user_id = get_or_create_user(username)
    rec_dict = recommenders[req.algorithm].recommend_for_user(user_id, req.top_k)
    return _format(rec_dict, req.algorithm)


@app.post("/api/search")
def search_by_query(req: SearchRequest):
    """자연어 검색 (BERT / LangChain 계열에서 구현)"""
    if req.algorithm not in recommenders:
        raise HTTPException(status_code=400, detail="알고리즘 없음")
    algo = recommenders[req.algorithm]
    if not hasattr(algo, "search_by_query"):
        raise HTTPException(status_code=400, detail="이 알고리즘은 자연어 검색을 지원하지 않습니다")
    rec_dict = algo.search_by_query(req.query, req.top_k)
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
    keys = list(recommenders.keys())
    body: dict = {"algorithms": keys}
    if not keys:
        body["notice"] = _EMPTY_ALGO_NOTICE
    return body
