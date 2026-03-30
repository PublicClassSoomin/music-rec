from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timedelta, timezone
import jwt
from data.database import (
    init_db, get_all_songs, get_song,
    create_user, authenticate_user,
    get_or_create_user, log_interaction,
    get_user_liked_songs,
)
from utils.config import JWT_SECRET, JWT_EXPIRE_HOURS
from typing import Any
from pydantic import Field

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

        from algorithms.hybrid_faiss_cooc import HybridFaissCoocRecommender

        # weight_content / weight_cooc 는 데이터 규모에 맞게 조정해야 함.
        # (좋아요 로그가 적을 때는 content를 더 크게: 예 0.75 / 0.25)
        recommenders["hybrid_faiss_cooc"] = HybridFaissCoocRecommender(
            faiss_index,
            weight_content=0.55,
            weight_cooc=0.45,
            faiss_candidate_multiplier=4,
        )
        recommenders["hybrid_faiss_cooc"].fit(song_df)
        print("[API] 알고리즘 등록: hybrid_faiss_cooc")

    # ── 팀원 추가 등록 예시 (같은 startup() 안에서 faiss_index 사용) ──
    # from algorithms.my_algo import MyRecommender
    # recommenders["my_algo"] = MyRecommender(faiss_index)
    # recommenders["my_algo"].fit(song_df)
    # ────────────────────────────────────────────────────────────────


# ── 웹 UI ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """브라우저 기본 요청으로 인한 404 방지"""
    return FileResponse("static/favicon.svg", media_type="image/svg+xml")


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
    options: dict[str, Any] | None = None

class UserRecommendRequest(BaseModel):
    username:  str
    algorithm: str = "default"
    top_k:     int = 10
    options: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query:     str
    algorithm: str = "default"
    top_k:     int = 10
    options: dict[str, Any] | None = None


class AuthRequest(BaseModel):
    username: str
    password: str

class HybridOptions(BaseModel):
    mode: str = "advanced" # simple | advanced
    weights: dict[str, float] = Field(default_factory=lambda: {"content": 0.55, "cooc": 0.45})
    threshold: float = 0.0
    filters: dict[str, str] = Field(default_factory=dict)
    use_llm_search: bool = False

class EvalRunRequest(BaseModel):
    """options 안에 넣어도 되고, eval_* 는 본문 최상위로내도 됨(누락 방지)."""

    model_config = ConfigDict(extra="allow")

    algorithm: str
    options: dict[str, Any] | None = None
    eval_only_current_user: bool | None = None


def _option_bool(v: Any) -> bool:
    """쿼리스트링/구 클라이언트 대비: eval_only_current_user 등을 안전하게 bool 로."""
    if v is True or v == 1:
        return True
    if v is False or v == 0:
        return False
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
    return bool(v)


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
    uid = get_or_create_user(username)
    return {"status": "ok", "username": username, "access_token": token, "user_id": uid}


@app.post("/api/login")
def login(req: AuthRequest):
    username = req.username.strip()
    if not authenticate_user(username, req.password):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다")
    token = _create_token(username)
    uid = get_or_create_user(username)
    return {"status": "ok", "username": username, "access_token": token, "user_id": uid}


@app.get("/api/me")
def me(authorization: str | None = Header(default=None)):
    username = _get_current_username(authorization)
    uid = get_or_create_user(username)
    return {"username": username, "user_id": uid}


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
    algo = recommenders[req.algorithm]
    if hasattr(algo, "recommend_with_options"):
        rec_dict = algo.recommend_with_options(req.song_id, req.top_k, req.options or {})
    else:
        rec_dict = algo.recommend(req.song_id, req.top_k)
    return _format(rec_dict, req.algorithm)


@app.post("/api/recommend/user")
def recommend_by_user(req: UserRecommendRequest, authorization: str | None = Header(default=None)):
    """유저 기반 추천"""
    username = _get_current_username(authorization)
    if req.algorithm not in recommenders:
        raise HTTPException(status_code=400, detail="알고리즘 없음")
    user_id = get_or_create_user(username)
    algo = recommenders[req.algorithm]
    if hasattr(algo, "recommend_for_user_with_options"):
        rec_dict = algo.recommend_for_user_with_options(user_id, req.top_k, req.options or {})
    else:
        rec_dict = algo.recommend_for_user(user_id, req.top_k)
    return _format(rec_dict, req.algorithm)


@app.post("/api/search")
def search_by_query(req: SearchRequest):
    """자연어 검색 (BERT / LangChain 계열에서 구현)"""
    if req.algorithm not in recommenders:
        raise HTTPException(status_code=400, detail="알고리즘 없음")
    algo = recommenders[req.algorithm]
    if hasattr(algo, "search_by_query_with_options"):
        rec_dict = algo.search_by_query_with_options(req.query, req.top_k, req.options or {})
    elif hasattr(algo, "search_by_query"):
        rec_dict = algo.search_by_query(req.query, req.top_k)
    else: 
        raise HTTPException(status_code=400, detail="이 알고리즘은 자연어 검색을 지원하지 않습니다")
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

@app.post("/api/eval/run")
def run_eval(
    req: EvalRunRequest,
    authorization: str | None = Header(default=None),
):
    if req.algorithm not in recommenders:
        raise HTTPException(status_code=400, detail=f"알고리즘 '{req.algorithm}' 없음")
    
    algo = recommenders[req.algorithm]

    # 요청 옵션 파싱 (최상위 eval_* 가 있으면 options 위에 덮어씀)
    opts = dict(req.options or {})
    if req.eval_only_current_user is not None:
        opts["eval_only_current_user"] = req.eval_only_current_user
    k_list = opts.get("k_list", [5, 10, 20])
    if not isinstance(k_list, list) or not k_list:
        k_list = [5, 10, 20]

    # 너무 큰 K나 음수 방어
    k_list = [int(k) for k in k_list if isinstance(k, (int, float)) and int(k) > 0]
    if not k_list:
        k_list = [5, 10, 20]

    max_cases = int(opts.get("max_cases", 0))
    run_variants = bool(opts.get("run_variants", True))

    only_me = _option_bool(opts.get("eval_only_current_user"))
    current_uid: int | None = None
    if only_me:
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="현재 계정만 곡 기반 평가는 로그인이 필요합니다.",
            )
        current_uid = get_or_create_user(_get_current_username(authorization))
    filter_uid: int | None = current_uid if only_me else None

    try:
        from evaluation.offline_eval import (
            eval_recommend_case_counts,
            hybrid_eval_weight_columns,
            run_offline_eval_for_algorithm,
            run_offline_eval_variants,
        )

        if run_variants:
            result = run_offline_eval_variants(
                algo,
                base_options=opts,
                k_list=k_list,
                max_cases=max_cases,
                filter_user_id=filter_uid,
            )
            body: dict[str, Any] = {
                "algorithm": req.algorithm,
                "mode": "variants",
                "k_list": k_list,
                "options_used": opts,
                "rows": result["rows"],
                "summary_rows": result["summary_rows"],
                "summary": result["summary"],
                "eval_only_current_user": only_me,
                "eval_filter_user_id": filter_uid,
                "eval_recommend_pool_all_users": result.get("eval_recommend_pool_all_users"),
                "eval_recommend_pool_filtered_user": result.get("eval_recommend_pool_filtered_user"),
                "eval_recommend_cases_evaluated": result.get("eval_recommend_cases_evaluated"),
            }
            if req.algorithm.startswith("hybrid"):
                body["hybrid_sidebar_snapshot"] = {
                    "mode": opts.get("mode"),
                    "weights": opts.get("weights"),
                    "threshold": opts.get("threshold"),
                    "use_llm_search": opts.get("use_llm_search"),
                }
            return body

        # 단일 옵션으로 1회 평가 (run_variants=False)
        single = run_offline_eval_for_algorithm(
            algo,
            base_options=opts,
            k_list=k_list,
            max_cases=max_cases,
            filter_user_id=filter_uid,
        )
        wcols = hybrid_eval_weight_columns(opts)
        if wcols:
            vm = str(opts.get("mode") or "")
            vllm = bool(opts.get("use_llm_search"))
            for r in single["rows"]:
                r.update(wcols)
                r["variant_mode"] = vm
                r["variant_use_llm_search"] = vllm
            summary_row = {
                **wcols,
                "variant_mode": vm,
                "variant_use_llm_search": vllm,
                "n_cases": single["n_cases"],
                **single["summary"],
            }
        else:
            summary_row = {
                "variant": "single",
                "n_cases": single["n_cases"],
                **single["summary"],
            }
        _ec = eval_recommend_case_counts(max_cases, filter_uid)
        single_body: dict[str, Any] = {
            "algorithm": req.algorithm,
            "mode": "single",
            "k_list": k_list,
            "options_used": opts,
            "rows": single["rows"],
            "summary_rows": [summary_row],
            "summary": single["summary"],
            "eval_only_current_user": only_me,
            "eval_filter_user_id": filter_uid,
            **_ec,
        }
        if req.algorithm.startswith("hybrid"):
            single_body["hybrid_sidebar_snapshot"] = {
                "mode": opts.get("mode"),
                "weights": opts.get("weights"),
                "threshold": opts.get("threshold"),
                "use_llm_search": opts.get("use_llm_search"),
            }
        return single_body
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"평가 실행 실패: {e}")

