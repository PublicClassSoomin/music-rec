from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from data.database import init_db, get_all_songs
from api.core.state import state

# 라우터 임포트
from api.domains.auth.router import router as auth_router
from api.domains.music.router import router as music_router
from api.domains.recommend.router import router as recommend_router
from api.domains.interaction.router import router as interaction_router

app = FastAPI(title="🎵 AI 음악 추천 시스템", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup():
    init_db()
    state.song_df = get_all_songs()
    print(f"[API] {len(state.song_df)}곡 로드 완료")

    # FAISS 인덱스 빌드 및 초기화
    faiss_index = None
    try:
        from data.faiss_index import MusicFaissIndex
        faiss_index = MusicFaissIndex()
        faiss_index.build(state.song_df)
    except Exception as e:
        print(f"[API] FAISS 초기화 실패: {e}")

    # 기존 샘플 알고리즘 등록
    if faiss_index is not None and getattr(faiss_index, "is_built", False):
        from algorithms.faiss_cbf import FaissContentRecommender
        state.recommenders["faiss_cbf"] = FaissContentRecommender(faiss_index)
        state.recommenders["faiss_cbf"].fit(state.song_df)
        print("[API] 샘플 알고리즘 등록: faiss_cbf")

    # ── 팀원 추가 알고리즘 등록은 이곳에서 진행! ──

# 라우터 연결
app.include_router(auth_router)
app.include_router(music_router)
app.include_router(recommend_router)
app.include_router(interaction_router)

@app.get("/")
async def root():
    return FileResponse("static/index.html")