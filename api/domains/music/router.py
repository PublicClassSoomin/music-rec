from fastapi import APIRouter, HTTPException
from api.core.state import state
from data.database import get_song

router = APIRouter(prefix="/api", tags=["Music"])

@router.get("/songs")
def list_songs(limit: int = 30, genre: str = None):
    if state.song_df is None:
        raise HTTPException(status_code=503, detail="서버 초기화 중입니다. 잠시 후 다시 시도하세요.")
    df = state.song_df.copy()
    if genre:
        df = df[df["genre"] == genre]
    return df.head(limit)[
        ["song_id", "title", "artist", "youtube_url", "thumbnail_url", "genre"]
    ].to_dict(orient="records")

@router.get("/songs/{song_id}")
def get_song_detail(song_id: str):
    song = get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="곡을 찾을 수 없습니다")
    return song

_EMPTY_ALGO_NOTICE = (
    "등록된 추천 알고리즘이 없습니다. "
    "Melon 파이프라인으로 audio_features를 채운 뒤 서버를 재시작하거나, "
    "api/main.py의 startup()에서 recommenders에 알고리즘을 등록하세요."
)

@router.get("/algorithms")
def list_algorithms():
    keys = list(state.recommenders.keys())
    body: dict = {"algorithms": keys}
    if not keys:
        body["notice"] = _EMPTY_ALGO_NOTICE
    return body