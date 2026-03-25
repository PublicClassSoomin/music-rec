from fastapi import APIRouter, HTTPException, Depends
from api.core.state import state
from api.core.security import get_current_username
from data.database import get_or_create_user
from .schema import RecommendRequest, UserRecommendRequest, SearchRequest

router = APIRouter(prefix="/api", tags=["Recommend"])

def _format(rec_dict: dict, algorithm: str) -> dict:
    results = []
    for sid, score in sorted(rec_dict.items(), key=lambda x: x[1], reverse=True):
        row = state.song_df[state.song_df["song_id"] == sid]
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

@router.post("/recommend")
def recommend_by_song(req: RecommendRequest):
    """곡 기반 추천"""
    if req.algorithm not in state.recommenders:
        raise HTTPException(
            status_code=400,
            detail=f"알고리즘 '{req.algorithm}' 없음. 사용 가능: {list(state.recommenders.keys())}",
        )
    rec_dict = state.recommenders[req.algorithm].recommend(req.song_id, req.top_k)
    return _format(rec_dict, req.algorithm)

@router.post("/recommend/user")
def recommend_by_user(req: UserRecommendRequest, username: str = Depends(get_current_username)):
    """유저 기반 추천"""
    if req.algorithm not in state.recommenders:
        raise HTTPException(status_code=400, detail="알고리즘 없음")
    user_id = get_or_create_user(username)
    rec_dict = state.recommenders[req.algorithm].recommend_for_user(user_id, req.top_k)
    return _format(rec_dict, req.algorithm)

@router.post("/search")
def search_by_query(req: SearchRequest):
    """자연어 검색"""
    if req.algorithm not in state.recommenders:
        raise HTTPException(status_code=400, detail="알고리즘 없음")
    algo = state.recommenders[req.algorithm]
    if not hasattr(algo, "search_by_query"):
        raise HTTPException(status_code=400, detail="이 알고리즘은 자연어 검색을 지원하지 않습니다")
    rec_dict = algo.search_by_query(req.query, req.top_k)
    return _format(rec_dict, req.algorithm)