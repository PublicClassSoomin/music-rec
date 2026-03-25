from fastapi import APIRouter, HTTPException, Depends
from api.core.security import get_current_username
from data.database import get_or_create_user, log_interaction, get_user_liked_songs
from .schema import InteractionRequest

router = APIRouter(prefix="/api", tags=["Interaction"])

@router.get("/likes")
def list_likes(username: str = Depends(get_current_username)):
    """현재 로그인 사용자의 좋아요 곡 목록"""
    user_id = get_or_create_user(username)
    likes = get_user_liked_songs(user_id)
    return {"likes": likes}

@router.post("/interact")
def interact(req: InteractionRequest, username: str = Depends(get_current_username)):
    """플레이어 동작 기록 (play, like 등)"""
    if req.action not in ("play", "like", "skip", "unlike"):
        raise HTTPException(status_code=400, detail="action은 play/like/skip/unlike 중 하나")
    user_id = get_or_create_user(username)
    log_interaction(user_id, req.song_id, req.action, req.play_seconds)
    return {"status": "ok"}