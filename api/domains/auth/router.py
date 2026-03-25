from fastapi import APIRouter, HTTPException, Depends
from api.core.security import create_token, get_current_username
from data.database import create_user, authenticate_user
from .schema import AuthRequest

router = APIRouter(prefix="/api", tags=["Auth"])

@router.post("/signup")
def signup(req: AuthRequest):
    username = req.username.strip()
    if len(username) < 2 or len(req.password) < 4:
        raise HTTPException(status_code=400, detail="아이디 2자 이상, 비밀번호 4자 이상이 필요합니다")
    if not create_user(username, req.password):
        raise HTTPException(status_code=409, detail="이미 존재하는 아이디입니다")
    token = create_token(username)
    return {"status": "ok", "username": username, "access_token": token}

@router.post("/login")
def login(req: AuthRequest):
    username = req.username.strip()
    if not authenticate_user(username, req.password):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다")
    token = create_token(username)
    return {"status": "ok", "username": username, "access_token": token}

@router.get("/me")
def me(username: str = Depends(get_current_username)):
    return {"username": username}