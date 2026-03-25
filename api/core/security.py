from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Header
import jwt
from utils.config import JWT_SECRET, JWT_EXPIRE_HOURS

def create_token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {"sub": username, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def get_current_username(authorization: str | None = Header(default=None)) -> str:
    """FastAPI Depends() 주입용 함수: 헤더에서 토큰을 추출해 검증하고 username을 반환"""
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