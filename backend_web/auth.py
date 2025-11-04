"""
인증 및 보안 유틸리티
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from datetime import timezone
from dotenv import load_dotenv
import os

from backend_web.database import get_db
from backend_web import models

# .env 파일 로드
load_dotenv()

# 설정
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production-12345678")
# JWT 토큰을 서명하고 검증할 때 사용하는 비밀 키
ALGORITHM = os.getenv("ALGORITHM", "HS256")
# 토큰 서명에 사용할 해시 알고리즘 
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__ident="2b")
# passlib 라이브러리를 사용해서 비밀번호 해시 방식을 설정한다 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")
# FastAPI의 보안 유틸리티로 클라이언트가 Authorization: Bearer <token> 헤더를 통해
# 토큰을 전송해야 함을 알려주는 의존성이다 
# tokenURL은 token을 발급받을수있는 API 경로이다 

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """비밀번호 검증"""
    return pwd_context.verify(plain_password, hashed_password)
    # 사용자가 입력한 비밀번호와 데이터베이스에 저장된 비밀번호가 일치하는지 검증한다 
    # hashed... = 데이터베이스에 저장된 해시값 
    # bool = 비밀번호가 일치하면 true, 아니면 false 

def get_password_hash(password: str) -> str:
    """비밀번호 해시 생성"""
    # bcrypt has a 72 byte limit
    if len(password.encode('utf-8')) > 72:
        password = password[:72]
    return pwd_context.hash(password)
    # 사용자의 비밀번호를 받아 해시값으로 변환, 이 해시값을 데이터베이스에 저장 

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """JWT 액세스 토큰 생성"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
    # 사용자가 성공적으로 로그인했을때, 사용자를 식별하는 JWT 액세스 토큰을 생성한다 

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """현재 인증된 사용자 가져오기"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user
    # FastAPI의 의존성이 목적이다 
    # 일반 함수처럼 직접 호출하는 것이 아니라 인증이 필요한 API 엔드포인트에서 사용된다 
    # 클라이언트가 보낸 JWT 토큰을 검증하고, 유효하다면 해당 토큰의 사용자 정보를 DB에서 찾아 반환한다

