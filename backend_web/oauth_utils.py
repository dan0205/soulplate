"""
OAuth 관련 유틸리티 함수
"""

import re
from sqlalchemy.orm import Session
import models


def sanitize_username(oauth_name: str, email: str, db: Session) -> str:
    """
    OAuth 이름을 username 제약조건에 맞게 변환
    
    제약조건:
    - 길이: 2-50자
    - 허용 문자: 영문, 한글, 숫자, _, -, 공백
    
    Args:
        oauth_name: OAuth 제공자로부터 받은 이름
        email: 사용자 이메일 (fallback용)
        db: 데이터베이스 세션
    
    Returns:
        str: 제약조건을 만족하고 중복되지 않는 username
    """
    
    # 1. 특수문자 제거 (허용: 영문, 한글, 숫자, _, -, 공백)
    cleaned = re.sub(r'[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ_\- ]', '', oauth_name)
    
    # 2. 연속된 공백을 하나로
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 3. 앞뒤 공백 제거
    cleaned = cleaned.strip()
    
    # 4. 길이가 너무 짧으면 email prefix 사용
    if len(cleaned) < 2:
        email_prefix = email.split('@')[0]
        cleaned = re.sub(r'[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ_\- ]', '', email_prefix)
        cleaned = cleaned.strip()
    
    # 5. 여전히 짧으면 "user" 사용
    if len(cleaned) < 2:
        cleaned = "user"
    
    # 6. 길이가 50자를 초과하면 자르기
    if len(cleaned) > 50:
        cleaned = cleaned[:50].strip()
    
    # 7. 중복 확인 및 처리
    base_username = cleaned
    counter = 1
    max_attempts = 100  # 무한 루프 방지
    
    while counter < max_attempts:
        # 중복 확인
        existing = db.query(models.User).filter(
            models.User.username == cleaned
        ).first()
        
        if not existing:
            # 중복 없음 - 사용 가능
            return cleaned
        
        # 중복 있음 - 숫자 추가
        suffix = str(counter)
        max_base_length = 50 - len(suffix)
        cleaned = base_username[:max_base_length] + suffix
        counter += 1
    
    # 최대 시도 횟수 초과 (거의 발생하지 않음)
    raise ValueError(f"Could not generate unique username for: {oauth_name}")


def extract_oauth_user_info(userinfo: dict) -> dict:
    """
    OAuth 제공자로부터 받은 사용자 정보 추출
    
    Args:
        userinfo: OAuth 제공자의 userinfo 응답
    
    Returns:
        dict: 표준화된 사용자 정보
            - oauth_id: OAuth 제공자의 고유 ID
            - email: 이메일
            - name: 이름
            - picture: 프로필 사진 URL
    """
    return {
        'oauth_id': userinfo.get('sub'),  # Google sub
        'email': userinfo.get('email'),
        'name': userinfo.get('name', userinfo.get('email', '').split('@')[0]),
        'picture': userinfo.get('picture')
    }

