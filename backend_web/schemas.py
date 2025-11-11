"""
Pydantic 스키마 정의
"""
#Pydantic 라이브러리를 사용해서 API 서버에서 사용할 데이터 스키마들을 정의한 것이다
# 데이터 유효성 검사: 클라이언트로부터 API 요청이 들어올 때, 
# 데이터가 우리가 정의한 형식에 맞는지 자동으로 검사한다
# 데이터 직렬화: 서버가 클라이언트에 응답을 보낼 때, 파이썬 객체를 이 스키마에 맞춰
# JSON 형식으로 변환해준다 

from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
# BaseModel = 모든 스키마가 상속받는 Pydantic의 기본 클래스
# EmailStr = 문자열이 유효한 이메일 형식인지 검사하는 타입
# Optional = 해당 필드가 T 타입이거나 None일 수 있음
# Filed = 필드의 기본값 외에 추가적인 유효성 검사 규칙을 설정  

# User Schemas
# 사용자 인증 및 정보 조회를 위해 사용된다 
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str
    # 회원가입 API를 요청할 때 사용 
    # UserBase를 상속받아서 username과 email을 포함하고, password를 받는다 

class UserLogin(BaseModel):
    username: str
    password: str
    # 로그인 API를 요청할 때 사용 

class UserResponse(UserBase):
    id: int
    created_at: datetime
    # 사용자 정보를 클라이언트에 응답할 때 사용 
    
    class Config:
        from_attributes = True
        # 매우 중요한 설정이다
        # SQLAlchemy같은 ORM 객체를 받았을 때, user.id가 아닌 user['id']로 접근하려다
        # 오류가 나는 걸 방지한다 

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str
    # 사용자가 로그인에 성공했을 때, 클라이언트에게 반환할 액세스 토큰의 형식 

class TokenData(BaseModel):
    username: Optional[str] = None
    # 클라이언트가 요청 헤더에 담아 보낸 access_token을 decode했을 때,
    # 그 안에 들어있는 데이터의 형식 

# Business Schemas
class BusinessBase(BaseModel):
    business_id: str
    name: str
    categories: Optional[str] = None
    stars: Optional[float] = None
    review_count: int = 0

class ABSAFeature(BaseModel):
    """ABSA 특징"""
    aspect: str  # 예: "맛"
    sentiment: str  # "긍정", "부정", "중립"
    score: float  # 0~1

class AIPrediction(BaseModel):
    """AI 예측 별점"""
    deepfm_rating: float
    multitower_rating: Optional[float] = None  # Multi-Tower 사용 불가 시 None
    ensemble_rating: float

class BusinessResponse(BusinessBase):
    id: int
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    absa_features: Optional[Dict[str, float]] = None  # 전체 ABSA 피처 (Detail용)
    top_features: Optional[List[ABSAFeature]] = None  # 상위 특징 (리스트용)
    ai_prediction: Optional[AIPrediction] = None  # AI 예측 (로그인 사용자)
    # 가게 정보를 클라이언트에 응답할 때 사용 
    
    class Config:
        from_attributes = True

# Review Schemas
class ReviewBase(BaseModel):
    stars: float = Field(..., ge=1.0, le=5.0)
    text: str

class ReviewCreate(ReviewBase):
    pass
    # 사용자가 새 리뷰를 작성할 때 받는 데이터 형식 

class ReviewResponse(ReviewBase):
    id: int
    user_id: int
    business_id: int
    created_at: datetime
    username: str  # 리뷰 작성자 이름
    absa_features: Optional[Dict[str, float]] = None  # ABSA 피처
    # 작성된 리뷰 정보를 응답할 때 사용 
    
    class Config:
        from_attributes = True

# Recommendation Schemas
class RecommendationRequest(BaseModel):
    top_k: int = Field(default=10, ge=1, le=50)
    # 추천을 요청할 때 받는 파라미터 형식 
    # top_k 필드에서 기본값은 10개, 1개 미만이나 50개 초과로 요청하면 오류 

class RecommendationItem(BaseModel):
    business: BusinessResponse
    score: float
    # 추천 결과 목록에 포함될 개별 항목 1개의 형식 
    # 가게 정보와 추천 점수를 묶어서 표현 

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    user_id: int
    # 추천 api의 최종 응답 형식

# Business List with Pagination
class BusinessListResponse(BaseModel):
    businesses: List[BusinessResponse]
    total: int
    skip: int
    limit: int 
    
