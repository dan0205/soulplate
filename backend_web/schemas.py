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

class TasteTestSubmit(BaseModel):
    """취향 테스트 제출"""
    test_type: str  # 'quick' or 'deep'
    answers: List[int]  # [1, 2, 3, 4, 5, ...] 형태의 답변 리스트

class TasteTestResult(BaseModel):
    """취향 테스트 결과"""
    mbti_type: str  # 예: "SAPA"
    type_name: str  # 예: "도파민 추구 미식회장"
    description: str
    recommendations: List[str]
    axis_scores: Optional[dict] = None  # 각 축의 확률
    emoji: Optional[str] = None  # 이모지
    catchphrase: Optional[str] = None  # 캐치프레이즈
    recommend: Optional[List[str]] = None  # 상세 추천 메뉴 & 장소
    avoid: Optional[List[str]] = None  # 피해야 할 식당

class UserStatusResponse(BaseModel):
    """사용자 상태 정보"""
    is_new_user: bool  # 리뷰 개수가 0인지
    review_count: int
    has_taste_test: bool  # 취향 테스트 완료 여부
    should_show_test_popup: bool  # 팝업 표시 여부
    mbti_type: Optional[str] = None  # 취향 테스트 완료 시 MBTI 타입

class UserProfileResponse(BaseModel):
    """사용자 프로필 정보"""
    id: int
    username: str
    review_count: int
    useful: int
    fans: int
    created_at: datetime
    absa_features: Optional[Dict[str, float]] = None
    taste_test_completed: bool = False
    taste_test_type: Optional[str] = None  # 'quick' or 'deep'
    taste_test_mbti_type: Optional[str] = None  # 'ABCD' 형태
    
    class Config:
        from_attributes = True 

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
    absa_food_avg: Optional[float] = None  # ABSA 음식 평균 점수
    absa_service_avg: Optional[float] = None  # ABSA 서비스 평균 점수
    absa_atmosphere_avg: Optional[float] = None  # ABSA 분위기 평균 점수
    ai_prediction: Optional[AIPrediction] = None  # AI 예측 (로그인 사용자)
    # 가게 정보를 클라이언트에 응답할 때 사용 
    
    class Config:
        from_attributes = True

# Review Schemas
class ReviewBase(BaseModel):
    stars: Optional[float] = Field(None, ge=1.0, le=5.0)  # 답글은 별점 없음 (nullable)
    text: str

class ReviewCreate(ReviewBase):
    stars: float = Field(..., ge=1.0, le=5.0)  # 리뷰 작성 시에는 별점 필수
    # 사용자가 새 리뷰를 작성할 때 받는 데이터 형식 

class ReviewUpdate(BaseModel):
    """리뷰 수정 시 사용"""
    stars: Optional[float] = Field(None, ge=1.0, le=5.0)
    text: Optional[str] = None

class ReplyCreate(BaseModel):
    """답글 작성 시 사용 (별점 없음)"""
    text: str

class ReviewResponse(ReviewBase):
    id: int
    user_id: int
    business_id: Optional[int]  # 답글의 경우 비즈니스 없을 수 있음
    created_at: datetime
    username: str  # 리뷰 작성자 이름
    useful: int = 0  # 유용성 점수
    absa_features: Optional[Dict[str, float]] = None  # ABSA 피처
    parent_review_id: Optional[int] = None  # 답글인 경우 부모 리뷰 ID
    reply_count: int = 0  # 이 리뷰에 달린 답글 개수
    # 작성된 리뷰 정보를 응답할 때 사용 
    
    class Config:
        from_attributes = True

class BusinessInfo(BaseModel):
    """리뷰에 포함될 음식점 정보"""
    business_id: str
    name: str
    
class UserReviewResponse(ReviewBase):
    """사용자 프로필에서 보여줄 리뷰 정보 (음식점 정보 포함)"""
    id: int
    user_id: int
    business_id: int
    created_at: datetime
    useful: int = 0
    business: BusinessInfo  # 음식점 정보
    
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
    
