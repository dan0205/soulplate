"""
Pydantic 스키마 정의
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class RecommendRequest(BaseModel):
    """추천 요청 스키마"""
    # 웹 백엔드가 모델 서버에게 추천을 요청할 때 보내는 입력 데이터의 형식을 의미한다 
    # 즉, 추천을 요청하기 위해 필요한 데이터 
    user_id: str = Field(..., description="User ID")
    # 추천받을 대상이 되는 사용자의 ID 
    user_features: Optional[Dict[str, float]] = Field(
        default=None, 
        description="User features (age, review_count, useful, average_stars)"
    )
    # 선택사항, 모델이 추천에 참고할 사용자의 정량적 특징들. 나이, 리뷰 수를 dict로 받는다 
    recent_business_ids: List[str] = Field(
        default_factory=list, # 이 값을 제공하지 않는다면 빈 리스트를 기본값으로 설정해라 
        description="Recently interacted business IDs"
    )
    # 사용자가 최근 상호작용한 가게 ID의 리스트 
    context: Optional[Dict] = Field(default=None, description="Additional context")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    # 상위 몇 개의 추천을 받을지 지정한다 
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "user_features": {
                    "age": 30.0,
                    "review_count": 50.0,
                    "useful": 20.0,
                    "average_stars": 4.5
                },
                "recent_business_ids": ["biz1", "biz2"],
                "context": {},
                "top_k": 10
            }
        } # API 문서에 보여줄 예시 JSON 

class RecommendResponse(BaseModel): 
    """추천 응답 스키마"""
    # 모델 서버가 추천 계산을 완료한 후, 웹 백엔드에게 반환하는 출력 데이터의 형식 
    recommendations: List[str] = Field(..., description="Recommended business IDs")
    # 추천받은 가게 ID의 리스트  
    scores: List[float] = Field(..., description="Similarity scores")
    # 추천 점수의 리스트 
    user_id: str = Field(..., description="User ID")
    # 추천을 요청한 사용자의 ID 
    class Config:
        json_schema_extra = {
            "example": {
                "recommendations": ["biz1", "biz2", "biz3"],
                "scores": [0.95, 0.92, 0.89],
                "user_id": "user123"
            }
        } # API 문서에 보여줄 예시 JSON 

class HealthResponse(BaseModel):
    """Health check 응답"""
    status: str
    model_loaded: bool
    index_loaded: bool
    num_items: int
    # 모델 API 서버의 상태를 확인하는 /health 같은 엔드포인트의 응답 형식이다 

