"""
Pydantic 스키마 정의
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class RecommendRequest(BaseModel):
    """추천 요청 스키마"""
    user_id: str = Field(..., description="User ID")
    user_features: Optional[Dict[str, float]] = Field(
        default=None, 
        description="User features (age, review_count, useful, average_stars)"
    )
    recent_business_ids: List[str] = Field(
        default_factory=list, 
        description="Recently interacted business IDs"
    )
    context: Optional[Dict] = Field(default=None, description="Additional context")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    
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
        }

class RecommendResponse(BaseModel):
    """추천 응답 스키마"""
    recommendations: List[str] = Field(..., description="Recommended business IDs")
    scores: List[float] = Field(..., description="Similarity scores")
    user_id: str = Field(..., description="User ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "recommendations": ["biz1", "biz2", "biz3"],
                "scores": [0.95, 0.92, 0.89],
                "user_id": "user123"
            }
        }

class HealthResponse(BaseModel):
    """Health check 응답"""
    status: str
    model_loaded: bool
    index_loaded: bool
    num_items: int

