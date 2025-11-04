"""
Pydantic 스키마 정의
"""

from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime

# User Schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str
    age: Optional[int] = None
    gender: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    age: Optional[int]
    gender: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Business Schemas
class BusinessBase(BaseModel):
    business_id: str
    name: str
    categories: Optional[str] = None
    stars: Optional[float] = None
    review_count: int = 0

class BusinessResponse(BusinessBase):
    id: int
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    is_open: bool
    
    class Config:
        from_attributes = True

# Review Schemas
class ReviewBase(BaseModel):
    stars: int = Field(..., ge=1, le=5)
    text: str

class ReviewCreate(ReviewBase):
    pass

class ReviewResponse(ReviewBase):
    id: int
    user_id: int
    business_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Recommendation Schemas
class RecommendationRequest(BaseModel):
    top_k: int = Field(default=10, ge=1, le=50)

class RecommendationItem(BaseModel):
    business: BusinessResponse
    score: float

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    user_id: int

