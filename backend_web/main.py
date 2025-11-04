"""
Tier 2: Web Backend Server
사용자 인증, DB 관리, 모델 API 게이트웨이
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List
import httpx
import logging

from backend_web import models, schemas, auth
from backend_web.database import engine, get_db

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 데이터베이스 테이블 생성
models.Base.metadata.create_all(bind=engine)

# FastAPI 앱
app = FastAPI(
    title="Two-Tower Recommendation Web API",
    description="Web backend for Two-Tower recommendation system",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model API URL
MODEL_API_URL = "http://localhost:8001"

# Root
@app.get("/")
async def root():
    return {
        "message": "Two-Tower Recommendation Web API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Health Check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/api/auth/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """회원가입"""
    # 중복 확인
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # 사용자 생성
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        age=user.age,
        gender=user.gender
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    logger.info(f"New user registered: {user.username}")
    return db_user

@app.post("/api/auth/login", response_model=schemas.Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """로그인"""
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=schemas.UserResponse)
async def get_current_user_info(current_user: models.User = Depends(auth.get_current_user)):
    """현재 사용자 정보"""
    return current_user

# ============================================================================
# Business Endpoints
# ============================================================================

@app.get("/api/businesses", response_model=List[schemas.BusinessResponse])
async def get_businesses(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """비즈니스 목록 조회"""
    businesses = db.query(models.Business).offset(skip).limit(limit).all()
    return businesses

@app.get("/api/businesses/{business_id}", response_model=schemas.BusinessResponse)
async def get_business(business_id: str, db: Session = Depends(get_db)):
    """비즈니스 상세 정보 조회"""
    business = db.query(models.Business).filter(
        models.Business.business_id == business_id
    ).first()
    
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    
    return business

# ============================================================================
# Review Endpoints
# ============================================================================

@app.get("/api/businesses/{business_id}/reviews", response_model=List[schemas.ReviewResponse])
async def get_reviews(
    business_id: str,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """비즈니스 리뷰 목록 조회"""
    business = db.query(models.Business).filter(
        models.Business.business_id == business_id
    ).first()
    
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    
    reviews = db.query(models.Review).filter(
        models.Review.business_id == business.id
    ).offset(skip).limit(limit).all()
    
    return reviews

@app.post("/api/businesses/{business_id}/reviews", response_model=schemas.ReviewResponse, status_code=status.HTTP_201_CREATED)
async def create_review(
    business_id: str,
    review: schemas.ReviewCreate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """리뷰 작성"""
    business = db.query(models.Business).filter(
        models.Business.business_id == business_id
    ).first()
    
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    
    # 리뷰 생성
    db_review = models.Review(
        user_id=current_user.id,
        business_id=business.id,
        stars=review.stars,
        text=review.text
    )
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    
    logger.info(f"New review created by user {current_user.username} for business {business_id}")
    return db_review

# ============================================================================
# Recommendation Endpoint
# ============================================================================

@app.get("/api/recommendations", response_model=schemas.RecommendationResponse)
async def get_recommendations(
    top_k: int = 10,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """개인화 추천"""
    logger.info(f"Recommendation request for user: {current_user.username}")
    
    try:
        # 사용자의 최근 리뷰한 비즈니스 ID 가져오기
        recent_reviews = db.query(models.Review).filter(
            models.Review.user_id == current_user.id
        ).order_by(models.Review.created_at.desc()).limit(5).all()
        
        recent_business_ids = [
            db.query(models.Business).filter(models.Business.id == review.business_id).first().business_id
            for review in recent_reviews
        ]
        
        # Model API 호출
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MODEL_API_URL}/recommend",
                json={
                    "user_id": current_user.username,
                    "user_features": {
                        "age": float(current_user.age) if current_user.age else 30.0,
                        "review_count": 50.0,  # 실제로는 DB에서 계산
                        "useful": 20.0,
                        "average_stars": 4.0
                    },
                    "recent_business_ids": recent_business_ids,
                    "top_k": top_k
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to get recommendations from model API"
                )
            
            model_response = response.json()
        
        # Business 정보와 함께 반환
        recommendations = []
        for business_id, score in zip(
            model_response["recommendations"],
            model_response["scores"]
        ):
            business = db.query(models.Business).filter(
                models.Business.business_id == business_id
            ).first()
            
            if business:
                recommendations.append(
                    schemas.RecommendationItem(
                        business=business,
                        score=score
                    )
                )
        
        logger.info(f"Returned {len(recommendations)} recommendations for user {current_user.username}")
        
        return schemas.RecommendationResponse(
            recommendations=recommendations,
            user_id=current_user.id
        )
        
    except httpx.RequestError as e:
        logger.error(f"Error connecting to model API: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model API is unavailable"
        )
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

