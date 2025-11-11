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

# ============================================================================
# ABSA 헬퍼 함수
# ============================================================================

def get_top_absa_features(absa_dict, top_k=5):
    """상위 ABSA 특징 추출"""
    if not absa_dict:
        return []
    
    # aspect_sentiment별로 점수 정리
    features = []
    for key, score in absa_dict.items():
        parts = key.rsplit('_', 1)  # 마지막 _를 기준으로 분리
        if len(parts) == 2:
            aspect, sentiment = parts
            features.append({
                'aspect': aspect,
                'sentiment': sentiment,
                'score': float(score)
            })
    
    # 긍정 특징 상위 3개
    positive = sorted([f for f in features if f['sentiment'] == '긍정'], 
                     key=lambda x: x['score'], reverse=True)[:3]
    
    # 부정 특징 상위 2개
    negative = sorted([f for f in features if f['sentiment'] == '부정'], 
                     key=lambda x: x['score'], reverse=True)[:2]
    
    # 합치기
    top_features = positive + negative
    
    return [schemas.ABSAFeature(**f) for f in top_features]

async def get_ai_prediction(user: models.User, business: models.Business):
    """AI 예측 별점 가져오기"""
    try:
        # ABSA features 가져오기 (직접 접근)
        user_absa = user.absa_features or {}
        business_absa = business.absa_features or {}
        
        # backend_model API 호출
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8001/predict_rating",
                json={
                    "user_data": {
                        "review_count": user.review_count,
                        "useful": user.useful,
                        "compliment": user.compliment,
                        "fans": user.fans,
                        "average_stars": user.average_stars,
                        "yelping_since_days": user.yelping_since_days,
                        "absa_features": user_absa
                    },
                    "business_data": {
                        "stars": business.stars,
                        "review_count": business.review_count,
                        "latitude": business.latitude,
                        "longitude": business.longitude,
                        "absa_features": business_absa
                    }
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return schemas.AIPrediction(**data)
            else:
                logger.warning(f"AI prediction failed: {response.status_code}")
                return None
                
    except Exception as e:
        logger.error(f"AI prediction error: {e}")
        return None

# 데이터베이스 테이블 생성
models.Base.metadata.create_all(bind=engine)
# 중요, 시작될 때, modesl.py에 정의한 클래스를 기반으로 실제 데이터베이스에 테이블을 생성한다
# 테이블이 이미 존재하면 아무 작업도 하지 않는다 

# FastAPI 앱
app = FastAPI(
    title="Two-Tower Recommendation Web API",
    description="Web backend for Two-Tower recommendation system",
    version="1.0.0"
)
# FastAPI 애플리케이션의 핵심 인스턴스를 생성한다 

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Cross Origin Resource Sharing 설정이다 
# 브라우저 보안 정책상 기본적으로 localhost:3000에서 localhost:8000으로 API를 호출할수없다
# 이 미들웨어는 allow_origins=["*"] 에서 요청을 허용하도록 설정하여
# 개발 환경에서 프론트엔드와 백엔드가 원할하게 통신할 수 있게 해준다
# 프로덕션에서는 ["*"] 대신 실제 프론트엔드 도메인을 적어야한다 

# Model API URL
MODEL_API_URL = "http://localhost:8001"
# 추천 요청을 보낼 머신러닝 모델 API 서버의 주소를 저장한다 

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

@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """회원가입"""
    # 새 사용자를 생성한다
    # 사용자가 입력한 username 등을 받고, UserResponse 객체를 출력한다 
    # 중복 확인
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # 사용자 생성
    hashed_password = auth.get_password_hash(user.password)
    # 사용자가 작성한 비밀번호를 해시한다 
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        # 신규 회원은 Yelp 데이터 없이 기본값으로 생성
        yelp_user_id=None,
        review_count=0,
        useful=0,
        compliment=0,
        fans=0,
        average_stars=0.0,
        yelping_since_days=0,
        absa_features=None
    )
    # models.User 객체를 생성해서 디비에 저장한다 
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    logger.info(f"New user registered: {user.username}")
    
    # 자동 로그인: 토큰 생성
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    
    # 토큰과 사용자 정보를 함께 반환
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

@app.post("/api/auth/login", response_model=schemas.Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """로그인"""
    # 사용자 인증 후 JWT 액세스 토큰을 발급한다 
    # Depends()는 FastAPI가 이 형식의 데이터를 자동으로 받아오게 한다 
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
    # 현재 로그인된 사용자의 정보를 반환한다 
    return current_user

# ============================================================================
# Business Endpoints
# ============================================================================

@app.get("/api/businesses", response_model=List[schemas.BusinessResponse])
async def get_businesses(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user_optional)
):
    """비즈니스 목록 조회 (ABSA 상위 특징 포함)"""
    businesses = db.query(models.Business).offset(skip).limit(limit).all()
    
    # 각 비즈니스에 상위 ABSA 특징 추가
    result = []
    for business in businesses:
        # ABSA features 가져오기 (직접 접근)
        absa_dict = business.absa_features
        
        business_dict = {
            "id": business.id,
            "business_id": business.business_id,
            "name": business.name,
            "categories": business.categories,
            "stars": business.stars,
            "review_count": business.review_count,
            "address": business.address,
            "city": business.city,
            "state": business.state,
            "latitude": business.latitude,
            "longitude": business.longitude,
            "top_features": get_top_absa_features(absa_dict)
        }
        
        # 로그인 사용자면 AI 예측 추가
        if current_user:
            prediction = await get_ai_prediction(current_user, business)
            if prediction:
                business_dict["ai_prediction"] = prediction
        
        result.append(schemas.BusinessResponse(**business_dict))
    
    return result
    # 가게 목록을 페이지네이션으로 조회한다 

@app.get("/api/businesses/{business_id}", response_model=schemas.BusinessResponse)
async def get_business(
    business_id: str, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user_optional)
):
    """비즈니스 상세 정보 조회 (전체 ABSA 피처 포함)"""
    business = db.query(models.Business).filter(
        models.Business.business_id == business_id
    ).first()
    
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    
    # ABSA features 가져오기 (직접 접근)
    absa_dict = business.absa_features
    
    # 상세 정보 구성
    business_dict = {
        "id": business.id,
        "business_id": business.business_id,
        "name": business.name,
        "categories": business.categories,
        "stars": business.stars,
        "review_count": business.review_count,
        "address": business.address,
        "city": business.city,
        "state": business.state,
        "latitude": business.latitude,
        "longitude": business.longitude,
        "absa_features": absa_dict,  # 전체 ABSA 피처
        "top_features": get_top_absa_features(absa_dict)  # 상위 특징도 포함
    }
    
    # 로그인 사용자면 AI 예측 추가
    if current_user:
        prediction = await get_ai_prediction(current_user, business)
        if prediction:
            business_dict["ai_prediction"] = prediction
    
    return schemas.BusinessResponse(**business_dict)
    # 특정 가게의 상세 정보를 본다 

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
    
    # User 정보를 함께 로드 (username 포함)
    reviews = db.query(models.Review).join(models.User).filter(
        models.Review.business_id == business.id
    ).offset(skip).limit(limit).all()
    
    # 각 리뷰에 username 추가
    result = []
    for review in reviews:
        review_dict = {
            "id": review.id,
            "user_id": review.user_id,
            "business_id": review.business_id,
            "stars": review.stars,
            "text": review.text,
            "created_at": review.created_at,
            "username": review.user.username  # User relationship을 통해 username 추가
        }
        result.append(review_dict)
    
    return result
    # app.get으로 가게에 작성된 리뷰를 가져오고, 각 리뷰에 작성자의 username을 포함한다 

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
    
    # username을 포함한 응답 반환
    return {
        "id": db_review.id,
        "user_id": db_review.user_id,
        "business_id": db_review.business_id,
        "stars": db_review.stars,
        "text": db_review.text,
        "created_at": db_review.created_at,
        "username": current_user.username
    }
    # app.post로 가게에 리뷰를 작성하고, 작성자의 username을 포함하여 반환한다 

# ============================================================================
# Recommendation Endpoint
# ============================================================================

# 모델 API 게이트웨이 역할을 수행한다
# 현재 로그인한 사용자에게 맞춤형 가게 목록을 추천한다 (인증필요) 
@app.get("/api/recommendations", response_model=schemas.RecommendationResponse)
async def get_recommendations(
    top_k: int = 10,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
): # Depends로 현재 사용자를 식별한다 
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
        # 현재 사용자가 최근에 리뷰를 작성한 5곳의 비즈니스 아이디를 가져온다 
        
        # Model API 호출
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MODEL_API_URL}/recommend",
                json={
                    "user_id": current_user.username,
                    "user_features": {
                        "review_count": db.query(models.Review).filter(models.Review.user_id == current_user.id).count(),
                        "useful": current_user.useful if current_user.useful else 0.0,
                        "compliment": current_user.compliment if current_user.compliment else 0.0,
                        "fans": current_user.fans if current_user.fans else 0.0,
                        "average_stars": current_user.average_stars if current_user.average_stars else 3.0,
                        "yelping_since_days": current_user.yelping_since_days if current_user.yelping_since_days else 0.0
                    },
                    "recent_business_ids": recent_business_ids,
                    "top_k": top_k
                },
                timeout=10.0
            ) 
            # 현재 사용자의 나이, 성별, 리뷰 수, 평균 별점 등을 추천 모델에 전송한다 
            # 즉, 백엔드에서 json 형식으로 데이터를 보내면, 모델은 해당 json을 이용해 결과를 분석하고, response 형태로 답장한다 
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to get recommendations from model API"
                )
            
            model_response = response.json()
        
        # Business 정보와 함께 반환
        recommendations = []
        for business_id, score in zip(
            model_response["recommendations"], # 비즈니스 아이디 목록
            model_response["scores"] # 추천 점수 목록
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
                ) # 추천 결과 목록의 1개 항목 형태로 추가한다 
        
        logger.info(f"Returned {len(recommendations)} recommendations for user {current_user.username}")
        
        return schemas.RecommendationResponse(
            recommendations=recommendations,
            user_id=current_user.id
        ) # 응답에 필요한 recommendations은 List[RecommendationItem] 형식이다 
        
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

