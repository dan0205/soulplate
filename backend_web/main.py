"""
Tier 2: Web Backend Server
사용자 인증, DB 관리, 모델 API 게이트웨이
"""

import sys
from pathlib import Path

# 현재 디렉토리를 Python path에 추가
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List, Optional
from collections import defaultdict
import httpx
import logging
import numpy as np

import models
import schemas
import auth
from database import engine, get_db, SessionLocal
from taste_test_questions import (
    QUICK_TEST_QUESTIONS, 
    DEEP_TEST_QUESTIONS,
    answers_to_absa,
    calculate_mbti_type,
    MBTI_TYPE_DESCRIPTIONS
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 프로필 업데이트 함수
# ============================================================================

def update_user_profile(user_id: int, db: Session):
    """
    사용자 프로필 재계산 (가중 평균)
    모든 리뷰 (실제 리뷰 + 취향 테스트)의 ABSA와 텍스트 임베딩을 평균냄
    """
    # 사용자의 모든 리뷰 조회 (ABSA가 있는 것만)
    reviews = db.query(models.Review).filter(
        models.Review.user_id == user_id,
        models.Review.absa_features.isnot(None)
    ).all()
    
    if not reviews:
        logger.info(f"User {user_id}: No reviews with ABSA features")
        return
    
    # 가중 평균 계산
    absa_sum = defaultdict(float)
    text_emb_sum = np.zeros(100, dtype=np.float32)
    total_weight = 0
    real_review_count = 0
    total_stars = 0.0
    
    for review in reviews:
        # 가중치 결정
        if review.is_taste_test:
            weight = review.taste_test_weight  # 0.7 or 1.0
        else:
            weight = 1.0
            real_review_count += 1
            if review.stars:
                total_stars += review.stars
        
        # ABSA 가중 합산
        for key, value in review.absa_features.items():
            absa_sum[key] += value * weight
        
        # 텍스트 임베딩 가중 합산 (리뷰에 저장되어 있지 않으므로 생략)
        
        total_weight += weight
    
    # 가중 평균 계산
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return
    
    user.absa_features = {
        key: value / total_weight 
        for key, value in absa_sum.items()
    }
    
    # 실제 리뷰 통계 업데이트
    user.review_count = real_review_count
    if real_review_count > 0:
        user.average_stars = total_stars / real_review_count
    
    db.commit()
    logger.info(f"User {user_id} profile updated: {real_review_count} reviews, {len(reviews)} total (with tests)")


async def process_review_features(review_id: int, user_id: int, text: str, stars: float):
    """
    백그라운드 작업: 리뷰 ABSA 분석, 프로필 업데이트, 예측 재계산
    """
    from prediction_cache import mark_predictions_stale, calculate_and_store_predictions
    
    db = SessionLocal()
    try:
        logger.info(f"Background task started for review {review_id}")
        
        # 1. backend_model API 호출 (ABSA + 텍스트 임베딩)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8001/analyze_review",
                json={"text": text}
            )
            
            if response.status_code != 200:
                logger.error(f"ABSA analysis failed: {response.status_code}")
                return
            
            result = response.json()
            absa_features = result["absa_features"]
            text_embedding = result["text_embedding"]
        
        logger.info(f"ABSA analysis completed for review {review_id}")
        
        # 2. Review에 ABSA 저장
        review = db.query(models.Review).filter(models.Review.id == review_id).first()
        if review:
            review.absa_features = absa_features
            db.commit()
            logger.info(f"Review {review_id} ABSA saved")
        
        # 3. User 프로필 업데이트
        update_user_profile(user_id, db)
        logger.info(f"User {user_id} profile updated after review {review_id}")
        
        # 4. 예측 캐시 재계산
        # 먼저 stale로 표시 (빠름)
        mark_predictions_stale(user_id, db)
        logger.info(f"Predictions marked as stale for user {user_id}")
        
        # 그 다음 재계산 (시간 걸림)
        await calculate_and_store_predictions(user_id, db)
        logger.info(f"Predictions recalculated for user {user_id}")
        
    except Exception as e:
        logger.error(f"Background task failed for review {review_id}: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


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
    allow_origins=[
        "http://localhost:3000",  # 로컬 개발
        "https://soulplate.vercel.app",  # 프로덕션
    ],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Cross Origin Resource Sharing 설정이다 
# 브라우저 보안 정책상 기본적으로 localhost:3000에서 localhost:8000으로 API를 호출할수없다
# 이 미들웨어는 allow_origins=["*"] 에서 요청을 허용하도록 설정하여
# 개발 환경에서 프론트엔드와 백엔드가 원할하게 통신할 수 있게 해준다
# 프로덕션에서는 ["*"] 대신 실제 프론트엔드 도메인을 적어야한다 

# Model API URL (별점 예측용)
MODEL_API_URL = "http://localhost:8001"
# 별점 예측 요청을 보낼 머신러닝 모델 API 서버의 주소를 저장한다 

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
async def register(
    user: schemas.UserCreate, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """회원가입"""
    from prediction_cache import calculate_and_store_predictions
    
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
    
    logger.info(f"New user registered: {user.username} (ID: {db_user.id})")
    
    # 백그라운드에서 초기 예측 계산
    background_tasks.add_task(calculate_and_store_predictions, db_user.id, db)
    logger.info(f"Background task scheduled for initial predictions: user {db_user.id}")
    
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
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
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
    
    # 예측값 확인 및 생성
    from prediction_cache import check_predictions_exist, calculate_and_store_predictions
    if not check_predictions_exist(user.id, db):
        logger.info(f"사용자 {user.username}의 예측값이 없어 백그라운드 생성 시작")
        if background_tasks:
            background_tasks.add_task(calculate_and_store_predictions, user.id, db)
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=schemas.UserResponse)
async def get_current_user_info(current_user: models.User = Depends(auth.get_current_user)):
    """현재 사용자 정보"""
    # 현재 로그인된 사용자의 정보를 반환한다 
    return current_user

# ============================================================================
# User Profile Endpoints
# ============================================================================

def get_user_taste_test_info(user_id: int, db: Session):
    """사용자의 취향 테스트 정보 조회"""
    taste_test_review = db.query(models.Review).filter(
        models.Review.user_id == user_id,
        models.Review.is_taste_test == True
    ).order_by(models.Review.created_at.desc()).first()
    
    if not taste_test_review:
        return False, None, None
    
    # MBTI 타입 계산
    from taste_test_questions import calculate_mbti_type
    mbti_type = calculate_mbti_type(taste_test_review.absa_features) if taste_test_review.absa_features else None
    
    return True, taste_test_review.taste_test_type, mbti_type

@app.get("/api/users/me/profile", response_model=schemas.UserProfileResponse)
async def get_my_profile(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """본인 프로필 조회"""
    taste_test_completed, taste_test_type, mbti_type = get_user_taste_test_info(current_user.id, db)
    
    return schemas.UserProfileResponse(
        id=current_user.id,
        username=current_user.username,
        review_count=current_user.review_count,
        useful=current_user.useful,
        fans=current_user.fans,
        created_at=current_user.created_at,
        absa_features=current_user.absa_features,
        taste_test_completed=taste_test_completed,
        taste_test_type=taste_test_type,
        taste_test_mbti_type=mbti_type
    )

@app.get("/api/users/{user_id}/profile", response_model=schemas.UserProfileResponse)
async def get_user_profile(
    user_id: int,
    db: Session = Depends(get_db)
):
    """다른 사용자 프로필 조회"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    taste_test_completed, taste_test_type, mbti_type = get_user_taste_test_info(user_id, db)
    
    return schemas.UserProfileResponse(
        id=user.id,
        username=user.username,
        review_count=user.review_count,
        useful=user.useful,
        fans=user.fans,
        created_at=user.created_at,
        absa_features=user.absa_features,
        taste_test_completed=taste_test_completed,
        taste_test_type=taste_test_type,
        taste_test_mbti_type=mbti_type
    )

@app.get("/api/users/{user_id}/reviews", response_model=List[schemas.UserReviewResponse])
async def get_user_reviews(
    user_id: int,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """사용자의 리뷰 목록 조회 (음식점 정보 포함)"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Business 정보를 JOIN하여 리뷰 조회
    reviews = db.query(models.Review).join(models.Business).filter(
        models.Review.user_id == user_id
    ).order_by(models.Review.created_at.desc()).offset(skip).limit(limit).all()
    
    result = []
    for review in reviews:
        result.append({
            "id": review.id,
            "user_id": review.user_id,
            "business_id": review.business_id,
            "stars": review.stars,
            "text": review.text,
            "created_at": review.created_at,
            "useful": review.useful or 0,
            "business": {
                "business_id": review.business.business_id,
                "name": review.business.name
            }
        })
    
    return result

# ============================================================================
# Business Endpoints
# ============================================================================

@app.get("/api/businesses", response_model=schemas.BusinessListResponse)
async def get_businesses(
    skip: int = 0,
    limit: int = 20,
    sort_by: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user_optional),
    background_tasks: BackgroundTasks = None
):
    """비즈니스 목록 조회 (캐시된 예측 사용, 정렬 지원, 검색 지원)"""
    from prediction_cache import check_predictions_exist, calculate_and_store_predictions
    from sqlalchemy import and_, or_
    
    # 검색 필터 생성
    search_filter = None
    if search:
        search_pattern = f'%{search}%'
        search_filter = or_(
            models.Business.name.ilike(search_pattern),
            models.Business.categories.ilike(search_pattern),
            models.Business.city.ilike(search_pattern)
        )
    
    # 총 개수 조회 (검색 필터 적용)
    total_query = db.query(models.Business)
    if search_filter is not None:
        total_query = total_query.filter(search_filter)
    total = total_query.count()
    
    # 정렬에 따라 다른 쿼리 사용
    if sort_by == "review_count":
        # 리뷰 개수 내림차순
        query = db.query(models.Business)
        if search_filter is not None:
            query = query.filter(search_filter)
        businesses = query.order_by(
            models.Business.review_count.desc()
        ).offset(skip).limit(limit).all()
    elif sort_by in ["deepfm", "multitower"] and current_user:
        # AI 예측 기반 정렬: DB 캐시에서 조회
        # 예측값이 없으면 백그라운드에서 계산 시작
        if not check_predictions_exist(current_user.id, db):
            logger.info(f"사용자 {current_user.id}의 예측값이 없어 백그라운드 계산 시작")
            if background_tasks:
                background_tasks.add_task(calculate_and_store_predictions, current_user.id, db)
            # 예측값이 없는 경우 기본 정렬로 폴백
            query = db.query(models.Business)
            if search_filter is not None:
                query = query.filter(search_filter)
            businesses = query.offset(skip).limit(limit).all()
        else:
            # DB에서 예측값과 함께 조회 (JOIN)
            if sort_by == "deepfm":
                query = db.query(models.Business).join(
                    models.UserBusinessPrediction,
                    and_(
                        models.UserBusinessPrediction.user_id == current_user.id,
                        models.UserBusinessPrediction.business_id == models.Business.id
                    )
                )
                if search_filter is not None:
                    query = query.filter(search_filter)
                query = query.order_by(models.UserBusinessPrediction.deepfm_score.desc())
            else:  # multitower
                query = db.query(models.Business).join(
                    models.UserBusinessPrediction,
                    and_(
                        models.UserBusinessPrediction.user_id == current_user.id,
                        models.UserBusinessPrediction.business_id == models.Business.id
                    )
                )
                if search_filter is not None:
                    query = query.filter(search_filter)
                query = query.order_by(models.UserBusinessPrediction.multitower_score.desc())
            
            businesses = query.offset(skip).limit(limit).all()
    else:
        # 기본: 정렬 없음
        query = db.query(models.Business)
        if search_filter is not None:
            query = query.filter(search_filter)
        businesses = query.offset(skip).limit(limit).all()
    
    # 각 비즈니스에 상위 ABSA 특징 추가
    result = []
    for business in businesses:
        # ABSA features 가져오기
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
        
        # 로그인 사용자면 캐시된 AI 예측 추가
        if current_user:
            cached_pred = db.query(models.UserBusinessPrediction).filter(
                and_(
                    models.UserBusinessPrediction.user_id == current_user.id,
                    models.UserBusinessPrediction.business_id == business.id
                )
            ).first()
            
            if cached_pred:
                # 캐시된 예측값 사용
                business_dict["ai_prediction"] = schemas.AIPrediction(
                    deepfm_rating=cached_pred.deepfm_score,
                    multitower_rating=cached_pred.multitower_score,
                    ensemble_rating=(cached_pred.deepfm_score + (cached_pred.multitower_score or cached_pred.deepfm_score)) / 2
                )
            else:
                # 예측값이 없으면 fallback (실시간 계산은 너무 느리므로 생략)
                logger.debug(f"No cached prediction for user {current_user.id}, business {business.id}")
        
        result.append(schemas.BusinessResponse(**business_dict))
    
    # 페이지네이션 정보와 함께 반환
    return schemas.BusinessListResponse(
        businesses=result,
        total=total,
        skip=skip,
        limit=limit
    )
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
        else:
            logger.warning(f"AI prediction failed for business {business.business_id}")
    else:
        logger.info("No current_user, skipping AI prediction")
    
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
    ).order_by(models.Review.created_at.desc()).offset(skip).limit(limit).all()
    
    # 각 리뷰에 username과 useful 추가
    result = []
    for review in reviews:
        review_dict = {
            "id": review.id,
            "user_id": review.user_id,
            "business_id": review.business_id,
            "stars": review.stars,
            "text": review.text,
            "created_at": review.created_at,
            "username": review.user.username,  # User relationship을 통해 username 추가
            "useful": review.useful or 0  # useful 값 추가
        }
        result.append(review_dict)
    
    return result
    # app.get으로 가게에 작성된 리뷰를 가져오고, 각 리뷰에 작성자의 username을 포함한다 

@app.post("/api/businesses/{business_id}/reviews", response_model=schemas.ReviewResponse, status_code=status.HTTP_201_CREATED)
async def create_review(
    business_id: str,
    review: schemas.ReviewCreate,
    background_tasks: BackgroundTasks,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """리뷰 작성 (백그라운드에서 ABSA 분석 및 프로필 업데이트)"""
    business = db.query(models.Business).filter(
        models.Business.business_id == business_id
    ).first()
    
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    
    # 리뷰만 먼저 저장 (빠른 응답)
    db_review = models.Review(
        user_id=current_user.id,
        business_id=business.id,
        stars=review.stars,
        text=review.text,
        is_taste_test=False  # 실제 리뷰
    )
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    
    logger.info(f"New review created by user {current_user.username} for business {business_id} (ID: {db_review.id})")
    
    # 백그라운드 작업 추가 (ABSA 분석 + 프로필 업데이트)
    background_tasks.add_task(
        process_review_features,
        db_review.id,
        current_user.id,
        review.text,
        review.stars
    )
    logger.info(f"Background task scheduled for review {db_review.id}")
    
    # 즉시 응답 반환
    return {
        "id": db_review.id,
        "user_id": db_review.user_id,
        "business_id": db_review.business_id,
        "stars": db_review.stars,
        "text": db_review.text,
        "created_at": db_review.created_at,
        "username": current_user.username,
        "useful": db_review.useful or 0
    } 

@app.put("/api/reviews/{review_id}/useful", response_model=schemas.ReviewResponse)
async def increment_review_useful(
    review_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """리뷰 useful 증가"""
    review = db.query(models.Review).filter(models.Review.id == review_id).first()
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    # useful 값 증가
    review.useful = (review.useful or 0) + 1
    db.commit()
    db.refresh(review)
    
    logger.info(f"Review {review_id} useful incremented by user {current_user.username}")
    
    # username을 포함한 응답 반환
    return {
        "id": review.id,
        "user_id": review.user_id,
        "business_id": review.business_id,
        "stars": review.stars,
        "text": review.text,
        "created_at": review.created_at,
        "username": review.user.username,
        "useful": review.useful or 0
    }

# ============================================================================
# Taste Test Endpoints
# ============================================================================

@app.get("/api/user/status", response_model=schemas.UserStatusResponse)
async def get_user_status(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """사용자 상태 확인 (신규 사용자 판별)"""
    
    # 실제 리뷰 개수 (취향 테스트 제외)
    real_review_count = db.query(models.Review).filter(
        models.Review.user_id == current_user.id,
        models.Review.is_taste_test == False
    ).count()
    
    # 취향 테스트 완료 여부
    taste_test_review = db.query(models.Review).filter(
        models.Review.user_id == current_user.id,
        models.Review.is_taste_test == True
    ).first()
    
    has_taste_test = taste_test_review is not None
    
    # MBTI 타입 계산 (취향 테스트 완료 시)
    mbti_type = None
    if has_taste_test and current_user.absa_features:
        mbti_type = calculate_mbti_type(current_user.absa_features)
    
    return schemas.UserStatusResponse(
        is_new_user=(real_review_count == 0),
        review_count=real_review_count,
        has_taste_test=has_taste_test,
        should_show_test_popup=(real_review_count == 0 and not has_taste_test),
        mbti_type=mbti_type
    )

@app.get("/api/taste-test/questions")
async def get_taste_test_questions(test_type: str = "quick"):
    """취향 테스트 질문 조회"""
    if test_type == "quick":
        questions = QUICK_TEST_QUESTIONS
    elif test_type == "deep":
        questions = DEEP_TEST_QUESTIONS
    else:
        raise HTTPException(status_code=400, detail="Invalid test_type. Use 'quick' or 'deep'")
    
    # absa_mapping 제외하고 반환 (클라이언트에 불필요)
    simplified_questions = []
    for q in questions:
        simplified_questions.append({
            "id": q["id"],
            "aspect": q["aspect"],
            "question": q["question"],
            "labels": q["labels"]
        })
    
    return {
        "test_type": test_type,
        "questions": simplified_questions
    }

@app.post("/api/taste-test/submit", response_model=schemas.TasteTestResult, status_code=status.HTTP_201_CREATED)
async def submit_taste_test(
    submission: schemas.TasteTestSubmit,
    background_tasks: BackgroundTasks,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """취향 테스트 제출"""
    
    # 테스트 타입 검증
    if submission.test_type not in ["quick", "deep"]:
        raise HTTPException(status_code=400, detail="Invalid test_type")
    
    expected_count = 8 if submission.test_type == "quick" else 20
    if len(submission.answers) != expected_count:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {expected_count} answers, got {len(submission.answers)}"
        )
    
    # 답변 범위 검증 (1-5)
    for answer in submission.answers:
        if answer < 1 or answer > 5:
            raise HTTPException(status_code=400, detail="Answers must be between 1 and 5")
    
    try:
        # 1. 답변을 ABSA 특징으로 변환
        absa_features = answers_to_absa(submission.answers, submission.test_type)
        logger.info(f"Taste test ABSA features generated: {len(absa_features)} features")
        
        # 2. MBTI 타입 계산
        mbti_type = calculate_mbti_type(absa_features)
        type_info = MBTI_TYPE_DESCRIPTIONS.get(mbti_type, {
            "name": "알 수 없음",
            "description": "타입 정보를 찾을 수 없습니다",
            "recommendations": []
        })
        
        # 3. 기존 취향 테스트 삭제 (재테스트 시)
        db.query(models.Review).filter(
            models.Review.user_id == current_user.id,
            models.Review.is_taste_test == True
        ).delete()
        db.commit()
        
        # 4. 가상 리뷰로 저장
        test_weight = 0.7 if submission.test_type == "quick" else 1.0
        taste_test_review = models.Review(
            user_id=current_user.id,
            business_id=None,  # 취향 테스트는 특정 비즈니스와 무관
            stars=None,
            text=f"취향 테스트 결과 ({submission.test_type})",
            is_taste_test=True,
            taste_test_type=submission.test_type,
            taste_test_weight=test_weight,
            absa_features=absa_features
        )
        db.add(taste_test_review)
        db.commit()
        db.refresh(taste_test_review)
        
        logger.info(f"Taste test saved for user {current_user.username} (type: {submission.test_type}, MBTI: {mbti_type})")
        
        # 5. User 프로필 업데이트 (백그라운드)
        background_tasks.add_task(update_user_profile, current_user.id, db)
        
        # 6. 예측 캐시 재계산 (백그라운드)
        from prediction_cache import mark_predictions_stale, calculate_and_store_predictions
        background_tasks.add_task(mark_predictions_stale, current_user.id, db)
        background_tasks.add_task(calculate_and_store_predictions, current_user.id, db)
        logger.info(f"Background tasks scheduled for user {current_user.id} predictions")
        
        # 7. 결과 반환
        return schemas.TasteTestResult(
            mbti_type=mbti_type,
            type_name=type_info["name"],
            description=type_info["description"],
            recommendations=type_info["recommendations"]
        )
        
    except ValueError as e:
        logger.error(f"Taste test error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Taste test failed: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to process taste test")

@app.delete("/api/taste-test")
async def delete_taste_test(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """취향 테스트 삭제 (재테스트 전 초기화)"""
    
    deleted_count = db.query(models.Review).filter(
        models.Review.user_id == current_user.id,
        models.Review.is_taste_test == True
    ).delete()
    
    db.commit()
    
    # 프로필 재계산
    update_user_profile(current_user.id, db)
    
    logger.info(f"Taste test deleted for user {current_user.username} ({deleted_count} records)")
    
    return {"message": "Taste test deleted", "deleted_count": deleted_count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

