"""
Tier 3: Model API Server
DeepFM과 Multi-Tower 모델을 사용한 별점 예측 API
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python path에 추가
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from prediction_service_309d import get_prediction_service
from absa_service import get_absa_service
from pydantic import BaseModel
from typing import Optional, List

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 예측 요청/응답 스키마
class PredictRatingRequest(BaseModel):
    """별점 예측 요청"""
    user_data: dict
    business_data: dict

class PredictRatingResponse(BaseModel):
    """별점 예측 응답"""
    deepfm_rating: float
    multitower_rating: Optional[float]  # Multi-Tower 사용 불가 시 None
    ensemble_rating: float
    confidence: float

class AnalyzeReviewRequest(BaseModel):
    """리뷰 분석 요청"""
    text: str

class AnalyzeReviewResponse(BaseModel):
    """리뷰 분석 응답"""
    absa_features: dict  # 51개 aspect-sentiment 확률
    text_embedding: List[float]  # 100차원 텍스트 임베딩

# FastAPI 앱 생명주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # Startup
    logger.info("Starting Model API Server...")
    try:
        # 예측 서비스 로딩
        pred_service = get_prediction_service()
        logger.info("Prediction Service loaded!")
        
        # ABSA 서비스 로딩
        absa_service = get_absa_service()
        logger.info("ABSA Service loaded!")
        
        logger.info("Model API Server started successfully!")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Model API Server...")

# FastAPI 앱 생성
app = FastAPI(
    title="DeepFM & Multi-Tower Rating Prediction API",
    description="Rating prediction API using DeepFM and Multi-Tower models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (개발 환경용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  # 로컬 Web Backend
        "https://backendweb-production-7b6c.up.railway.app",  # 프로덕션
    ],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "DeepFM & Multi-Tower Rating Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check 엔드포인트"""
    try:
        pred_service = get_prediction_service()
        absa_service = get_absa_service()
        return {
            "status": "healthy",
            "deepfm_loaded": pred_service.deepfm_model is not None,
            "multitower_loaded": pred_service.multitower_model is not None,
            "absa_loaded": absa_service.model is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/predict_rating", response_model=PredictRatingResponse, tags=["Prediction"])
async def predict_rating(request: PredictRatingRequest):
    """
    별점 예측 엔드포인트
    
    DeepFM과 Multi-Tower 모델을 사용하여 사용자가 특정 비즈니스에 매길 별점을 예측합니다.
    """
    # 요청 로그는 디버그 모드에서만 출력 (로그 과다 방지)
    if os.getenv("DEBUG_PREDICTION", "false").lower() == "true":
        logger.debug(f"Rating prediction request")
    
    try:
        pred_service = get_prediction_service()
        
        # 예측 수행
        result = pred_service.predict_rating(
            user_data=request.user_data,
            business_data=request.business_data
        )
        
        # 예측 결과 로그는 디버그 모드에서만 출력 (로그 과다 방지)
        if os.getenv("DEBUG_PREDICTION", "false").lower() == "true":
            logger.debug(f"Prediction: DeepFM={result['deepfm_rating']}, MT={result['multitower_rating']}, Ensemble={result['ensemble_rating']}")
        
        return PredictRatingResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in rating prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to predict rating: {str(e)}"
        )

@app.post("/analyze_review", response_model=AnalyzeReviewResponse, tags=["ABSA"])
async def analyze_review(request: AnalyzeReviewRequest):
    """
    리뷰 분석 엔드포인트
    
    리뷰 텍스트를 받아서 ABSA 분석 및 텍스트 임베딩을 반환합니다.
    - ABSA: 51개 aspect-sentiment 확률 (예: 맛_긍정, 서비스_부정 등)
    - 텍스트 임베딩: TF-IDF 기반 100차원 벡터
    """
    import time
    
    endpoint_start = time.time()
    text_sample = request.text[:50] + "..." if len(request.text) > 50 else request.text
    logger.info(f"📥 [ABSA Endpoint] 요청 시작 (텍스트 길이: {len(request.text)}자)")
    logger.info(f"   텍스트 샘플: \"{text_sample}\"")
    
    try:
        absa_service = get_absa_service()
        pred_service = get_prediction_service()
        
        # Step 1: ABSA 분석
        step1_start = time.time()
        absa_features = absa_service.analyze_review(request.text)
        step1_time = time.time() - step1_start
        logger.info(f"  ⏱️  Step 1: ABSA 분석 - {step1_time:.2f}s ({len(absa_features)} features)")
        
        # Step 2: 텍스트 임베딩
        step2_start = time.time()
        if pred_service.text_embedding_service is not None:
            text_embedding = pred_service.text_embedding_service.transform_text(request.text)
            text_embedding_list = text_embedding.tolist()
        else:
            # 텍스트 임베딩 서비스 없으면 0 벡터
            text_embedding_list = [0.0] * 100
        step2_time = time.time() - step2_start
        logger.info(f"  ⏱️  Step 2: 텍스트 임베딩 - {step2_time:.3f}s ({len(text_embedding_list)} dims)")
        
        # Step 3: 응답 생성
        step3_start = time.time()
        response = AnalyzeReviewResponse(
            absa_features=absa_features,
            text_embedding=text_embedding_list
        )
        step3_time = time.time() - step3_start
        logger.info(f"  ⏱️  Step 3: 응답 생성 - {step3_time:.3f}s")
        
        # 전체 소요 시간
        total_time = time.time() - endpoint_start
        logger.info(f"✅ [ABSA Endpoint] 완료 - 총 소요: {total_time:.2f}s")
        
        return response
        
    except Exception as e:
        total_time = time.time() - endpoint_start
        logger.error(f"❌ [ABSA Endpoint] 실패 after {total_time:.2f}s: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze review: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

