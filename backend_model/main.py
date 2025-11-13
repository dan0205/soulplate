"""
Tier 3: Model API Server
DeepFM과 Multi-Tower 모델을 사용한 별점 예측 API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from backend_model.prediction_service import get_prediction_service
from backend_model.absa_service import get_absa_service
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
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
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
    logger.info(f"Rating prediction request")
    
    try:
        pred_service = get_prediction_service()
        
        # 예측 수행
        result = pred_service.predict_rating(
            user_data=request.user_data,
            business_data=request.business_data
        )
        
        logger.info(f"Prediction: DeepFM={result['deepfm_rating']}, MT={result['multitower_rating']}, Ensemble={result['ensemble_rating']}")
        
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
    logger.info(f"Review analysis request: {len(request.text)} chars")
    
    try:
        absa_service = get_absa_service()
        pred_service = get_prediction_service()
        
        # ABSA 분석
        absa_features = absa_service.analyze_review(request.text)
        logger.info(f"ABSA analysis completed: {len(absa_features)} features")
        
        # 텍스트 임베딩
        if pred_service.text_embedding_service is not None:
            text_embedding = pred_service.text_embedding_service.transform_text(request.text)
            text_embedding_list = text_embedding.tolist()
        else:
            # 텍스트 임베딩 서비스 없으면 0 벡터
            text_embedding_list = [0.0] * 100
        
        logger.info(f"Text embedding completed: {len(text_embedding_list)} dims")
        
        return AnalyzeReviewResponse(
            absa_features=absa_features,
            text_embedding=text_embedding_list
        )
        
    except Exception as e:
        logger.error(f"Error in review analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze review: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

