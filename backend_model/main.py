"""
Tier 3: Model API Server
DeepFM과 Multi-Tower 모델을 사용한 별점 예측 API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from backend_model.prediction_service import get_prediction_service
from pydantic import BaseModel
from typing import Optional

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
        models_loaded = pred_service.deepfm_model is not None
        return {
            "status": "healthy",
            "deepfm_loaded": pred_service.deepfm_model is not None,
            "multitower_loaded": pred_service.multitower_model is not None
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

