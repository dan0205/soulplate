"""
Tier 3: Model API Server
Two-Tower 모델을 사용한 추천 API
"""
# 웹 백엔드와는 별개의 서버로, uvicorn을 통해 포트 8001에서 독립적으로 실행된다
# 이 서버의 유일한 목적은 추천 계산이라는 무거운 AI작업을 전담하는 것이다
# 이 서버는 사용자 DB나 비밀번호같은 것은 전혀 모르며, 오직 AI모델과 벡터 계산에만 집중한다

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from backend_model.schemas import RecommendRequest, RecommendResponse, HealthResponse
from backend_model.model_loader import get_model_loader
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
@asynccontextmanager #FastAPI 서버의 시작과 종료 시점에 특정 작업을 수행하도록 관리한다 
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # Startup
    logger.info("Starting Model API Server...")
    try:
        model_loader = get_model_loader()
        model_loader.load_all()
        
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
    title="Two-Tower Recommendation Model API",
    description="Real-time recommendation API using Two-Tower architecture",
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
        "message": "Two-Tower Recommendation Model API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check 엔드포인트"""
    # 이 모델 서버가 살아있는지, 제대로 작동할 준비가 되었는지를 확인한다 
    model_loader = get_model_loader()
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.user_tower is not None,
        index_loaded=model_loader.faiss_index is not None,
        num_items=model_loader.faiss_index.ntotal if model_loader.faiss_index else 0
    )

# 웹 백엔드로부터 실제 추천 요청을 받아 처리하는 메인 엔드포인트 
@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation"])
async def recommend(request: RecommendRequest):
    """
    개인화 추천 엔드포인트
    
    사용자 정보를 기반으로 가장 관련성 높은 비즈니스를 추천합니다.
    """
    logger.info(f"Recommendation request for user: {request.user_id}, top_k: {request.top_k}")
    
    try:
        model_loader = get_model_loader()
        
        # 1. 사용자 벡터 생성
        user_vector = model_loader.get_user_vector(
            user_id=request.user_id,
            user_features=request.user_features
        ) # 유저 아이디와 피처를 유저 타워 모델에 통과시켜 사용자의 현재 취향 벡터를 생성 
        
        # 2. FAISS로 유사 아이템 검색
        business_ids, scores = model_loader.search_similar_items(
            user_vector=user_vector,
            top_k=request.top_k
        )
        
        # 3. 응답 생성
        response = RecommendResponse(
            recommendations=business_ids,
            scores=scores,
            user_id=request.user_id
        )
        
        logger.info(f"Returned {len(business_ids)} recommendations for user {request.user_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )

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

@app.get("/model/info", tags=["Model"])
async def model_info():
    """모델 정보 조회"""
    model_loader = get_model_loader()
    
    return {
        "user_tower_loaded": model_loader.user_tower is not None,
        "index_loaded": model_loader.faiss_index is not None,
        "num_users": len(model_loader.users_df) if model_loader.users_df is not None else 0,
        "num_items": model_loader.faiss_index.ntotal if model_loader.faiss_index else 0,
        "embedding_dim": 128,
        "device": str(model_loader.device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

