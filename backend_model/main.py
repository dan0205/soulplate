"""
Tier 3: Model API Server
Two-Tower 모델을 사용한 추천 API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from backend_model.schemas import RecommendRequest, RecommendResponse, HealthResponse
from backend_model.model_loader import get_model_loader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생명주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # Startup
    logger.info("Starting Model API Server...")
    try:
        model_loader = get_model_loader()
        model_loader.load_all()
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
    model_loader = get_model_loader()
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.user_tower is not None,
        index_loaded=model_loader.faiss_index is not None,
        num_items=model_loader.faiss_index.ntotal if model_loader.faiss_index else 0
    )

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
        )
        
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

