"""
사용자-음식점 예측 점수 캐싱 서비스
"""

import logging
import httpx
import os
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_

import models

logger = logging.getLogger(__name__)

# AI 모델 서버 URL
MODEL_SERVER_URL = os.getenv("MODEL_API_URL", "https://backendmodel-production-77a7.up.railway.app")


async def calculate_and_store_predictions(user_id: int, db: Session):
    """
    특정 사용자의 모든 음식점에 대한 예측 점수를 계산하고 DB에 저장
    
    Args:
        user_id: 사용자 ID
        db: 데이터베이스 세션
    """
    import time
    
    total_start_time = time.time()
    logger.info(f"🔮 [Prediction Cache] 사용자 {user_id}의 예측 계산 시작")
    
    try:
        # 1. 사용자 정보 조회
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            logger.error(f"❌ [Prediction Cache] 사용자 {user_id}를 찾을 수 없습니다")
            return
        
        # 2. 모든 음식점 조회
        businesses = db.query(models.Business).all()
        total_businesses = len(businesses)
        logger.info(f"📊 [Prediction Cache] {total_businesses}개 음식점에 대해 예측 계산 중 (timeout=120s per request)...")
        
        # 3. 각 음식점에 대해 예측 요청
        calculated_at = datetime.now(timezone.utc)
        success_count = 0
        error_count = 0
        timeout_count = 0
        
        # 시간 측정을 위한 변수들
        api_call_times = []
        slow_businesses = []  # 3초 이상 걸린 요청들 추적
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            for idx, business in enumerate(businesses, 1):
                try:
                    # 진행상황 로그 (10개마다)
                    if idx % 10 == 0 or idx == 1:
                        logger.info(f"  📈 진행중: {idx}/{total_businesses} ({(idx/total_businesses*100):.1f}%)")
                    
                    # AI 모델 서버에 예측 요청
                    request_start = time.time()
                    
                    response = await client.post(
                        f"{MODEL_SERVER_URL}/predict_rating",
                        json={
                            "user_data": {
                                "review_count": user.review_count,
                                "useful": user.useful,
                                "compliment": user.compliment,
                                "fans": user.fans,
                                "average_stars": user.average_stars,
                                "yelping_since_days": user.yelping_since_days,
                                "absa_features": user.absa_features or {}
                            },
                            "business_data": {
                                "stars": business.stars,
                                "review_count": business.review_count,
                                "latitude": business.latitude,
                                "longitude": business.longitude,
                                "absa_features": business.absa_features or {}
                            }
                        }
                    )
                    
                    request_time = time.time() - request_start
                    api_call_times.append(request_time)
                    
                    # 느린 요청 추적 (3초 이상)
                    if request_time > 3.0:
                        slow_businesses.append({
                            "id": business.id,
                            "name": business.name or f"Business {business.id}",
                            "time": request_time
                        })
                    
                    if response.status_code == 200:
                        data = response.json()
                        deepfm_score = data.get("deepfm_rating", 3.0)
                        multitower_score = data.get("multitower_rating", 3.0)
                        
                        # DB에 저장 (UPSERT)
                        existing = db.query(models.UserBusinessPrediction).filter(
                            and_(
                                models.UserBusinessPrediction.user_id == user_id,
                                models.UserBusinessPrediction.business_id == business.id
                            )
                        ).first()
                        
                        if existing:
                            # 업데이트
                            existing.deepfm_score = deepfm_score
                            existing.multitower_score = multitower_score
                            existing.is_stale = False
                            existing.calculated_at = calculated_at
                        else:
                            # 신규 삽입
                            prediction = models.UserBusinessPrediction(
                                user_id=user_id,
                                business_id=business.id,
                                deepfm_score=deepfm_score,
                                multitower_score=multitower_score,
                                is_stale=False,
                                calculated_at=calculated_at
                            )
                            db.add(prediction)
                        
                        success_count += 1
                    else:
                        logger.warning(f"⚠️  [Prediction Cache] 음식점 {business.id} 예측 실패: HTTP {response.status_code} (소요: {request_time:.2f}s)")
                        error_count += 1
                
                except httpx.TimeoutException:
                    request_time = time.time() - request_start
                    logger.error(f"⏱️  [Prediction Cache] 음식점 {business.id} 타임아웃 (120초 초과, 실제: {request_time:.2f}s)")
                    timeout_count += 1
                    error_count += 1
                
                except Exception as e:
                    request_time = time.time() - request_start
                    logger.error(f"❌ [Prediction Cache] 음식점 {business.id} 예측 중 오류 (소요: {request_time:.2f}s): {e}")
                    error_count += 1
            
            # 4. DB 커밋
            commit_start = time.time()
            db.commit()
            commit_time = time.time() - commit_start
            
            # 5. 최종 통계 계산
            total_time = time.time() - total_start_time
            
            # API 호출 시간 통계
            if api_call_times:
                avg_time = sum(api_call_times) / len(api_call_times)
                min_time = min(api_call_times)
                max_time = max(api_call_times)
            else:
                avg_time = min_time = max_time = 0
            
            # 최종 로그
            logger.info(f"✅ [Prediction Cache] 완료 - 총 소요시간: {total_time:.2f}s")
            logger.info(f"   📊 결과: 성공 {success_count}, 실패 {error_count} (타임아웃: {timeout_count})")
            logger.info(f"   ⏱️  API 호출 시간: 평균 {avg_time:.2f}s, 최소 {min_time:.2f}s, 최대 {max_time:.2f}s")
            logger.info(f"   💾 DB 커밋 시간: {commit_time:.2f}s")
            
            # 느린 요청들 로그
            if slow_businesses:
                logger.warning(f"   ⚠️  느린 요청 (3초 이상): {len(slow_businesses)}개")
                for slow in slow_businesses[:5]:  # 상위 5개만 표시
                    logger.warning(f"      - {slow['name']} (ID: {slow['id']}): {slow['time']:.2f}s")
    
    except Exception as e:
        total_time = time.time() - total_start_time
        logger.error(f"❌ [Prediction Cache] 사용자 {user_id} 예측 계산 중 치명적 오류 (소요: {total_time:.2f}s): {e}")
        import traceback
        traceback.print_exc()
        db.rollback()


def mark_predictions_stale(user_id: int, db: Session):
    """
    사용자의 모든 예측을 재계산 필요 상태로 표시
    
    Args:
        user_id: 사용자 ID
        db: 데이터베이스 세션
    """
    logger.info(f"[Prediction Cache] 사용자 {user_id}의 예측을 stale로 표시")
    
    try:
        updated_count = db.query(models.UserBusinessPrediction).filter(
            models.UserBusinessPrediction.user_id == user_id
        ).update({"is_stale": True})
        
        db.commit()
        logger.info(f"[Prediction Cache] {updated_count}개 예측을 stale로 표시 완료")
    
    except Exception as e:
        logger.error(f"[Prediction Cache] stale 표시 중 오류: {e}")
        db.rollback()


def get_cached_predictions(user_id: int, business_ids: list, db: Session) -> dict:
    """
    캐시된 예측값을 조회
    
    Args:
        user_id: 사용자 ID
        business_ids: 조회할 음식점 ID 리스트
        db: 데이터베이스 세션
    
    Returns:
        {business_id: {"deepfm": score, "multitower": score, "is_stale": bool}}
    """
    predictions = db.query(models.UserBusinessPrediction).filter(
        and_(
            models.UserBusinessPrediction.user_id == user_id,
            models.UserBusinessPrediction.business_id.in_(business_ids)
        )
    ).all()
    
    result = {}
    for pred in predictions:
        result[pred.business_id] = {
            "deepfm": pred.deepfm_score,
            "multitower": pred.multitower_score,
            "is_stale": pred.is_stale
        }
    
    return result


def check_predictions_exist(user_id: int, db: Session) -> bool:
    """
    사용자의 예측값이 존재하는지 확인
    
    Args:
        user_id: 사용자 ID
        db: 데이터베이스 세션
    
    Returns:
        bool: 예측값 존재 여부
    """
    count = db.query(models.UserBusinessPrediction).filter(
        models.UserBusinessPrediction.user_id == user_id
    ).count()
    
    return count > 0

