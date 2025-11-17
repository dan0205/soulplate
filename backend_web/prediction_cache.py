"""
사용자-음식점 예측 점수 캐싱 서비스
"""

import logging
import httpx
import os
import asyncio
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_

import models

logger = logging.getLogger(__name__)

# AI 모델 서버 URL
MODEL_SERVER_URL = os.getenv("MODEL_API_URL", "https://backendmodel-production-77a7.up.railway.app")

# 병렬 처리 설정
CONCURRENCY = int(os.getenv("PREDICTION_CONCURRENCY", "1"))
CHUNK_SIZE = int(os.getenv("PREDICTION_CHUNK_SIZE", "50"))
TIMEOUT = int(os.getenv("PREDICTION_TIMEOUT", "360"))


async def predict_for_business(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    user_data: dict,
    business: "models.Business",
    calculated_at: datetime
) -> dict:
    """
    단일 음식점에 대한 예측을 수행 (병렬 실행용 헬퍼 함수)
    
    Args:
        client: HTTP 클라이언트
        semaphore: 동시 실행 제어용 세마포어
        user_data: 사용자 데이터 dict
        business: 음식점 객체
        calculated_at: 계산 시각
    
    Returns:
        dict: {
            "success": bool,
            "business_id": int,
            "business_name": str,
            "deepfm_score": float,
            "multitower_score": float,
            "request_time": float,
            "error": str (실패 시)
        }
    """
    import time
    
    async with semaphore:
        try:
            start_time = time.time()
            
            response = await client.post(
                f"{MODEL_SERVER_URL}/predict_rating",
                json={
                    "user_data": user_data,
                    "business_data": {
                        "stars": business.stars,
                        "review_count": business.review_count,
                        "latitude": business.latitude,
                        "longitude": business.longitude,
                        "absa_features": business.absa_features or {}
                    }
                }
            )
            
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "business_id": business.id,
                    "business_name": business.name or f"Business {business.id}",
                    "deepfm_score": data.get("deepfm_rating", 3.0),
                    "multitower_score": data.get("multitower_rating", 3.0),
                    "request_time": request_time
                }
            else:
                return {
                    "success": False,
                    "business_id": business.id,
                    "business_name": business.name or f"Business {business.id}",
                    "error": f"HTTP {response.status_code}",
                    "request_time": request_time
                }
                
        except httpx.TimeoutException:
            request_time = time.time() - start_time
            return {
                "success": False,
                "business_id": business.id,
                "business_name": business.name or f"Business {business.id}",
                "error": "Timeout",
                "request_time": request_time
            }
            
        except Exception as e:
            request_time = time.time() - start_time
            return {
                "success": False,
                "business_id": business.id,
                "business_name": business.name or f"Business {business.id}",
                "error": str(e),
                "request_time": request_time
            }


async def calculate_and_store_predictions(user_id: int, db: Session):
    """
    특정 사용자의 모든 음식점에 대한 예측 점수를 계산하고 DB에 저장 (청크 기반 병렬 처리)
    
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
        total_chunks = (total_businesses - 1) // CHUNK_SIZE + 1
        
        logger.info(f"📊 [Prediction Cache] {total_businesses}개 음식점에 대해 예측 계산 중")
        logger.info(f"   ⚙️  설정: concurrency={CONCURRENCY}, chunk={CHUNK_SIZE}, timeout={TIMEOUT}s")
        
        # 3. 사용자 데이터 미리 준비 (반복 사용)
        user_data = {
            "review_count": user.review_count,
            "useful": user.useful,
            "compliment": user.compliment,
            "fans": user.fans,
            "average_stars": user.average_stars,
            "yelping_since_days": user.yelping_since_days,
            "absa_features": user.absa_features or {}
        }
        
        # 4. 전역 통계 변수
        calculated_at = datetime.now(timezone.utc)
        success_count = 0
        error_count = 0
        timeout_count = 0
        api_call_times = []
        slow_businesses = []  # 3초 이상 걸린 요청들
        
        # 5. Semaphore 생성 (동시 실행 제어)
        semaphore = asyncio.Semaphore(CONCURRENCY)
        
        # 6. 청크 단위로 처리
        async with httpx.AsyncClient(timeout=float(TIMEOUT)) as client:
            for chunk_idx in range(0, total_businesses, CHUNK_SIZE):
                chunk = businesses[chunk_idx:chunk_idx + CHUNK_SIZE]
                chunk_num = chunk_idx // CHUNK_SIZE + 1
                chunk_start = time.time()
                
                logger.info(f"  📦 청크 {chunk_num}/{total_chunks}: {len(chunk)}개 음식점 처리 중...")
                
                # 병렬 실행 (asyncio.gather)
                tasks = [
                    predict_for_business(client, semaphore, user_data, business, calculated_at)
                    for business in chunk
                ]
                results = await asyncio.gather(*tasks)
                
                # 결과 처리 및 DB 저장
                chunk_success = 0
                chunk_errors = 0
                chunk_timeouts = 0
                
                for result in results:
                    if result["success"]:
                        # DB에 저장 (UPSERT)
                        existing = db.query(models.UserBusinessPrediction).filter(
                            and_(
                                models.UserBusinessPrediction.user_id == user_id,
                                models.UserBusinessPrediction.business_id == result["business_id"]
                            )
                        ).first()
                        
                        if existing:
                            # 업데이트
                            existing.deepfm_score = result["deepfm_score"]
                            existing.multitower_score = result["multitower_score"]
                            existing.is_stale = False
                            existing.calculated_at = calculated_at
                        else:
                            # 신규 삽입
                            prediction = models.UserBusinessPrediction(
                                user_id=user_id,
                                business_id=result["business_id"],
                                deepfm_score=result["deepfm_score"],
                                multitower_score=result["multitower_score"],
                                is_stale=False,
                                calculated_at=calculated_at
                            )
                            db.add(prediction)
                        
                        chunk_success += 1
                        api_call_times.append(result["request_time"])
                        
                        # 느린 요청 추적 (3초 이상)
                        if result["request_time"] > 3.0:
                            slow_businesses.append({
                                "id": result["business_id"],
                                "name": result["business_name"],
                                "time": result["request_time"]
                            })
                    else:
                        # 에러 처리
                        chunk_errors += 1
                        if result.get("error") == "Timeout":
                            chunk_timeouts += 1
                            logger.error(f"     ⏱️  타임아웃: {result['business_name']} ({result['request_time']:.1f}s)")
                        else:
                            logger.warning(f"     ⚠️  실패: {result['business_name']} - {result.get('error', 'Unknown')}")
                
                # 청크마다 커밋
                db.commit()
                chunk_time = time.time() - chunk_start
                
                # 청크 통계 업데이트
                success_count += chunk_success
                error_count += chunk_errors
                timeout_count += chunk_timeouts
                
                logger.info(f"  ✅ 청크 {chunk_num}/{total_chunks} 완료: {chunk_time:.1f}s (성공={chunk_success}, 실패={chunk_errors})")
        
        # 7. 최종 통계 계산
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
        logger.info(f"   ⚡ 병렬 처리: concurrency={CONCURRENCY}, 청크={total_chunks}개")
        logger.info(f"   ⏱️  API 호출 시간: 평균 {avg_time:.2f}s, 최소 {min_time:.2f}s, 최대 {max_time:.2f}s")
        
        # 느린 요청들 로그
        if slow_businesses:
            logger.warning(f"   ⚠️  느린 요청 (3초 이상): {len(slow_businesses)}개")
            # 가장 느린 것부터 정렬
            slow_businesses.sort(key=lambda x: x["time"], reverse=True)
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

