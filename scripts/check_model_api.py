"""
MODEL API Health Check 스크립트
MODEL API 서버가 정상 작동하는지 확인합니다.

사용법:
    python scripts/check_model_api.py
"""

import sys
import os
import asyncio
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_model_api():
    """MODEL API 상태 확인"""
    
    # 환경변수에서 MODEL API URL 가져오기
    model_api_url = os.getenv("MODEL_API_URL")
    
    if not model_api_url:
        logger.error("❌ MODEL_API_URL 환경변수가 설정되지 않았습니다.")
        logger.info("   예: export MODEL_API_URL=https://backendmodel-production-xxxx.up.railway.app")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("🔍 MODEL API Health Check")
    logger.info("=" * 80)
    logger.info(f"🤖 MODEL API: {model_api_url}")
    logger.info("=" * 80)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. Root 엔드포인트 확인
            logger.info("\n1️⃣ Root 엔드포인트 확인 (GET /)")
            logger.info("-" * 80)
            
            try:
                response = await client.get(model_api_url)
                logger.info(f"   상태 코드: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"   ✅ 응답: {data}")
                else:
                    logger.error(f"   ❌ 실패: {response.text}")
            except Exception as e:
                logger.error(f"   ❌ 연결 실패: {e}")
                return False
            
            # 2. Health Check 엔드포인트
            logger.info("\n2️⃣ Health Check (GET /health)")
            logger.info("-" * 80)
            
            try:
                response = await client.get(f"{model_api_url}/health")
                logger.info(f"   상태 코드: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"   상태: {data.get('status')}")
                    logger.info(f"   DeepFM 로딩: {'✅' if data.get('deepfm_loaded') else '❌'}")
                    logger.info(f"   MultiTower 로딩: {'✅' if data.get('multitower_loaded') else '❌'}")
                    logger.info(f"   ABSA 로딩: {'✅' if data.get('absa_loaded') else '❌'}")
                    
                    if data.get('status') == 'healthy' and data.get('absa_loaded'):
                        logger.info("   ✅ 모든 모델이 정상 로딩됨")
                    else:
                        logger.error("   ❌ 일부 모델 로딩 실패")
                        return False
                else:
                    logger.error(f"   ❌ 실패: {response.text}")
                    return False
            except Exception as e:
                logger.error(f"   ❌ 연결 실패: {e}")
                return False
            
            # 3. ABSA 분석 테스트
            logger.info("\n3️⃣ ABSA 분석 테스트 (POST /analyze_review)")
            logger.info("-" * 80)
            
            test_text = "음식이 정말 맛있고 서비스도 친절했습니다. 분위기도 좋았어요!"
            logger.info(f"   테스트 리뷰: {test_text}")
            
            try:
                response = await client.post(
                    f"{model_api_url}/analyze_review",
                    json={"text": test_text},
                    timeout=60.0  # ABSA 분석은 시간이 걸릴 수 있음
                )
                
                logger.info(f"   상태 코드: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    absa_features = data.get('absa_features', {})
                    text_embedding = data.get('text_embedding', [])
                    
                    logger.info(f"   ✅ ABSA 분석 성공!")
                    logger.info(f"   ABSA 특성 수: {len(absa_features)}개")
                    logger.info(f"   텍스트 임베딩 차원: {len(text_embedding)}차원")
                    
                    # 주요 ABSA 특성 표시
                    if absa_features:
                        logger.info("\n   주요 ABSA 특성 (상위 5개):")
                        sorted_features = sorted(absa_features.items(), key=lambda x: x[1], reverse=True)
                        for key, value in sorted_features[:5]:
                            logger.info(f"     - {key}: {value:.4f}")
                    
                    return True
                else:
                    logger.error(f"   ❌ 실패 (HTTP {response.status_code})")
                    logger.error(f"   응답: {response.text}")
                    return False
                    
            except httpx.TimeoutException:
                logger.error("   ❌ 타임아웃: 60초 이내에 응답 없음")
                logger.error("   MODEL API 서버가 느리거나 응답하지 않습니다.")
                return False
            except Exception as e:
                logger.error(f"   ❌ 분석 실패: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    except Exception as e:
        logger.error(f"\n❌ 전체 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 함수"""
    success = await check_model_api()
    
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ MODEL API Health Check 성공!")
        logger.info("=" * 80)
        logger.info("\n💡 다음 단계:")
        logger.info("   python scripts/reanalyze_reviews_absa.py --username admin")
        sys.exit(0)
    else:
        logger.info("❌ MODEL API Health Check 실패")
        logger.info("=" * 80)
        logger.info("\n💡 문제 해결:")
        logger.info("   1. Railway 대시보드에서 backend_model 서비스가 실행 중인지 확인")
        logger.info("   2. MODEL_API_URL이 올바른지 확인")
        logger.info("   3. Railway 로그를 확인하여 에러 확인")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

