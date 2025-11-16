"""
리뷰 ABSA 재분석 스크립트
ABSA가 없는 리뷰들을 찾아서 MODEL API에 재분석 요청

사용법:
    # 모든 ABSA 없는 리뷰 재분석
    python scripts/reanalyze_reviews_absa.py --all
    
    # 특정 사용자의 리뷰만 재분석
    python scripts/reanalyze_reviews_absa.py --username admin
    
    # 수원시 음식점의 리뷰만 재분석
    python scripts/reanalyze_reviews_absa.py --suwon
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
from collections import defaultdict
import httpx

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))

import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def analyze_review_absa(review_text: str, model_api_url: str):
    """
    MODEL API를 호출하여 리뷰 ABSA 분석
    
    Args:
        review_text: 리뷰 텍스트
        model_api_url: MODEL API URL
    
    Returns:
        dict: ABSA features 또는 None (실패 시)
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{model_api_url}/analyze_review",
                json={"text": review_text}
            )
            
            logger.info(f"  HTTP 상태: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                absa_features = result.get("absa_features")
                if absa_features:
                    return absa_features
                else:
                    logger.error(f"  응답에 absa_features가 없음: {result}")
                    return None
            else:
                logger.error(f"  HTTP {response.status_code} 에러")
                logger.error(f"  응답 본문: {response.text[:200]}")
                return None
                
    except httpx.TimeoutException:
        logger.error(f"  타임아웃: 60초 이내 응답 없음")
        return None
    except httpx.ConnectError as e:
        logger.error(f"  연결 실패: {e}")
        logger.error(f"  MODEL_API_URL이 올바른지 확인하세요: {model_api_url}")
        return None
    except Exception as e:
        logger.error(f"  예외 발생: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"  상세:\n{traceback.format_exc()}")
        return None


def update_business_profile(business_id: int, db):
    """비즈니스 프로필 업데이트"""
    reviews = db.query(models.Review).filter(
        models.Review.business_id == business_id,
        models.Review.absa_features.isnot(None)
    ).all()
    
    if not reviews:
        return False
    
    absa_sum = defaultdict(float)
    total_stars = 0.0
    
    for review in reviews:
        for key, value in review.absa_features.items():
            absa_sum[key] += value
        if review.stars:
            total_stars += review.stars
    
    business = db.query(models.Business).filter(models.Business.id == business_id).first()
    if business:
        business.absa_features = {
            key: value / len(reviews) 
            for key, value in absa_sum.items()
        }
        business.stars = total_stars / len(reviews)
        business.review_count = len(reviews)
        db.commit()
        return True
    
    return False


async def reanalyze_reviews(reviews, model_api_url, db):
    """
    리뷰들의 ABSA 재분석
    
    Args:
        reviews: 재분석할 리뷰 리스트
        model_api_url: MODEL API URL
        db: 데이터베이스 세션
    """
    success_count = 0
    fail_count = 0
    updated_businesses = set()
    
    for idx, review in enumerate(reviews, 1):
        user = db.query(models.User).filter(models.User.id == review.user_id).first()
        business = db.query(models.Business).filter(models.Business.id == review.business_id).first()
        
        username = user.username if user else "Unknown"
        business_name = business.name if business else "Unknown"
        
        logger.info(f"[{idx}/{len(reviews)}] {username} → {business_name}")
        logger.info(f"  리뷰: {review.text[:50]}...")
        
        # ABSA 분석 요청
        absa_features = await analyze_review_absa(review.text, model_api_url)
        
        if absa_features:
            # Review에 ABSA 저장
            review.absa_features = absa_features
            db.commit()
            
            success_count += 1
            updated_businesses.add(review.business_id)
            
            logger.info(f"  ✅ ABSA 분석 완료 ({len(absa_features)}개 특성)")
        else:
            fail_count += 1
            logger.error(f"  ❌ ABSA 분석 실패")
    
    # 업데이트된 비즈니스들의 ABSA 재계산
    logger.info(f"\n🔄 {len(updated_businesses)}개 비즈니스의 ABSA 재계산 중...")
    
    for business_id in updated_businesses:
        business = db.query(models.Business).filter(models.Business.id == business_id).first()
        if business:
            if update_business_profile(business_id, db):
                logger.info(f"  ✅ {business.name} 업데이트 완료")
            else:
                logger.warning(f"  ⚠️  {business.name} 업데이트 스킵")
    
    return success_count, fail_count


async def reanalyze_all(db, model_api_url):
    """모든 ABSA 없는 리뷰 재분석"""
    logger.info("\n🔄 ABSA가 없는 모든 리뷰 재분석 시작...")
    logger.info("-" * 80)
    
    reviews = db.query(models.Review).filter(
        models.Review.absa_features.is_(None)
    ).all()
    
    logger.info(f"ABSA가 없는 리뷰: {len(reviews)}개\n")
    
    if not reviews:
        logger.info("✅ 재분석할 리뷰가 없습니다.")
        return
    
    success, fail = await reanalyze_reviews(reviews, model_api_url, db)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ 완료: {success}개 성공, {fail}개 실패")
    logger.info("=" * 80)


async def reanalyze_by_username(username, db, model_api_url):
    """특정 사용자의 ABSA 없는 리뷰 재분석"""
    logger.info(f"\n🔄 사용자 '{username}'의 ABSA 없는 리뷰 재분석 시작...")
    logger.info("-" * 80)
    
    user = db.query(models.User).filter(models.User.username == username).first()
    
    if not user:
        logger.error(f"❌ 사용자 '{username}'를 찾을 수 없습니다.")
        return
    
    reviews = db.query(models.Review).filter(
        models.Review.user_id == user.id,
        models.Review.absa_features.is_(None),
        models.Review.business_id.isnot(None)  # 취향 테스트 제외
    ).all()
    
    logger.info(f"사용자: {username} (ID: {user.id})")
    logger.info(f"ABSA가 없는 리뷰: {len(reviews)}개\n")
    
    if not reviews:
        logger.info("✅ 재분석할 리뷰가 없습니다.")
        return
    
    success, fail = await reanalyze_reviews(reviews, model_api_url, db)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ 완료: {success}개 성공, {fail}개 실패")
    logger.info("=" * 80)


async def reanalyze_suwon(db, model_api_url):
    """수원시 음식점의 ABSA 없는 리뷰 재분석"""
    logger.info("\n🔄 수원시 음식점의 ABSA 없는 리뷰 재분석 시작...")
    logger.info("-" * 80)
    
    # 수원시 음식점 조회
    suwon_businesses = db.query(models.Business).filter(
        and_(
            models.Business.latitude.between(37.2, 37.3),
            models.Business.longitude.between(126.9, 127.1)
        )
    ).all()
    
    if not suwon_businesses:
        logger.warning("⚠️  수원시 음식점이 없습니다.")
        return
    
    business_ids = [b.id for b in suwon_businesses]
    
    # ABSA 없는 리뷰 조회
    reviews = db.query(models.Review).filter(
        models.Review.business_id.in_(business_ids),
        models.Review.absa_features.is_(None)
    ).all()
    
    logger.info(f"수원시 음식점: {len(suwon_businesses)}개")
    logger.info(f"ABSA가 없는 리뷰: {len(reviews)}개\n")
    
    if not reviews:
        logger.info("✅ 재분석할 리뷰가 없습니다.")
        return
    
    success, fail = await reanalyze_reviews(reviews, model_api_url, db)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ 완료: {success}개 성공, {fail}개 실패")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="리뷰 ABSA 재분석 스크립트")
    parser.add_argument("--all", action="store_true", help="모든 ABSA 없는 리뷰 재분석")
    parser.add_argument("--username", type=str, help="특정 사용자의 리뷰만 재분석")
    parser.add_argument("--suwon", action="store_true", help="수원시 음식점의 리뷰만 재분석")
    
    args = parser.parse_args()
    
    # 환경변수 확인
    database_url = os.getenv("RAILWAY_DATABASE_URL")
    model_api_url = os.getenv("MODEL_API_URL")
    
    if not database_url:
        logger.error("❌ RAILWAY_DATABASE_URL 환경변수가 설정되지 않았습니다.")
        sys.exit(1)
    
    if not model_api_url:
        logger.error("❌ MODEL_API_URL 환경변수가 설정되지 않았습니다.")
        logger.info("   예: export MODEL_API_URL=https://backendmodel-production-xxxx.up.railway.app")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("🚀 리뷰 ABSA 재분석 스크립트")
    logger.info("=" * 80)
    logger.info(f"🗄️  DB: {database_url.split('@')[1].split('/')[0]}")
    logger.info(f"🤖 MODEL API: {model_api_url}")
    logger.info("=" * 80)
    
    # DB 연결
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        if args.all:
            asyncio.run(reanalyze_all(db, model_api_url))
        elif args.username:
            asyncio.run(reanalyze_by_username(args.username, db, model_api_url))
        elif args.suwon:
            asyncio.run(reanalyze_suwon(db, model_api_url))
        else:
            logger.error("❌ 옵션을 선택해주세요: --all, --username, 또는 --suwon")
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()

