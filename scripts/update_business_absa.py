"""
Business ABSA 재집계 스크립트
기존 음식점의 ABSA features를 리뷰로부터 다시 계산합니다.

사용법:
    # 모든 음식점 업데이트
    python scripts/update_business_absa.py --all
    
    # 수원시 음식점만 업데이트
    python scripts/update_business_absa.py --suwon
    
    # 특정 business_id 업데이트
    python scripts/update_business_absa.py --business-id BUSINESS_ID_HERE
"""

import sys
import os
import argparse
from pathlib import Path
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
from collections import defaultdict

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))

import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_business_absa_from_reviews(business, db):
    """
    특정 비즈니스의 ABSA를 리뷰로부터 재계산
    
    Args:
        business: Business 객체
        db: 데이터베이스 세션
    
    Returns:
        bool: 업데이트 성공 여부
    """
    # 해당 비즈니스의 모든 리뷰 조회 (ABSA가 있는 것만)
    reviews = db.query(models.Review).filter(
        models.Review.business_id == business.id,
        models.Review.absa_features.isnot(None)
    ).all()
    
    if not reviews:
        logger.warning(f"  ⚠️  {business.name}: ABSA가 있는 리뷰가 없습니다. (총 리뷰: {business.review_count}개)")
        return False
    
    # ABSA 평균 계산
    absa_sum = defaultdict(float)
    total_stars = 0.0
    
    for review in reviews:
        # ABSA 합산
        for key, value in review.absa_features.items():
            absa_sum[key] += value
        
        # 별점 합산
        if review.stars:
            total_stars += review.stars
    
    # ABSA 평균 저장
    business.absa_features = {
        key: value / len(reviews) 
        for key, value in absa_sum.items()
    }
    
    # 별점 평균 및 리뷰 수 업데이트
    business.stars = total_stars / len(reviews)
    business.review_count = len(reviews)
    
    db.commit()
    
    logger.info(f"  ✅ {business.name}: {len(reviews)}개 리뷰로부터 ABSA 업데이트 (평균 별점: {business.stars:.2f})")
    return True


def update_all_businesses(db):
    """모든 비즈니스의 ABSA 업데이트"""
    logger.info("\n🔄 모든 음식점의 ABSA 재집계 시작...")
    logger.info("-" * 80)
    
    businesses = db.query(models.Business).all()
    
    logger.info(f"총 {len(businesses)}개 음식점 발견\n")
    
    success_count = 0
    skip_count = 0
    
    for idx, business in enumerate(businesses, 1):
        logger.info(f"[{idx}/{len(businesses)}] {business.name}")
        
        if update_business_absa_from_reviews(business, db):
            success_count += 1
        else:
            skip_count += 1
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ 완료: {success_count}개 업데이트, {skip_count}개 스킵")
    logger.info("=" * 80)


def update_suwon_businesses(db):
    """수원시 비즈니스만 ABSA 업데이트"""
    logger.info("\n🔄 수원시 음식점의 ABSA 재집계 시작...")
    logger.info("-" * 80)
    
    businesses = db.query(models.Business).filter(
        and_(
            models.Business.latitude.between(37.2, 37.3),
            models.Business.longitude.between(126.9, 127.1)
        )
    ).all()
    
    logger.info(f"수원시 {len(businesses)}개 음식점 발견\n")
    
    if not businesses:
        logger.warning("⚠️  수원시 음식점이 없습니다.")
        return
    
    success_count = 0
    skip_count = 0
    
    for idx, business in enumerate(businesses, 1):
        logger.info(f"[{idx}/{len(businesses)}] {business.name}")
        
        if update_business_absa_from_reviews(business, db):
            success_count += 1
        else:
            skip_count += 1
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ 완료: {success_count}개 업데이트, {skip_count}개 스킵")
    logger.info("=" * 80)


def update_specific_business(business_id, db):
    """특정 비즈니스의 ABSA 업데이트"""
    logger.info(f"\n🔄 비즈니스 ID '{business_id}'의 ABSA 재집계 시작...")
    logger.info("-" * 80)
    
    business = db.query(models.Business).filter(
        models.Business.business_id == business_id
    ).first()
    
    if not business:
        logger.error(f"❌ 비즈니스 ID '{business_id}'를 찾을 수 없습니다.")
        return
    
    logger.info(f"음식점: {business.name}")
    logger.info(f"주소: {business.address or business.city}\n")
    
    if update_business_absa_from_reviews(business, db):
        logger.info("\n" + "=" * 80)
        logger.info("✅ 업데이트 완료")
        logger.info("=" * 80)
    else:
        logger.info("\n" + "=" * 80)
        logger.info("⚠️  업데이트 스킵")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Business ABSA 재집계 스크립트")
    parser.add_argument("--all", action="store_true", help="모든 음식점 업데이트")
    parser.add_argument("--suwon", action="store_true", help="수원시 음식점만 업데이트")
    parser.add_argument("--business-id", type=str, help="특정 비즈니스 ID 업데이트")
    
    args = parser.parse_args()
    
    # 환경변수에서 DB URL 가져오기
    database_url = os.getenv("RAILWAY_DATABASE_URL")
    
    if not database_url:
        logger.error("❌ RAILWAY_DATABASE_URL 환경변수가 설정되지 않았습니다.")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("🚀 Business ABSA 재집계 스크립트")
    logger.info("=" * 80)
    logger.info(f"🗄️  DB: {database_url.split('@')[1].split('/')[0]}")  # 호스트만 표시
    logger.info("=" * 80)
    
    # DB 연결
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        if args.all:
            update_all_businesses(db)
        elif args.suwon:
            update_suwon_businesses(db)
        elif args.business_id:
            update_specific_business(args.business_id, db)
        else:
            logger.error("❌ 옵션을 선택해주세요: --all, --suwon, 또는 --business-id")
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

