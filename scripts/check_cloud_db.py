"""
클라우드 Railway PostgreSQL DB 상태 확인 스크립트
수원시 음식점의 데이터 상태를 진단합니다.
"""

import sys
import os
from pathlib import Path
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))

import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_cloud_db():
    """클라우드 DB 상태 확인"""
    
    # 환경변수에서 DB URL 가져오기
    database_url = os.getenv("RAILWAY_DATABASE_URL")
    
    if not database_url:
        logger.error("❌ RAILWAY_DATABASE_URL 환경변수가 설정되지 않았습니다.")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("🔍 클라우드 DB 상태 확인")
    logger.info("=" * 80)
    logger.info(f"🗄️  DB: {database_url.split('@')[1].split('/')[0]}")  # 호스트만 표시
    logger.info("=" * 80)
    
    # DB 연결
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        # 1. 수원시 음식점 조회 (latitude 37.2-37.3, longitude 126.9-127.1)
        logger.info("\n📍 수원시 음식점 조회 (위도 37.2-37.3, 경도 126.9-127.1)")
        logger.info("-" * 80)
        
        suwon_businesses = db.query(models.Business).filter(
            and_(
                models.Business.latitude.between(37.2, 37.3),
                models.Business.longitude.between(126.9, 127.1)
            )
        ).all()
        
        logger.info(f"✅ 수원시 음식점 총 {len(suwon_businesses)}개 발견\n")
        
        if not suwon_businesses:
            logger.warning("⚠️  수원시 음식점이 없습니다.")
            return
        
        # 2. 각 음식점 상세 정보 확인
        logger.info("📊 각 음식점 상세 정보:")
        logger.info("-" * 80)
        
        for idx, business in enumerate(suwon_businesses, 1):
            logger.info(f"\n[{idx}] {business.name}")
            logger.info(f"    ID: {business.business_id}")
            logger.info(f"    주소: {business.address or business.city}")
            logger.info(f"    위치: ({business.latitude:.4f}, {business.longitude:.4f})")
            logger.info(f"    별점: {business.stars:.1f} / 리뷰 수: {business.review_count}")
            
            # ABSA features 확인
            has_absa = business.absa_features is not None and len(business.absa_features) > 0
            logger.info(f"    ABSA features: {'✅ 있음' if has_absa else '❌ 없음'}")
            
            if has_absa:
                # 주요 ABSA 특성 표시
                absa = business.absa_features
                food_pos = absa.get('음식_긍정', 0)
                service_pos = absa.get('서비스_긍정', 0)
                atmosphere_pos = absa.get('분위기_긍정', 0)
                logger.info(f"      - 음식 긍정: {food_pos:.2f}")
                logger.info(f"      - 서비스 긍정: {service_pos:.2f}")
                logger.info(f"      - 분위기 긍정: {atmosphere_pos:.2f}")
            
            # 리뷰 확인
            reviews = db.query(models.Review).filter(
                models.Review.business_id == business.id
            ).all()
            
            logger.info(f"    실제 리뷰: {len(reviews)}개")
            
            if reviews:
                reviews_with_absa = [r for r in reviews if r.absa_features is not None and len(r.absa_features) > 0]
                logger.info(f"      - ABSA가 있는 리뷰: {len(reviews_with_absa)}개")
                logger.info(f"      - ABSA가 없는 리뷰: {len(reviews) - len(reviews_with_absa)}개")
                
                # 리뷰 작성자 정보
                for review in reviews[:3]:  # 최대 3개만 표시
                    user = db.query(models.User).filter(models.User.id == review.user_id).first()
                    has_review_absa = review.absa_features is not None and len(review.absa_features) > 0
                    logger.info(f"      - [{user.username if user else 'Unknown'}] {review.stars}점, ABSA: {'✅' if has_review_absa else '❌'}")
            else:
                logger.info(f"      ⚠️  리뷰가 없습니다.")
            
            # AI 예측 캐시 확인
            predictions = db.query(models.UserBusinessPrediction).filter(
                models.UserBusinessPrediction.business_id == business.id
            ).all()
            
            logger.info(f"    AI 예측 캐시: {len(predictions)}개 사용자")
            
            if predictions:
                for pred in predictions[:3]:  # 최대 3개만 표시
                    user = db.query(models.User).filter(models.User.id == pred.user_id).first()
                    stale_status = "🔄 재계산 필요" if pred.is_stale else "✅ 최신"
                    logger.info(f"      - [{user.username if user else 'Unknown'}] DeepFM: {pred.deepfm_score:.2f}, {stale_status}")
        
        # 3. 전체 통계
        logger.info("\n" + "=" * 80)
        logger.info("📈 전체 통계")
        logger.info("=" * 80)
        
        total_businesses_with_absa = sum(1 for b in suwon_businesses if b.absa_features and len(b.absa_features) > 0)
        total_reviews = sum(db.query(models.Review).filter(models.Review.business_id == b.id).count() for b in suwon_businesses)
        total_reviews_with_absa = sum(
            len([r for r in db.query(models.Review).filter(models.Review.business_id == b.id).all() 
                 if r.absa_features and len(r.absa_features) > 0])
            for b in suwon_businesses
        )
        total_predictions = sum(
            db.query(models.UserBusinessPrediction).filter(models.UserBusinessPrediction.business_id == b.id).count()
            for b in suwon_businesses
        )
        
        logger.info(f"수원시 음식점: {len(suwon_businesses)}개")
        logger.info(f"  - ABSA가 있는 음식점: {total_businesses_with_absa}개 ({total_businesses_with_absa/len(suwon_businesses)*100:.1f}%)")
        logger.info(f"  - ABSA가 없는 음식점: {len(suwon_businesses) - total_businesses_with_absa}개")
        logger.info(f"\n총 리뷰: {total_reviews}개")
        logger.info(f"  - ABSA가 있는 리뷰: {total_reviews_with_absa}개 ({total_reviews_with_absa/total_reviews*100:.1f}% if total_reviews > 0 else 0)")
        logger.info(f"  - ABSA가 없는 리뷰: {total_reviews - total_reviews_with_absa}개")
        logger.info(f"\nAI 예측 캐시: {total_predictions}개")
        
        # 4. 권장 사항
        logger.info("\n" + "=" * 80)
        logger.info("💡 권장 사항")
        logger.info("=" * 80)
        
        if total_businesses_with_absa < len(suwon_businesses):
            logger.warning(f"⚠️  {len(suwon_businesses) - total_businesses_with_absa}개 음식점의 ABSA가 없습니다.")
            logger.warning(f"   👉 'python scripts/update_business_absa.py' 실행을 권장합니다.")
        
        if total_reviews_with_absa < total_reviews:
            logger.warning(f"⚠️  {total_reviews - total_reviews_with_absa}개 리뷰의 ABSA가 없습니다.")
            logger.warning(f"   👉 리뷰 작성 시 백그라운드 ABSA 분석이 실패했을 가능성이 있습니다.")
        
        if total_predictions == 0:
            logger.warning(f"⚠️  AI 예측 캐시가 없습니다.")
            logger.warning(f"   👉 사용자가 로그인하면 자동으로 생성됩니다.")
        
        if total_businesses_with_absa == len(suwon_businesses):
            logger.info("✅ 모든 음식점의 ABSA가 정상적으로 설정되어 있습니다!")
        
    except Exception as e:
        logger.error(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ DB 확인 완료")
    logger.info("=" * 80)


if __name__ == "__main__":
    check_cloud_db()

