"""
클라우드 Railway PostgreSQL에 리뷰 추가 스크립트
"""

import sys
import csv
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))

import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_csv(csv_path):
    """CSV 파일 파싱"""
    reviews = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            username = row.get('username', '').strip()
            restaurant_name = row.get('restaurant_name', '').strip()
            stars_str = row.get('stars', '').strip()
            text = row.get('text', '').strip()
            date_str = row.get('date', '').strip()
            
            # 필수 필드 검증
            if not username:
                logger.warning(f"[{idx}] ❌ username 누락. 건너뜀")
                continue
            
            if not restaurant_name:
                logger.warning(f"[{idx}] ❌ restaurant_name 누락. 건너뜀")
                continue
            
            if not text:
                logger.warning(f"[{idx}] ❌ text 누락. 건너뜀")
                continue
            
            # stars 검증 (1-5)
            try:
                stars = float(stars_str)
                if stars < 1.0 or stars > 5.0:
                    logger.warning(f"[{idx}] ❌ stars는 1.0~5.0 사이여야 합니다: {stars}. 건너뜀")
                    continue
            except ValueError:
                logger.warning(f"[{idx}] ❌ stars 형식 오류: {stars_str}. 건너뜀")
                continue
            
            # date 파싱 (선택)
            review_date = None
            if date_str:
                try:
                    # 여러 날짜 형식 지원
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%Y-%m-%d %H:%M:%S']:
                        try:
                            review_date = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if not review_date:
                        logger.warning(f"[{idx}] ⚠️  날짜 형식 인식 실패: {date_str}. 날짜 없이 진행")
                except Exception as e:
                    logger.warning(f"[{idx}] ⚠️  날짜 파싱 오류: {e}. 날짜 없이 진행")
            
            reviews.append({
                'username': username,
                'restaurant_name': restaurant_name,
                'stars': stars,
                'text': text,
                'date': review_date,
            })
            
            logger.info(f"[{idx}] ✅ 파싱 성공: {username} -> {restaurant_name} ({stars}점)")
    
    return reviews


def add_reviews_to_cloud(reviews, database_url):
    """리뷰를 클라우드 DB에 추가"""
    # Railway DB 연결
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    added_count = 0
    skipped_count = 0
    
    try:
        for review in reviews:
            # 1. username으로 user 찾기
            user = db.query(models.User).filter(
                models.User.username == review['username']
            ).first()
            
            if not user:
                logger.warning(f"❌ 사용자를 찾을 수 없습니다: {review['username']}")
                skipped_count += 1
                continue
            
            # 2. restaurant_name으로 business 찾기
            business = db.query(models.Business).filter(
                models.Business.name == review['restaurant_name']
            ).first()
            
            if not business:
                logger.warning(f"❌ 음식점을 찾을 수 없습니다: {review['restaurant_name']}")
                skipped_count += 1
                continue
            
            # 3. 중복 리뷰 확인 (같은 user + business + 비슷한 내용)
            existing = db.query(models.Review).filter(
                models.Review.user_id == user.id,
                models.Review.business_id == business.id,
                models.Review.text == review['text']
            ).first()
            
            if existing:
                logger.info(f"⏭️  이미 존재하는 리뷰: {user.username} -> {business.name}")
                skipped_count += 1
                continue
            
            # 4. 리뷰 객체 생성
            db_review = models.Review(
                user_id=user.id,
                business_id=business.id,
                stars=review['stars'],
                text=review['text'],
                date=review['date'],
                useful=0,
                is_taste_test=False,
                taste_test_type=None,
                taste_test_weight=1.0,
                absa_features={}  # 빈 ABSA (나중에 계산)
            )
            
            db.add(db_review)
            added_count += 1
            logger.info(f"✅ 추가: {user.username} -> {business.name} ({review['stars']}점)")
        
        # 5. 커밋
        if added_count > 0:
            db.commit()
            logger.info(f"\n🎉 완료: {added_count}개 추가, {skipped_count}개 스킵")
            
            # 6. 음식점의 평균 별점 및 리뷰 수 업데이트
            logger.info("\n🔄 음식점 통계 업데이트 중...")
            update_business_stats(db)
        else:
            logger.info(f"\n⚠️  추가된 리뷰 없음. {skipped_count}개 스킵")
        
    except Exception as e:
        logger.error(f"❌ 에러 발생: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def update_business_stats(db):
    """음식점의 평균 별점과 리뷰 수 업데이트"""
    from sqlalchemy import func
    
    # 모든 음식점에 대해 평균 별점과 리뷰 수 계산
    stats = db.query(
        models.Review.business_id,
        func.avg(models.Review.stars).label('avg_stars'),
        func.count(models.Review.id).label('review_count')
    ).filter(
        models.Review.business_id.isnot(None)
    ).group_by(
        models.Review.business_id
    ).all()
    
    for stat in stats:
        business = db.query(models.Business).filter(
            models.Business.id == stat.business_id
        ).first()
        
        if business:
            old_stars = business.stars
            old_count = business.review_count
            business.stars = round(stat.avg_stars, 2)
            business.review_count = stat.review_count
            
            logger.info(f"  📊 {business.name}: {old_stars}점({old_count}개) -> {business.stars}점({business.review_count}개)")
    
    db.commit()
    logger.info("✅ 통계 업데이트 완료")


def main():
    # 환경변수에서 설정 가져오기
    database_url = os.getenv("RAILWAY_DATABASE_URL")
    
    if not database_url:
        logger.error("❌ RAILWAY_DATABASE_URL 환경변수가 설정되지 않았습니다.")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        logger.error("❌ CSV 파일 경로를 제공해주세요.")
        logger.error("사용법: python scripts/add_reviews_to_cloud.py data/reviews_to_add.csv")
        logger.error("\nCSV 형식:")
        logger.error("  username,restaurant_name,stars,text,date")
        logger.error("  testuser,우만동족발집 아주대점,5.0,정말 맛있어요!,2025-01-15")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("🚀 클라우드 Railway DB에 리뷰 추가")
    logger.info("=" * 60)
    logger.info(f"📁 CSV 파일: {csv_path}")
    logger.info(f"🗄️  DB: {database_url.split('@')[1].split('/')[0]}")  # 호스트만 표시
    logger.info("=" * 60)
    
    logger.info(f"\n📖 CSV 파일 읽는 중: {csv_path}")
    reviews = parse_csv(csv_path)
    logger.info(f"✅ {len(reviews)}개 리뷰 발견\n")
    
    if not reviews:
        logger.error("❌ 추가할 리뷰가 없습니다.")
        sys.exit(1)
    
    logger.info("🔄 DB에 추가 중...")
    add_reviews_to_cloud(reviews, database_url)
    
    logger.info("\n✅ 모든 작업 완료!")


if __name__ == "__main__":
    main()

