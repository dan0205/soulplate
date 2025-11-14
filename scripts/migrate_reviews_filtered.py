"""
Reviews 마이그레이션 (Foreign Key 필터링)
Railway DB에 있는 Users/Businesses만 참조하는 Reviews만 마이그레이션
"""

import sys
import os
from pathlib import Path
import csv
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Review, User, Business
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_URL = "postgresql://two_tower_user:twotower2024@localhost:5432/two_tower_db"
RAILWAY_URL = "postgresql://postgres:fYHkhuVDnSfOqBOmpAEqigXEsqlRIDEX@crossover.proxy.rlwy.net:47399/railway"


def get_valid_ids():
    """Railway DB에 있는 유효한 user_id와 business_id를 가져옴"""
    logger.info("📋 Railway DB에서 유효한 user_id와 business_id 가져오는 중...")
    
    engine = create_engine(RAILWAY_URL, connect_args={"connect_timeout": 30})
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Railway DB의 유효한 user_id들
    valid_user_ids = set([user.id for user in session.query(User.id).all()])
    logger.info(f"  - 유효한 Users: {len(valid_user_ids):,}명")
    
    # Railway DB의 유효한 business_id들
    valid_business_ids = set([biz.id for biz in session.query(Business.id).all()])
    logger.info(f"  - 유효한 Businesses: {len(valid_business_ids):,}개")
    
    session.close()
    return valid_user_ids, valid_business_ids


def export_filtered_reviews(valid_user_ids, valid_business_ids):
    """로컬 DB에서 유효한 Foreign Key를 가진 Reviews만 export"""
    logger.info("\n📤 로컬 DB에서 필터링된 Reviews를 CSV로 export 중...")
    
    engine = create_engine(LOCAL_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    os.makedirs("temp_export", exist_ok=True)
    
    # 모든 Reviews 가져오기
    all_reviews = session.query(Review).all()
    logger.info(f"  - 전체 Reviews: {len(all_reviews):,}개")
    
    # Foreign Key 필터링
    filtered_reviews = []
    for review in all_reviews:
        # user_id가 유효하고, business_id가 None이거나 유효한 경우만
        if review.user_id in valid_user_ids and (review.business_id is None or review.business_id in valid_business_ids):
            filtered_reviews.append(review)
    
    logger.info(f"  - 필터링된 Reviews: {len(filtered_reviews):,}개 (유효한 Foreign Key만)")
    
    with open("temp_export/reviews.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "user_id", "business_id", "stars", "useful",
                        "text", "date", "absa_features", "created_at",
                        "is_taste_test", "taste_test_type", "taste_test_weight"])
        for review in tqdm(filtered_reviews, desc="Export"):
            writer.writerow([
                review.id, review.user_id, review.business_id, review.stars,
                review.useful, review.text, review.date,
                json.dumps(review.absa_features) if review.absa_features else "null",
                review.created_at,
                review.is_taste_test, review.taste_test_type, review.taste_test_weight
            ])
    
    logger.info(f"✅ {len(filtered_reviews):,}개 export 완료")
    session.close()
    return len(filtered_reviews)


def import_reviews():
    """CSV를 Railway DB로 import"""
    logger.info("\n📥 CSV에서 Railway DB로 Reviews import 중...")
    
    engine = create_engine(RAILWAY_URL, connect_args={"connect_timeout": 30})
    conn = engine.raw_connection()
    cursor = conn.cursor()
    
    try:
        with open("temp_export/reviews.csv", "r", encoding="utf-8") as f:
            cursor.copy_expert("""
                COPY reviews (id, user_id, business_id, stars, useful,
                            text, date, absa_features, created_at,
                            is_taste_test, taste_test_type, taste_test_weight)
                FROM STDIN WITH CSV HEADER
            """, f)
        conn.commit()
        logger.info("✅ Reviews import 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import 실패: {e}")
        conn.rollback()
        return False
        
    finally:
        cursor.close()
        conn.close()


def main():
    logger.info("="*60)
    logger.info("🚀 Reviews 마이그레이션 시작 (Foreign Key 필터링)")
    logger.info("="*60)
    
    # 1. Railway DB에서 유효한 ID들 가져오기
    valid_user_ids, valid_business_ids = get_valid_ids()
    
    # 2. 필터링된 Reviews Export
    review_count = export_filtered_reviews(valid_user_ids, valid_business_ids)
    
    # 3. Import
    success = import_reviews()
    
    # 4. Cleanup
    logger.info("\n🧹 임시 파일 정리...")
    import shutil
    shutil.rmtree("temp_export")
    
    if success:
        logger.info("\n✅ 마이그레이션 완료!")
        logger.info(f"   - Reviews: {review_count:,}개")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

