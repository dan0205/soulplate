"""
Reviews만 마이그레이션
로컬 DB → CSV → Railway DB
"""

import sys
import os
from pathlib import Path
import csv
import json
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Review
import logging
from tqdm import tqdm

# .env 파일 로드
load_dotenv(project_root / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_URL = os.getenv("LOCAL_DATABASE_URL")
RAILWAY_URL = os.getenv("RAILWAY_DATABASE_URL")

if not LOCAL_URL or not RAILWAY_URL:
    raise ValueError("환경 변수 LOCAL_DATABASE_URL 및 RAILWAY_DATABASE_URL이 설정되지 않았습니다.")


def export_reviews():
    """로컬 DB의 Reviews를 CSV로 export"""
    logger.info("📤 로컬 DB에서 Reviews를 CSV로 export 중...")
    
    engine = create_engine(LOCAL_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    os.makedirs("temp_export", exist_ok=True)
    
    # Reviews export
    reviews = session.query(Review).all()
    logger.info(f"  - {len(reviews):,}개의 Reviews를 export 중...")
    
    with open("temp_export/reviews.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "user_id", "business_id", "stars", "useful",
                        "text", "date", "absa_features", "created_at",
                        "is_taste_test", "taste_test_type", "taste_test_weight"])
        for review in tqdm(reviews, desc="Export"):
            writer.writerow([
                review.id, review.user_id, review.business_id, review.stars,
                review.useful, review.text, review.date,
                json.dumps(review.absa_features) if review.absa_features else "null",
                review.created_at,
                review.is_taste_test, review.taste_test_type, review.taste_test_weight
            ])
    
    logger.info(f"✅ {len(reviews):,}개 export 완료")
    session.close()
    return len(reviews)


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
    logger.info("🚀 Reviews 마이그레이션 시작")
    logger.info("="*60)
    
    # 1. Export
    review_count = export_reviews()
    
    # 2. Import
    success = import_reviews()
    
    # 3. Cleanup
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

