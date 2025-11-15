"""
간단한 CSV 기반 마이그레이션
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
from models import User, Business, Review
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


def export_to_csv():
    """로컬 DB를 CSV로 export"""
    logger.info("📤 로컬 DB에서 CSV로 export 중...")
    
    engine = create_engine(LOCAL_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    os.makedirs("temp_export", exist_ok=True)
    
    # Users export
    logger.info("  - Users 테이블 export...")
    users = session.query(User).all()
    with open("temp_export/users.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["yelp_user_id", "username", "email", "hashed_password", "created_at",
                        "review_count", "useful", "compliment", "fans", "average_stars",
                        "yelping_since_days", "absa_features", "text_embedding"])
        for user in users:
            writer.writerow([
                user.yelp_user_id, user.username, user.email, user.hashed_password, user.created_at,
                user.review_count, user.useful, user.compliment, user.fans, user.average_stars,
                user.yelping_since_days,
                json.dumps(user.absa_features) if user.absa_features else "null",
                json.dumps(user.text_embedding) if user.text_embedding else "null"
            ])
    logger.info(f"    ✅ {len(users):,}명 export 완료")
    
    # Businesses export
    logger.info("  - Businesses 테이블 export...")
    businesses = session.query(Business).all()
    with open("temp_export/businesses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["business_id", "name", "categories", "stars", "review_count",
                        "address", "city", "state", "latitude", "longitude", "absa_features"])
        for biz in businesses:
            writer.writerow([
                biz.business_id, biz.name, biz.categories, biz.stars, biz.review_count,
                biz.address, biz.city, biz.state, biz.latitude, biz.longitude,
                json.dumps(biz.absa_features) if biz.absa_features else "null"
            ])
    logger.info(f"    ✅ {len(businesses):,}개 export 완료")
    
    # Reviews export
    logger.info("  - Reviews 테이블 export...")
    reviews = session.query(Review).all()
    with open("temp_export/reviews.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "user_id", "business_id", "stars", "useful", "funny", "cool",
                        "text", "date", "absa_features", "text_embedding"])
        for review in reviews:
            writer.writerow([
                review.review_id, review.user_id, review.business_id, review.stars,
                review.useful, review.funny, review.cool, review.text, review.date,
                json.dumps(review.absa_features) if review.absa_features else "null",
                json.dumps(review.text_embedding) if review.text_embedding else "null"
            ])
    logger.info(f"    ✅ {len(reviews):,}개 export 완료")
    
    session.close()
    return len(users), len(businesses), len(reviews)


def import_from_csv():
    """CSV를 Railway DB로 import"""
    logger.info("\n📥 CSV에서 Railway DB로 import 중...")
    
    engine = create_engine(RAILWAY_URL, connect_args={"connect_timeout": 30})
    conn = engine.raw_connection()
    cursor = conn.cursor()
    
    try:
        # Users import
        logger.info("  - Users 테이블 import...")
        with open("temp_export/users.csv", "r", encoding="utf-8") as f:
            cursor.copy_expert("""
                COPY users (yelp_user_id, username, email, hashed_password, created_at,
                           review_count, useful, compliment, fans, average_stars,
                           yelping_since_days, absa_features, text_embedding)
                FROM STDIN WITH CSV HEADER
            """, f)
        conn.commit()
        logger.info("    ✅ Users import 완료")
        
        # Businesses import
        logger.info("  - Businesses 테이블 import...")
        with open("temp_export/businesses.csv", "r", encoding="utf-8") as f:
            cursor.copy_expert("""
                COPY businesses (business_id, name, categories, stars, review_count,
                               address, city, state, latitude, longitude, absa_features)
                FROM STDIN WITH CSV HEADER
            """, f)
        conn.commit()
        logger.info("    ✅ Businesses import 완료")
        
        # Reviews import
        logger.info("  - Reviews 테이블 import...")
        with open("temp_export/reviews.csv", "r", encoding="utf-8") as f:
            cursor.copy_expert("""
                COPY reviews (review_id, user_id, business_id, stars, useful, funny, cool,
                            text, date, absa_features, text_embedding)
                FROM STDIN WITH CSV HEADER
            """, f)
        conn.commit()
        logger.info("    ✅ Reviews import 완료")
        
        logger.info("\n🎉 마이그레이션 완료!")
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
    logger.info("🚀 간단 DB 마이그레이션 시작")
    logger.info("="*60)
    
    # 1. Export
    user_count, biz_count, review_count = export_to_csv()
    
    # 2. Import
    success = import_from_csv()
    
    # 3. Cleanup
    logger.info("\n🧹 임시 파일 정리...")
    import shutil
    shutil.rmtree("temp_export")
    
    if success:
        logger.info("\n✅ 모든 작업 완료!")
        logger.info(f"   - Users: {user_count:,}")
        logger.info(f"   - Businesses: {biz_count:,}")
        logger.info(f"   - Reviews: {review_count:,}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

