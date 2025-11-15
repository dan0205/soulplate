"""
전체 DB 마이그레이션 (ID 포함)
로컬 DB → CSV → Railway DB (ID를 유지하면서)
"""

import sys
import os
from pathlib import Path
import csv
import json
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import User, Business, Review, UserBusinessPrediction
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


def clear_railway_db():
    """Railway DB의 모든 데이터 삭제"""
    logger.info("🗑️  Railway DB 데이터 삭제 중...")
    
    engine = create_engine(RAILWAY_URL, connect_args={"connect_timeout": 30})
    conn = engine.raw_connection()
    cursor = conn.cursor()
    
    try:
        # Foreign Key 제약을 고려해서 순서대로 삭제
        cursor.execute("DELETE FROM reviews;")
        cursor.execute("DELETE FROM user_business_predictions;")  # 추가: predictions 테이블
        cursor.execute("DELETE FROM businesses;")
        cursor.execute("DELETE FROM users;")
        conn.commit()
        logger.info("✅ Railway DB 데이터 삭제 완료")
        return True
    except Exception as e:
        logger.error(f"❌ 삭제 실패: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def export_to_csv():
    """로컬 DB를 CSV로 export (ID 포함)"""
    logger.info("\n📤 로컬 DB에서 CSV로 export 중 (ID 포함)...")
    
    engine = create_engine(LOCAL_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    os.makedirs("temp_export", exist_ok=True)
    
    # Users export
    logger.info("  - Users 테이블 export...")
    users = session.query(User).all()
    with open("temp_export/users.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "yelp_user_id", "username", "email", "hashed_password", "created_at",
                        "review_count", "useful", "compliment", "fans", "average_stars",
                        "yelping_since_days", "absa_features", "text_embedding"])
        for user in tqdm(users, desc="Users"):
            writer.writerow([
                user.id, user.yelp_user_id, user.username, user.email, user.hashed_password, user.created_at,
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
        writer.writerow(["id", "business_id", "name", "categories", "stars", "review_count",
                        "address", "city", "state", "latitude", "longitude", "absa_features"])
        for biz in tqdm(businesses, desc="Businesses"):
            writer.writerow([
                biz.id, biz.business_id, biz.name, biz.categories, biz.stars, biz.review_count,
                biz.address, biz.city, biz.state, biz.latitude, biz.longitude,
                json.dumps(biz.absa_features) if biz.absa_features else "null"
            ])
    logger.info(f"    ✅ {len(businesses):,}개 export 완료")
    
    # Reviews export
    logger.info("  - Reviews 테이블 export...")
    reviews = session.query(Review).all()
    with open("temp_export/reviews.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "user_id", "business_id", "stars", "useful",
                        "text", "date", "absa_features", "created_at",
                        "is_taste_test", "taste_test_type", "taste_test_weight"])
        for review in tqdm(reviews, desc="Reviews"):
            writer.writerow([
                review.id, review.user_id, review.business_id, review.stars,
                review.useful, review.text, review.date,
                json.dumps(review.absa_features) if review.absa_features else "null",
                review.created_at,
                review.is_taste_test, review.taste_test_type, review.taste_test_weight
            ])
    logger.info(f"    ✅ {len(reviews):,}개 export 완료")
    
    # UserBusinessPredictions export
    logger.info("  - UserBusinessPredictions 테이블 export...")
    predictions = session.query(UserBusinessPrediction).all()
    with open("temp_export/predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "user_id", "business_id", "deepfm_score", "multitower_score",
                        "is_stale", "calculated_at", "created_at"])
        for pred in tqdm(predictions, desc="Predictions"):
            writer.writerow([
                pred.id, pred.user_id, pred.business_id, pred.deepfm_score, pred.multitower_score,
                pred.is_stale, pred.calculated_at, pred.created_at
            ])
    logger.info(f"    ✅ {len(predictions):,}개 export 완료")
    
    session.close()
    return len(users), len(businesses), len(reviews), len(predictions)


def import_from_csv():
    """CSV를 Railway DB로 import (ID 포함)"""
    logger.info("\n📥 CSV에서 Railway DB로 import 중 (ID 포함)...")
    
    engine = create_engine(RAILWAY_URL, connect_args={"connect_timeout": 30})
    conn = engine.raw_connection()
    cursor = conn.cursor()
    
    try:
        # Users import
        logger.info("  - Users 테이블 import...")
        with open("temp_export/users.csv", "r", encoding="utf-8") as f:
            cursor.copy_expert("""
                COPY users (id, yelp_user_id, username, email, hashed_password, created_at,
                           review_count, useful, compliment, fans, average_stars,
                           yelping_since_days, absa_features, text_embedding)
                FROM STDIN WITH CSV HEADER
            """, f)
        conn.commit()
        
        # Users 테이블의 sequence를 현재 최대 ID로 업데이트
        cursor.execute("SELECT setval('users_id_seq', (SELECT MAX(id) FROM users));")
        conn.commit()
        logger.info("    ✅ Users import 완료")
        
        # Businesses import
        logger.info("  - Businesses 테이블 import...")
        with open("temp_export/businesses.csv", "r", encoding="utf-8") as f:
            cursor.copy_expert("""
                COPY businesses (id, business_id, name, categories, stars, review_count,
                               address, city, state, latitude, longitude, absa_features)
                FROM STDIN WITH CSV HEADER
            """, f)
        conn.commit()
        
        # Businesses 테이블의 sequence 업데이트
        cursor.execute("SELECT setval('businesses_id_seq', (SELECT MAX(id) FROM businesses));")
        conn.commit()
        logger.info("    ✅ Businesses import 완료")
        
        # Reviews import
        logger.info("  - Reviews 테이블 import...")
        with open("temp_export/reviews.csv", "r", encoding="utf-8") as f:
            cursor.copy_expert("""
                COPY reviews (id, user_id, business_id, stars, useful,
                            text, date, absa_features, created_at,
                            is_taste_test, taste_test_type, taste_test_weight)
                FROM STDIN WITH CSV HEADER
            """, f)
        conn.commit()
        
        # Reviews 테이블의 sequence 업데이트
        cursor.execute("SELECT setval('reviews_id_seq', (SELECT MAX(id) FROM reviews));")
        conn.commit()
        logger.info("    ✅ Reviews import 완료")
        
        # UserBusinessPredictions import
        logger.info("  - UserBusinessPredictions 테이블 import...")
        with open("temp_export/predictions.csv", "r", encoding="utf-8") as f:
            cursor.copy_expert("""
                COPY user_business_predictions (id, user_id, business_id, deepfm_score, multitower_score,
                                                is_stale, calculated_at, created_at)
                FROM STDIN WITH CSV HEADER
            """, f)
        conn.commit()
        
        # UserBusinessPredictions 테이블의 sequence 업데이트
        cursor.execute("SELECT setval('user_business_predictions_id_seq', (SELECT MAX(id) FROM user_business_predictions));")
        conn.commit()
        logger.info("    ✅ UserBusinessPredictions import 완료")
        
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
    logger.info("🚀 전체 DB 마이그레이션 시작 (ID 유지)")
    logger.info("="*60)
    
    # 0. Railway DB 데이터 삭제
    if not clear_railway_db():
        logger.error("Railway DB 삭제 실패!")
        return 1
    
    # 1. Export
    user_count, biz_count, review_count, pred_count = export_to_csv()
    
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
        logger.info(f"   - Predictions: {pred_count:,}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

