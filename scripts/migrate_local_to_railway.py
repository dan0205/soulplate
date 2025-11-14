"""
로컬 PostgreSQL → Railway PostgreSQL 데이터 마이그레이션 스크립트

로컬 DB의 모든 데이터를 Railway DB로 안전하게 복사합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import Base, User, Business, Review, UserBusinessPrediction
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 로컬 PostgreSQL 연결 정보
LOCAL_DATABASE_URL = "postgresql://two_tower_user:twotower2024@localhost:5432/two_tower_db"

# Railway PostgreSQL 연결 정보 (외부 접속)
RAILWAY_DATABASE_URL = "postgresql://postgres:fYHkhuVDnSfOqBOmpAEqigXEsqlRIDEX@crossover.proxy.rlwy.net:47399/railway?connect_timeout=30&keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"


def test_connection(db_url, name):
    """데이터베이스 연결 테스트"""
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✅ {name} 연결 성공!")
            logger.info(f"   버전: {version[:50]}...")
            return True
    except Exception as e:
        logger.error(f"❌ {name} 연결 실패: {e}")
        return False


def create_tables(db_url):
    """Railway DB에 테이블 생성"""
    logger.info("📋 Railway DB에 테이블 생성 중...")
    try:
        engine = create_engine(db_url)
        Base.metadata.create_all(engine)
        logger.info("✅ 테이블 생성 완료!")
        
        # 생성된 테이블 확인
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]
            logger.info(f"   생성된 테이블: {', '.join(tables)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 테이블 생성 실패: {e}")
        return False


def get_table_counts(session, name):
    """각 테이블의 데이터 개수 확인"""
    counts = {
        'users': session.query(User).count(),
        'businesses': session.query(Business).count(),
        'reviews': session.query(Review).count(),
        'predictions': session.query(UserBusinessPrediction).count()
    }
    logger.info(f"{name} 데이터 개수:")
    for table, count in counts.items():
        logger.info(f"  - {table}: {count:,}")
    return counts


def migrate_data():
    """데이터 마이그레이션 실행"""
    logger.info("="*60)
    logger.info("🚀 로컬 → Railway DB 마이그레이션 시작")
    logger.info("="*60)
    
    # 1. 연결 테스트
    logger.info("\n1️⃣ 데이터베이스 연결 테스트")
    if not test_connection(LOCAL_DATABASE_URL, "로컬 DB"):
        return False
    if not test_connection(RAILWAY_DATABASE_URL, "Railway DB"):
        return False
    
    # 2. Railway DB에 테이블 생성
    logger.info("\n2️⃣ Railway DB 테이블 생성")
    if not create_tables(RAILWAY_DATABASE_URL):
        return False
    
    # 3. 세션 생성
    logger.info("\n3️⃣ 데이터베이스 세션 생성")
    local_engine = create_engine(LOCAL_DATABASE_URL)
    railway_engine = create_engine(
        RAILWAY_DATABASE_URL,
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        pool_recycle=300
    )
    
    LocalSession = sessionmaker(bind=local_engine)
    RailwaySession = sessionmaker(bind=railway_engine, autoflush=False)
    
    local_session = LocalSession()
    railway_session = RailwaySession()
    
    try:
        # 4. 로컬 DB 데이터 확인
        logger.info("\n4️⃣ 로컬 DB 데이터 확인")
        local_counts = get_table_counts(local_session, "로컬 DB")
        
        # 5. Railway DB 기존 데이터 확인
        logger.info("\n5️⃣ Railway DB 기존 데이터 확인")
        railway_counts = get_table_counts(railway_session, "Railway DB")
        
        # 6. Users 마이그레이션
        logger.info("\n6️⃣ Users 테이블 마이그레이션")
        users = local_session.query(User).all()
        batch_size = 50  # 배치 크기 줄임
        
        migrated_count = 0
        for i in tqdm(range(0, len(users), batch_size), desc="Users"):
            batch = users[i:i+batch_size]
            try:
                for user in batch:
                    # ORM 대신 직접 SQL 사용 (더 안정적)
                    railway_session.execute(text("""
                        INSERT INTO users (yelp_user_id, username, email, hashed_password, created_at, 
                                         review_count, useful, compliment, fans, average_stars, 
                                         yelping_since_days, absa_features, text_embedding)
                        VALUES (:yelp_user_id, :username, :email, :hashed_password, :created_at,
                               :review_count, :useful, :compliment, :fans, :average_stars,
                               :yelping_since_days, :absa_features::jsonb, :text_embedding::jsonb)
                        ON CONFLICT (username) DO NOTHING
                    """), {
                        'yelp_user_id': user.yelp_user_id,
                        'username': user.username,
                        'email': user.email,
                        'hashed_password': user.hashed_password,
                        'created_at': user.created_at,
                        'review_count': user.review_count,
                        'useful': user.useful,
                        'compliment': user.compliment,
                        'fans': user.fans,
                        'average_stars': user.average_stars,
                        'yelping_since_days': user.yelping_since_days,
                        'absa_features': str(user.absa_features) if user.absa_features else 'null',
                        'text_embedding': str(user.text_embedding) if user.text_embedding else 'null'
                    })
                    migrated_count += 1
                railway_session.commit()
            except Exception as e:
                logger.warning(f"배치 {i} 실패, 재시도... ({e})")
                railway_session.rollback()
                # 개별 재시도
                for user in batch:
                    try:
                        railway_session.execute(text("""
                            INSERT INTO users (yelp_user_id, username, email, hashed_password, created_at, 
                                             review_count, useful, compliment, fans, average_stars, 
                                             yelping_since_days, absa_features, text_embedding)
                            VALUES (:yelp_user_id, :username, :email, :hashed_password, :created_at,
                                   :review_count, :useful, :compliment, :fans, :average_stars,
                                   :yelping_since_days, :absa_features::jsonb, :text_embedding::jsonb)
                            ON CONFLICT (username) DO NOTHING
                        """), {
                            'yelp_user_id': user.yelp_user_id,
                            'username': user.username,
                            'email': user.email,
                            'hashed_password': user.hashed_password,
                            'created_at': user.created_at,
                            'review_count': user.review_count,
                            'useful': user.useful,
                            'compliment': user.compliment,
                            'fans': user.fans,
                            'average_stars': user.average_stars,
                            'yelping_since_days': user.yelping_since_days,
                            'absa_features': str(user.absa_features) if user.absa_features else 'null',
                            'text_embedding': str(user.text_embedding) if user.text_embedding else 'null'
                        })
                        railway_session.commit()
                        migrated_count += 1
                    except:
                        pass
        
        logger.info(f"✅ {migrated_count:,}명의 사용자 마이그레이션 완료")
        
        # 7. Businesses 마이그레이션
        logger.info("\n7️⃣ Businesses 테이블 마이그레이션")
        businesses = local_session.query(Business).all()
        
        biz_count = 0
        for i in tqdm(range(0, len(businesses), batch_size), desc="Businesses"):
            batch = businesses[i:i+batch_size]
            try:
                for biz in batch:
                    railway_session.execute(text("""
                        INSERT INTO businesses (business_id, name, categories, stars, review_count,
                                              address, city, state, latitude, longitude, absa_features)
                        VALUES (:business_id, :name, :categories, :stars, :review_count,
                               :address, :city, :state, :latitude, :longitude, :absa_features::jsonb)
                        ON CONFLICT (business_id) DO NOTHING
                    """), {
                        'business_id': biz.business_id,
                        'name': biz.name,
                        'categories': biz.categories,
                        'stars': biz.stars,
                        'review_count': biz.review_count,
                        'address': biz.address,
                        'city': biz.city,
                        'state': biz.state,
                        'latitude': biz.latitude,
                        'longitude': biz.longitude,
                        'absa_features': str(biz.absa_features) if biz.absa_features else 'null'
                    })
                    biz_count += 1
                railway_session.commit()
            except Exception as e:
                logger.warning(f"배치 {i} 실패 ({e[:50]}...)")
                railway_session.rollback()
        
        logger.info(f"✅ {biz_count:,}개의 비즈니스 마이그레이션 완료")
        
        # 8. Reviews 마이그레이션 (ID 매핑 사용)
        logger.info("\n8️⃣ Reviews 테이블 마이그레이션")
        logger.info("   (Users와 Businesses ID 매핑 생성 중...)")
        
        # ID 매핑 직접 SQL로 생성
        user_mapping = railway_session.execute(text("""
            SELECT username, id FROM users
        """)).fetchall()
        username_to_id = {row[0]: row[1] for row in user_mapping}
        
        biz_mapping = railway_session.execute(text("""
            SELECT business_id, id FROM businesses
        """)).fetchall()
        bizid_to_id = {row[0]: row[1] for row in biz_mapping}
        
        reviews = local_session.query(Review).all()
        review_count = 0
        
        for i in tqdm(range(0, len(reviews), batch_size), desc="Reviews"):
            batch = reviews[i:i+batch_size]
            try:
                for review in batch:
                    # 로컬 User와 Business를 Railway ID로 변환
                    local_user = local_session.query(User).get(review.user_id)
                    railway_user_id = username_to_id.get(local_user.username) if local_user else None
                    
                    railway_biz_id = None
                    if review.business_id:
                        local_biz = local_session.query(Business).get(review.business_id)
                        railway_biz_id = bizid_to_id.get(local_biz.business_id) if local_biz else None
                    
                    if railway_user_id:
                        railway_session.execute(text("""
                            INSERT INTO reviews (user_id, business_id, stars, text, date, created_at,
                                               absa_features, useful, is_taste_test, taste_test_type, taste_test_weight)
                            VALUES (:user_id, :business_id, :stars, :text, :date, :created_at,
                                   :absa_features::jsonb, :useful, :is_taste_test, :taste_test_type, :taste_test_weight)
                        """), {
                            'user_id': railway_user_id,
                            'business_id': railway_biz_id,
                            'stars': review.stars,
                            'text': review.text,
                            'date': review.date,
                            'created_at': review.created_at,
                            'absa_features': str(review.absa_features) if review.absa_features else 'null',
                            'useful': review.useful,
                            'is_taste_test': review.is_taste_test,
                            'taste_test_type': review.taste_test_type,
                            'taste_test_weight': review.taste_test_weight
                        })
                        review_count += 1
                railway_session.commit()
            except Exception as e:
                logger.warning(f"배치 {i} 실패 ({str(e)[:50]}...)")
                railway_session.rollback()
        
        logger.info(f"✅ {review_count:,}개의 리뷰 마이그레이션 완료")
        
        # 9. 최종 확인
        logger.info("\n9️⃣ 마이그레이션 결과 확인")
        final_counts = get_table_counts(railway_session, "Railway DB (최종)")
        
        # 10. 성공 메시지
        logger.info("\n" + "="*60)
        logger.info("🎉 마이그레이션 완료!")
        logger.info("="*60)
        logger.info(f"✅ Users: {final_counts['users']:,}")
        logger.info(f"✅ Businesses: {final_counts['businesses']:,}")
        logger.info(f"✅ Reviews: {final_counts['reviews']:,}")
        logger.info("\n🌐 Railway DB URL: https://backendweb-production-7b6c.up.railway.app")
        logger.info("📱 Frontend URL: https://soulplate.vercel.app")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()
        railway_session.rollback()
        return False
        
    finally:
        local_session.close()
        railway_session.close()


if __name__ == "__main__":
    success = migrate_data()
    sys.exit(0 if success else 1)

