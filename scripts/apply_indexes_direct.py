"""
Railway PostgreSQL에 인덱스를 직접 적용하는 스크립트 (SQL 파일 파싱 문제 해결)
"""

import os
import sys
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 인덱스 생성 SQL 목록
INDEXES = [
    # pg_trgm 확장
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    
    # reviews 테이블
    "CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_reviews_business_id ON reviews(business_id)",
    "CREATE INDEX IF NOT EXISTS idx_reviews_created_at ON reviews(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_reviews_useful ON reviews(useful DESC)",
    "CREATE INDEX IF NOT EXISTS idx_reviews_business_created ON reviews(business_id, created_at DESC)",
    
    # businesses 테이블
    "CREATE INDEX IF NOT EXISTS idx_businesses_lat_lng ON businesses(latitude, longitude)",
    "CREATE INDEX IF NOT EXISTS idx_businesses_review_count ON businesses(review_count DESC)",
    "CREATE INDEX IF NOT EXISTS idx_businesses_name_trgm ON businesses USING gin(name gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS idx_businesses_categories_trgm ON businesses USING gin(categories gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS idx_businesses_city ON businesses(city)",
    "CREATE INDEX IF NOT EXISTS idx_businesses_business_id ON businesses(business_id)",
    
    # user_business_predictions 테이블
    "CREATE INDEX IF NOT EXISTS idx_predictions_user_business ON user_business_predictions(user_id, business_id)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_deepfm ON user_business_predictions(user_id, deepfm_score DESC)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_multitower ON user_business_predictions(user_id, multitower_score DESC)",
    
    # users 테이블
    "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
    "CREATE INDEX IF NOT EXISTS idx_users_yelp_user_id ON users(yelp_user_id)",
]


def apply_indexes(database_url):
    """인덱스를 직접 적용"""
    
    logger.info(f"🔌 데이터베이스 연결 중...")
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as conn:
            total = len(INDEXES)
            success = 0
            failed = 0
            skipped = 0
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 인덱스 적용 시작 (총 {total}개 명령)")
            logger.info(f"{'='*60}\n")
            
            for idx, sql in enumerate(INDEXES, 1):
                try:
                    # 인덱스 이름 추출 (로깅용)
                    if 'CREATE INDEX' in sql.upper():
                        index_name = sql.split('IF NOT EXISTS')[-1].split('ON')[0].strip()
                        logger.info(f"[{idx}/{total}] 생성 중: {index_name}")
                    elif 'CREATE EXTENSION' in sql.upper():
                        logger.info(f"[{idx}/{total}] pg_trgm 확장 활성화 중...")
                    else:
                        logger.info(f"[{idx}/{total}] 실행 중...")
                    
                    # SQL 실행
                    conn.execute(text(sql))
                    conn.commit()
                    success += 1
                    logger.info(f"  ✅ 성공\n")
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # 이미 존재하는 인덱스는 에러가 아님
                    if 'already exists' in error_msg.lower():
                        logger.info(f"  ⏭️  이미 존재함\n")
                        skipped += 1
                        success += 1
                    else:
                        logger.error(f"  ❌ 실패: {error_msg}\n")
                        failed += 1
            
            logger.info(f"{'='*60}")
            logger.info(f"📊 결과: 성공 {success}개, 스킵 {skipped}개, 실패 {failed}개")
            logger.info(f"{'='*60}\n")
            
            # 생성된 인덱스 목록 확인
            logger.info("📋 생성된 인덱스 목록:\n")
            result = conn.execute(text("""
                SELECT 
                    tablename,
                    indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                    AND indexname LIKE 'idx_%'
                ORDER BY tablename, indexname;
            """))
            
            indexes = result.fetchall()
            current_table = None
            
            for table, idx_name in indexes:
                if current_table != table:
                    current_table = table
                    logger.info(f"\n📁 {table}:")
                logger.info(f"  ✓ {idx_name}")
            
            logger.info(f"\n✅ 총 {len(indexes)}개 인덱스 확인됨")
            
            return failed == 0
    
    except Exception as e:
        logger.error(f"❌ 데이터베이스 연결 실패: {e}")
        return False
    finally:
        engine.dispose()


def main():
    # 환경변수에서 DATABASE_URL 가져오기
    database_url = os.getenv("RAILWAY_DATABASE_URL")
    
    if not database_url:
        logger.error("❌ RAILWAY_DATABASE_URL 환경변수가 설정되지 않았습니다.")
        logger.error("\n사용법:")
        logger.error("  export RAILWAY_DATABASE_URL='postgresql://user:pass@host:port/db'")
        logger.error("  python scripts/apply_indexes_direct.py")
        sys.exit(1)
    
    # 데이터베이스 호스트 정보 표시 (비밀번호 제외)
    host_info = database_url.split('@')[1].split('/')[0] if '@' in database_url else 'unknown'
    logger.info(f"🗄️  대상 데이터베이스: {host_info}")
    
    # 인덱스 적용
    success = apply_indexes(database_url)
    
    if success:
        logger.info("\n🎉 모든 인덱스가 성공적으로 적용되었습니다!")
        logger.info("\n다음 단계:")
        logger.info("  1. 코드를 git push하여 배포")
        logger.info("  2. 프론트엔드에서 API 호출 테스트")
        logger.info("  3. Railway 로그에서 성능 개선 확인")
    else:
        logger.error("\n⚠️  일부 인덱스 적용에 실패했습니다.")
        logger.error("위 에러 메시지를 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()

