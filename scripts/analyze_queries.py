"""
Railway PostgreSQL 쿼리 실행 계획 분석 스크립트

사용법:
    export RAILWAY_DATABASE_URL="postgresql://..."
    python scripts/analyze_queries.py
"""

import os
import sys
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 분석할 쿼리 목록 (실제 느린 쿼리들)
QUERIES_TO_ANALYZE = [
    {
        "name": "Business by business_id",
        "sql": """
            SELECT * FROM businesses 
            WHERE business_id = 'KR_우만동족발집_아주대점_1763279166431'
            LIMIT 1
        """,
        "expected_index": "idx_businesses_business_id"
    },
    {
        "name": "Reviews by business_id",
        "sql": """
            SELECT * FROM reviews 
            WHERE business_id = 677 
            ORDER BY created_at DESC 
            LIMIT 10
        """,
        "expected_index": "idx_reviews_business_created"
    },
    {
        "name": "User review count",
        "sql": """
            SELECT user_id, COUNT(id) as review_count
            FROM reviews 
            WHERE user_id IN (24290, 24256, 23653)
            GROUP BY user_id
        """,
        "expected_index": "idx_reviews_user_id"
    },
    {
        "name": "AI Prediction cache",
        "sql": """
            SELECT * FROM user_business_predictions 
            WHERE user_id = 24288 AND business_id IN (495, 536, 275)
        """,
        "expected_index": "idx_predictions_user_business"
    },
    {
        "name": "Businesses by location",
        "sql": """
            SELECT * FROM businesses 
            WHERE latitude BETWEEN 37.28 AND 37.48 
              AND longitude BETWEEN 126.86 AND 127.08
              AND latitude IS NOT NULL 
              AND longitude IS NOT NULL
            LIMIT 100
        """,
        "expected_index": "idx_businesses_lat_lng"
    },
    {
        "name": "User by username",
        "sql": """
            SELECT * FROM users 
            WHERE username = 'abc' 
            LIMIT 1
        """,
        "expected_index": "idx_users_username"
    },
    {
        "name": "User by id",
        "sql": """
            SELECT * FROM users 
            WHERE id = 24290
        """,
        "expected_index": "PRIMARY KEY"
    },
]


def analyze_query(conn, query_info):
    """단일 쿼리의 실행 계획 분석"""
    name = query_info["name"]
    sql = query_info["sql"]
    expected_index = query_info.get("expected_index", "N/A")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 쿼리: {name}")
    logger.info(f"{'='*70}")
    logger.info(f"예상 인덱스: {expected_index}")
    logger.info(f"\n쿼리:")
    logger.info(sql.strip())
    logger.info(f"\n{'-'*70}")
    
    try:
        # EXPLAIN ANALYZE 실행
        result = conn.execute(text(f"EXPLAIN ANALYZE {sql}"))
        plan = result.fetchall()
        
        logger.info("실행 계획:\n")
        
        index_used = False
        seq_scan_used = False
        execution_time = None
        
        for row in plan:
            line = row[0]
            logger.info(f"  {line}")
            
            # 인덱스 사용 여부 확인
            if "Index Scan" in line or "Index Only Scan" in line:
                index_used = True
            if "Seq Scan" in line:
                seq_scan_used = True
            
            # 실행 시간 추출
            if "Execution Time:" in line:
                execution_time = line.split("Execution Time:")[1].strip()
        
        logger.info(f"\n{'-'*70}")
        
        # 결과 분석
        if index_used:
            logger.info(f"✅ 인덱스 사용: YES")
        elif seq_scan_used:
            logger.warning(f"❌ 전체 테이블 스캔 (Seq Scan) 발생!")
            logger.warning(f"   인덱스가 사용되지 않았습니다.")
        
        if execution_time:
            logger.info(f"⏱️  실행 시간: {execution_time}")
        
        return {
            "name": name,
            "index_used": index_used,
            "seq_scan": seq_scan_used,
            "execution_time": execution_time,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ 쿼리 분석 실패: {e}")
        return {
            "name": name,
            "success": False,
            "error": str(e)
        }


def main():
    # 환경변수에서 DATABASE_URL 가져오기
    database_url = os.getenv("RAILWAY_DATABASE_URL")
    
    if not database_url:
        logger.error("❌ RAILWAY_DATABASE_URL 환경변수가 설정되지 않았습니다.")
        sys.exit(1)
    
    # 데이터베이스 호스트 정보 표시
    host_info = database_url.split('@')[1].split('/')[0] if '@' in database_url else 'unknown'
    logger.info(f"🗄️  대상 데이터베이스: {host_info}")
    
    # 데이터베이스 연결
    logger.info(f"🔌 데이터베이스 연결 중...\n")
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as conn:
            results = []
            
            logger.info("=" * 70)
            logger.info("🚀 쿼리 실행 계획 분석 시작")
            logger.info(f"총 {len(QUERIES_TO_ANALYZE)}개 쿼리 분석")
            logger.info("=" * 70)
            
            for query_info in QUERIES_TO_ANALYZE:
                result = analyze_query(conn, query_info)
                results.append(result)
            
            # 요약
            logger.info(f"\n\n{'='*70}")
            logger.info("📋 분석 요약")
            logger.info(f"{'='*70}\n")
            
            success_count = sum(1 for r in results if r.get("success"))
            index_used_count = sum(1 for r in results if r.get("index_used"))
            seq_scan_count = sum(1 for r in results if r.get("seq_scan"))
            
            logger.info(f"총 쿼리: {len(results)}개")
            logger.info(f"성공: {success_count}개")
            logger.info(f"인덱스 사용: {index_used_count}개")
            logger.info(f"전체 스캔 (Seq Scan): {seq_scan_count}개")
            
            if seq_scan_count > 0:
                logger.warning(f"\n⚠️  {seq_scan_count}개 쿼리에서 전체 테이블 스캔 발생!")
                logger.warning("인덱스가 제대로 사용되지 않고 있습니다.")
                logger.warning("\n가능한 원인:")
                logger.warning("  1. 데이터가 너무 적어 PostgreSQL이 인덱스보다 Seq Scan을 선택")
                logger.warning("  2. 통계 정보가 최신화되지 않음 (ANALYZE 필요)")
                logger.warning("  3. 쿼리 조건이 인덱스와 맞지 않음")
            else:
                logger.info("\n✅ 모든 쿼리가 인덱스를 사용하고 있습니다!")
            
            # 개별 쿼리 결과
            logger.info(f"\n{'='*70}")
            logger.info("상세 결과")
            logger.info(f"{'='*70}\n")
            
            for r in results:
                if r.get("success"):
                    status = "✅ Index" if r.get("index_used") else "❌ Seq Scan"
                    time_info = f"({r.get('execution_time', 'N/A')})" if r.get('execution_time') else ""
                    logger.info(f"{status:15} {r['name']:30} {time_info}")
                else:
                    logger.error(f"❌ FAILED      {r['name']:30} Error: {r.get('error', 'Unknown')}")
            
    except Exception as e:
        logger.error(f"❌ 데이터베이스 연결 실패: {e}")
        sys.exit(1)
    finally:
        engine.dispose()
    
    logger.info("\n✅ 분석 완료!")


if __name__ == "__main__":
    main()

