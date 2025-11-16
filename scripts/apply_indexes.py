"""
Railway PostgreSQL에 인덱스를 적용하는 스크립트

사용법:
    export RAILWAY_DATABASE_URL="postgresql://..."
    python scripts/apply_indexes.py
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_indexes(database_url):
    """SQL 파일을 읽어 인덱스를 적용"""
    
    # SQL 파일 경로
    sql_file = Path(__file__).parent / "add_indexes.sql"
    
    if not sql_file.exists():
        logger.error(f"SQL 파일을 찾을 수 없습니다: {sql_file}")
        return False
    
    # SQL 파일 읽기
    logger.info(f"📖 SQL 파일 읽는 중: {sql_file}")
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # 데이터베이스 연결
    logger.info(f"🔌 데이터베이스 연결 중...")
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as conn:
            # SQL을 세미콜론으로 분리하여 각각 실행
            statements = [s.strip() for s in sql_content.split(';') if s.strip() and not s.strip().startswith('--')]
            
            total = len(statements)
            success = 0
            failed = 0
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 인덱스 적용 시작 (총 {total}개 명령)")
            logger.info(f"{'='*60}\n")
            
            for idx, statement in enumerate(statements, 1):
                # 주석 제거 및 공백 정리
                clean_statement = '\n'.join([
                    line for line in statement.split('\n') 
                    if not line.strip().startswith('--')
                ])
                
                if not clean_statement.strip():
                    continue
                
                try:
                    # 인덱스 이름 추출 (로깅용)
                    if 'CREATE INDEX' in clean_statement.upper():
                        index_name = clean_statement.split('IF NOT EXISTS')[-1].split('ON')[0].strip()
                        logger.info(f"[{idx}/{total}] 생성 중: {index_name}")
                    elif 'CREATE EXTENSION' in clean_statement.upper():
                        logger.info(f"[{idx}/{total}] 확장 활성화 중...")
                    else:
                        logger.info(f"[{idx}/{total}] 실행 중...")
                    
                    # SQL 실행
                    conn.execute(text(clean_statement))
                    conn.commit()
                    success += 1
                    logger.info(f"  ✅ 성공")
                    
                except Exception as e:
                    failed += 1
                    error_msg = str(e)
                    
                    # 이미 존재하는 인덱스는 에러가 아님
                    if 'already exists' in error_msg.lower():
                        logger.info(f"  ⏭️  이미 존재함")
                        success += 1
                        failed -= 1
                    else:
                        logger.error(f"  ❌ 실패: {error_msg}")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 결과: 성공 {success}개, 실패 {failed}개")
            logger.info(f"{'='*60}\n")
            
            # 생성된 인덱스 목록 확인
            logger.info("📋 생성된 인덱스 목록:\n")
            result = conn.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                    AND indexname LIKE 'idx_%'
                ORDER BY tablename, indexname;
            """))
            
            indexes = result.fetchall()
            current_table = None
            
            for schema, table, idx_name, idx_def in indexes:
                if current_table != table:
                    current_table = table
                    logger.info(f"\n📁 {table}:")
                logger.info(f"  - {idx_name}")
            
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
        logger.error("  python scripts/apply_indexes.py")
        sys.exit(1)
    
    # 데이터베이스 호스트 정보 표시 (비밀번호 제외)
    host_info = database_url.split('@')[1].split('/')[0] if '@' in database_url else 'unknown'
    logger.info(f"🗄️  대상 데이터베이스: {host_info}")
    
    # 인덱스 적용
    success = apply_indexes(database_url)
    
    if success:
        logger.info("\n🎉 모든 인덱스가 성공적으로 적용되었습니다!")
        logger.info("\n다음 단계:")
        logger.info("  1. 프론트엔드에서 API 호출 테스트")
        logger.info("  2. Railway 로그에서 성능 개선 확인")
        logger.info("  3. 슬로우 쿼리 감소 확인")
    else:
        logger.error("\n⚠️  일부 인덱스 적용에 실패했습니다.")
        logger.error("위 에러 메시지를 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()

