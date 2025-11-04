"""
PostgreSQL 연결 테스트 스크립트
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv("backend_web/.env")

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    DATABASE_URL = "postgresql://two_tower_user:twotower2024@localhost:5432/two_tower_db"

print("=" * 60)
print("PostgreSQL 연결 테스트")
print("=" * 60)
print(f"\n연결 문자열: {DATABASE_URL.replace('twotower2024', '****')}\n")

try:
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    print("[OK] PostgreSQL 연결 성공!\n")
    
    # 테스트 쿼리
    result = connection.execute(text("SELECT version();"))
    version = result.fetchone()
    print(f"PostgreSQL 버전:")
    print(f"  {version[0]}\n")
    
    # 테이블 목록 확인
    result = connection.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """))
    tables = result.fetchall()
    
    if tables:
        print("생성된 테이블 목록:")
        for table in tables:
            print(f"  * {table[0]}")
    else:
        print("아직 테이블이 생성되지 않았습니다.")
        print("다음 명령으로 테이블을 생성하세요:")
        print("  python scripts/init_db.py")
    
    connection.close()
    print("\n" + "=" * 60)
    print("연결 테스트 완료!")
    print("=" * 60)
    
except Exception as e:
    print(f"[ERROR] 연결 실패: {e}\n")
    print("해결 방법:")
    print("1. PostgreSQL이 실행 중인지 확인")
    print("2. 데이터베이스가 생성되었는지 확인")
    print("   - psql -U postgres (비밀번호: 0205)")
    print("   - 또는 scripts/create_postgres_db.sql 실행")
    print("3. .env 파일의 DATABASE_URL 확인")
    print("=" * 60)
