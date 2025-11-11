"""
PostgreSQL 데이터베이스 설정 및 초기화
- 데이터베이스 생성
- 테이블 생성
- 기존 SQLite DB 백업
"""

import sys
sys.path.append('.')

import os
import shutil
from datetime import datetime
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from backend_web.database import SQLALCHEMY_DATABASE_URL, engine
from backend_web import models

def backup_sqlite_db():
    """기존 SQLite DB 백업"""
    print("=" * 80)
    print("기존 SQLite DB 백업")
    print("=" * 80)
    
    sqlite_path = "app.db"
    if os.path.exists(sqlite_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"app.db.backup_{timestamp}"
        shutil.copy2(sqlite_path, backup_path)
        print(f"[OK] 백업 완료: {backup_path}")
        
        # 원본 삭제
        os.remove(sqlite_path)
        print(f"[OK] 기존 DB 삭제: {sqlite_path}")
    else:
        print("[INFO] app.db 파일이 없습니다. 백업 건너뜀")

def create_postgresql_db():
    """PostgreSQL 데이터베이스 생성"""
    print("\n" + "=" * 80)
    print("PostgreSQL 데이터베이스 생성")
    print("=" * 80)
    
    # 연결 문자열 파싱
    # postgresql://user:password@host:port/dbname
    import re
    match = re.match(
        r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)',
        SQLALCHEMY_DATABASE_URL
    )
    
    if not match:
        print("[ERROR] DATABASE_URL 형식이 잘못되었습니다")
        return False
    
    user, password, host, port, dbname = match.groups()
    
    try:
        # postgres 기본 DB에 연결 (데이터베이스 생성용)
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # 기존 DB 삭제 (있으면)
        print(f"\n[1/2] 기존 '{dbname}' 데이터베이스 확인 중...")
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
        exists = cursor.fetchone()
        
        if exists:
            print(f"  기존 DB 발견. 삭제 중...")
            # 연결 강제 종료
            cursor.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{dbname}'
                AND pid <> pg_backend_pid()
            """)
            cursor.execute(f"DROP DATABASE {dbname}")
            print(f"  [OK] 기존 DB 삭제 완료")
        
        # 새 DB 생성
        print(f"\n[2/2] 새 '{dbname}' 데이터베이스 생성 중...")
        cursor.execute(f"CREATE DATABASE {dbname} ENCODING 'UTF8'")
        print(f"  [OK] 데이터베이스 생성 완료")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL 연결 실패: {e}")
        print("\nPostgreSQL이 설치되어 있고 실행 중인지 확인하세요:")
        print("  Windows: PostgreSQL 서비스 실행 확인")
        print("  Mac: brew services start postgresql")
        print("  Linux: sudo systemctl start postgresql")
        return False

def create_tables():
    """SQLAlchemy를 사용하여 테이블 생성"""
    print("\n" + "=" * 80)
    print("데이터베이스 테이블 생성")
    print("=" * 80)
    
    try:
        # models.py에 정의된 모든 테이블 생성
        models.Base.metadata.create_all(bind=engine)
        
        print("[OK] 테이블 생성 완료:")
        print("  - users (Yelp 사용자 + 신규 회원)")
        print("  - businesses (Yelp 가게 + ABSA)")
        print("  - reviews (Yelp 리뷰 + ABSA)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 테이블 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_setup():
    """설정 확인"""
    print("\n" + "=" * 80)
    print("설정 확인")
    print("=" * 80)
    
    try:
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print(f"[OK] 생성된 테이블 ({len(tables)}개):")
        for table in tables:
            columns = inspector.get_columns(table)
            print(f"  - {table}: {len(columns)}개 컬럼")
        
        # 주요 컬럼 확인
        print("\n주요 JSONB 컬럼 확인:")
        for table in ['users', 'businesses', 'reviews']:
            if table in tables:
                columns = inspector.get_columns(table)
                jsonb_cols = [c['name'] for c in columns if 'absa_features' in c['name']]
                if jsonb_cols:
                    print(f"  ✓ {table}.absa_features (JSONB)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 확인 실패: {e}")
        return False

def main():
    print("\n" + "=" * 80)
    print("PostgreSQL 설정 시작")
    print("=" * 80)
    print(f"\n데이터베이스 URL: {SQLALCHEMY_DATABASE_URL}")
    print("\n[주의] 기존 데이터베이스가 삭제됩니다!")
    
    # 1. SQLite 백업
    backup_sqlite_db()
    
    # 2. PostgreSQL DB 생성
    if not create_postgresql_db():
        print("\n[FAILED] PostgreSQL 데이터베이스 생성 실패")
        return
    
    # 3. 테이블 생성
    if not create_tables():
        print("\n[FAILED] 테이블 생성 실패")
        return
    
    # 4. 확인
    if verify_setup():
        print("\n" + "=" * 80)
        print("[SUCCESS] PostgreSQL 설정 완료!")
        print("\n다음 단계:")
        print("  python scripts/migrate_data_to_postgresql.py")
        print("=" * 80)
    else:
        print("\n[FAILED] 설정 확인 실패")

if __name__ == "__main__":
    main()

