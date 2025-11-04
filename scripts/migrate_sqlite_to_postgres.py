"""
SQLite에서 PostgreSQL로 데이터 마이그레이션
"""

import sqlite3
import sys
import os

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("psycopg2가 설치되지 않았습니다.")
    print("설치: pip install psycopg2-binary")
    sys.exit(1)

# 설정
SQLITE_DB = "app.db"
POSTGRES_URL = "postgresql://two_tower_user:your_password@localhost:5432/two_tower_db"

def migrate():
    """마이그레이션 실행"""
    
    print("=" * 60)
    print("SQLite → PostgreSQL 데이터 마이그레이션")
    print("=" * 60)
    
    # SQLite 연결
    print("\n[1/5] SQLite 연결 중...")
    if not os.path.exists(SQLITE_DB):
        print(f"✗ {SQLITE_DB} 파일을 찾을 수 없습니다.")
        return False
    
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_cursor = sqlite_conn.cursor()
    print("✓ SQLite 연결 성공")
    
    # PostgreSQL 연결
    print("\n[2/5] PostgreSQL 연결 중...")
    try:
        pg_conn = psycopg2.connect(POSTGRES_URL)
        pg_cursor = pg_conn.cursor()
        print("✓ PostgreSQL 연결 성공")
    except Exception as e:
        print(f"✗ PostgreSQL 연결 실패: {e}")
        sqlite_conn.close()
        return False
    
    try:
        # Users 마이그레이션
        print("\n[3/5] Users 테이블 마이그레이션...")
        sqlite_cursor.execute("SELECT * FROM users")
        users = sqlite_cursor.fetchall()
        
        if users:
            # 기존 데이터 삭제 (옵션)
            pg_cursor.execute("TRUNCATE TABLE users CASCADE;")
            
            # 데이터 삽입
            for user in users:
                pg_cursor.execute(
                    """INSERT INTO users 
                       (id, username, email, hashed_password, age, gender, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    user
                )
            print(f"  ✓ {len(users)}명의 사용자 마이그레이션 완료")
        else:
            print("  - Users 테이블이 비어있습니다")
        
        # Businesses 마이그레이션
        print("\n[4/5] Businesses 테이블 마이그레이션...")
        sqlite_cursor.execute("SELECT * FROM businesses")
        businesses = sqlite_cursor.fetchall()
        
        if businesses:
            pg_cursor.execute("TRUNCATE TABLE businesses CASCADE;")
            
            for biz in businesses:
                pg_cursor.execute(
                    """INSERT INTO businesses 
                       (id, business_id, name, categories, stars, review_count,
                        address, city, state, latitude, longitude, is_open)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    biz
                )
            print(f"  ✓ {len(businesses)}개의 비즈니스 마이그레이션 완료")
        else:
            print("  - Businesses 테이블이 비어있습니다")
        
        # Reviews 마이그레이션
        print("\n[5/5] Reviews 테이블 마이그레이션...")
        sqlite_cursor.execute("SELECT * FROM reviews")
        reviews = sqlite_cursor.fetchall()
        
        if reviews:
            pg_cursor.execute("TRUNCATE TABLE reviews CASCADE;")
            
            for review in reviews:
                pg_cursor.execute(
                    """INSERT INTO reviews 
                       (id, user_id, business_id, stars, text, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    review
                )
            print(f"  ✓ {len(reviews)}개의 리뷰 마이그레이션 완료")
        else:
            print("  - Reviews 테이블이 비어있습니다")
        
        # Sequence 재설정 (Auto-increment)
        print("\nSequence 재설정 중...")
        pg_cursor.execute("SELECT setval('users_id_seq', (SELECT COALESCE(MAX(id), 1) FROM users));")
        pg_cursor.execute("SELECT setval('businesses_id_seq', (SELECT COALESCE(MAX(id), 1) FROM businesses));")
        pg_cursor.execute("SELECT setval('reviews_id_seq', (SELECT COALESCE(MAX(id), 1) FROM reviews));")
        print("✓ Sequence 재설정 완료")
        
        # 커밋
        pg_conn.commit()
        
        print("\n" + "=" * 60)
        print("마이그레이션 완료!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 마이그레이션 중 오류 발생: {e}")
        pg_conn.rollback()
        return False
        
    finally:
        sqlite_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    print("\n⚠️  주의: 이 스크립트는 PostgreSQL의 기존 데이터를 삭제합니다!")
    print("계속하시겠습니까? (y/N): ", end="")
    
    response = input().strip().lower()
    if response == 'y':
        success = migrate()
        sys.exit(0 if success else 1)
    else:
        print("마이그레이션 취소됨")
        sys.exit(0)

