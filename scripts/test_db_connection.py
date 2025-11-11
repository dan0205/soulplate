"""
DB 연결 테스트
"""

import sys
sys.path.append('.')

print("1. 모듈 import 중...")
try:
    from backend_web.database import SessionLocal, engine
    from backend_web import models
    print("   [OK] 모듈 import 성공")
except Exception as e:
    print(f"   [ERROR] 모듈 import 실패: {e}")
    sys.exit(1)

print("\n2. DB 엔진 확인 중...")
print(f"   엔진: {engine.url}")

print("\n3. DB 연결 테스트 중...")
try:
    with engine.connect() as conn:
        print("   [OK] DB 연결 성공")
except Exception as e:
    print(f"   [ERROR] DB 연결 실패: {e}")
    print("\n원인:")
    print("  - PostgreSQL이 실행 중인지 확인")
    print("  - DATABASE_URL 환경변수가 올바른지 확인")
    print("  - 데이터베이스가 생성되어 있는지 확인")
    sys.exit(1)

print("\n4. 세션 생성 테스트 중...")
try:
    session = SessionLocal()
    print("   [OK] 세션 생성 성공")
    session.close()
except Exception as e:
    print(f"   [ERROR] 세션 생성 실패: {e}")
    sys.exit(1)

print("\n5. 테이블 존재 확인 중...")
try:
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"   발견된 테이블: {len(tables)}개")
    for table in tables:
        print(f"     - {table}")
except Exception as e:
    print(f"   [ERROR] 테이블 확인 실패: {e}")
    sys.exit(1)

print("\n6. 데이터 개수 확인 중...")
try:
    session = SessionLocal()
    user_count = session.query(models.User).count()
    business_count = session.query(models.Business).count()
    review_count = session.query(models.Review).count()
    
    print(f"   Users: {user_count:,}개")
    print(f"   Businesses: {business_count:,}개")
    print(f"   Reviews: {review_count:,}개")
    
    session.close()
except Exception as e:
    print(f"   [ERROR] 데이터 확인 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] 모든 DB 연결 테스트 통과!")
print("=" * 60)

