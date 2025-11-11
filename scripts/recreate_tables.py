"""
PostgreSQL 테이블 재생성
기존 데이터베이스는 유지하고 테이블만 삭제 후 재생성합니다.
"""
import sys
sys.path.append('.')

from backend_web.database import engine
from backend_web import models
from sqlalchemy import inspect, text

def drop_all_tables():
    """모든 테이블 삭제"""
    print("=" * 80)
    print("기존 테이블 삭제")
    print("=" * 80)
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if not tables:
            print("[INFO] 삭제할 테이블이 없습니다.")
            return True
        
        print(f"삭제할 테이블: {', '.join(tables)}")
        
        with engine.connect() as conn:
            # CASCADE로 외래키 제약 무시하고 삭제
            for table in tables:
                print(f"  삭제 중: {table}")
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                conn.commit()
        
        print("[OK] 모든 테이블 삭제 완료")
        return True
        
    except Exception as e:
        print(f"[ERROR] 테이블 삭제 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_tables():
    """SQLAlchemy를 사용하여 테이블 생성"""
    print("\n" + "=" * 80)
    print("새 테이블 생성")
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
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print(f"[OK] 생성된 테이블 ({len(tables)}개):")
        for table in tables:
            columns = inspector.get_columns(table)
            print(f"\n  [{table}] - {len(columns)}개 컬럼:")
            for col in columns:
                col_type = str(col['type'])
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                print(f"    - {col['name']}: {col_type} ({nullable})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 확인 실패: {e}")
        return False

def main():
    print("\n" + "=" * 80)
    print("PostgreSQL 테이블 재생성 시작")
    print("=" * 80)
    print(f"\n데이터베이스: two_tower_db")
    print("\n[주의] 기존 테이블의 모든 데이터가 삭제됩니다!")
    
    # 1. 기존 테이블 삭제
    if not drop_all_tables():
        print("\n[FAILED] 테이블 삭제 실패")
        return
    
    # 2. 새 테이블 생성
    if not create_tables():
        print("\n[FAILED] 테이블 생성 실패")
        return
    
    # 3. 확인
    if verify_setup():
        print("\n" + "=" * 80)
        print("[SUCCESS] PostgreSQL 테이블 재생성 완료!")
        print("\n이제 backend_web 서버를 다시 시작하고 회원가입을 테스트하세요:")
        print("  uvicorn backend_web.main:app --reload --port 8000")
        print("=" * 80)
    else:
        print("\n[FAILED] 설정 확인 실패")

if __name__ == "__main__":
    main()

