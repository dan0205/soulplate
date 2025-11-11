"""
Review 테이블에 useful 컬럼 추가
"""
import sys
sys.path.append('.')

from backend_web.database import engine
from sqlalchemy import text

def add_useful_column():
    """Review 테이블에 useful 컬럼 추가"""
    print("=" * 80)
    print("Review 테이블에 useful 컬럼 추가")
    print("=" * 80)
    
    try:
        with engine.connect() as conn:
            # useful 컬럼이 이미 있는지 확인
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='reviews' AND column_name='useful'
            """))
            
            if result.fetchone():
                print("\n[INFO] useful 컬럼이 이미 존재합니다.")
                return
            
            # useful 컬럼 추가
            print("\n[1/2] useful 컬럼 추가 중...")
            conn.execute(text("""
                ALTER TABLE reviews 
                ADD COLUMN useful INTEGER NOT NULL DEFAULT 0
            """))
            conn.commit()
            print("  [OK] useful 컬럼 추가 완료")
            
            # 기존 리뷰의 useful 값을 0으로 설정 (이미 default 0이지만 명시적으로)
            print("\n[2/2] 기존 리뷰의 useful 값 초기화 중...")
            conn.execute(text("""
                UPDATE reviews 
                SET useful = 0 
                WHERE useful IS NULL
            """))
            conn.commit()
            print("  [OK] 초기화 완료")
            
        print("\n" + "=" * 80)
        print("[SUCCESS] useful 컬럼 추가 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    add_useful_column()

