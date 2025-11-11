"""
ABSA 컬럼 복구 스크립트
- user_absa_features, business_absa_features 테이블 삭제
- users, businesses 테이블에 absa_features 컬럼 추가
"""

import sys
sys.path.append('.')

from backend_web.database import engine
from sqlalchemy import text

def main():
    print("=" * 80)
    print("ABSA 구조 복구")
    print("=" * 80)
    
    print("\n경고: 이 작업은 되돌릴 수 없습니다!")
    print("  - user_absa_features, business_absa_features 테이블 삭제")
    print("  - users, businesses 테이블에 absa_features 컬럼 추가\n")
    
    with engine.connect() as conn:
        try:
            # 1. ABSA 테이블 삭제
            print("[1/3] ABSA 테이블 삭제 중...")
            conn.execute(text("DROP TABLE IF EXISTS user_absa_features CASCADE"))
            conn.commit()
            print("  [OK] user_absa_features 삭제 완료")
            
            conn.execute(text("DROP TABLE IF EXISTS business_absa_features CASCADE"))
            conn.commit()
            print("  [OK] business_absa_features 삭제 완료")
            
            # 2. users 테이블에 absa_features 컬럼 추가
            print("\n[2/3] users.absa_features 컬럼 추가 중...")
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS absa_features JSONB"))
                conn.commit()
                print("  [OK] 완료")
            except Exception as e:
                if "already exists" in str(e):
                    print("  [INFO] 이미 존재합니다")
                    conn.rollback()
                else:
                    raise
            
            # 3. businesses 테이블에 absa_features 컬럼 추가
            print("\n[3/3] businesses.absa_features 컬럼 추가 중...")
            try:
                conn.execute(text("ALTER TABLE businesses ADD COLUMN IF NOT EXISTS absa_features JSONB"))
                conn.commit()
                print("  [OK] 완료")
            except Exception as e:
                if "already exists" in str(e):
                    print("  [INFO] 이미 존재합니다")
                    conn.rollback()
                else:
                    raise
            
            print("\n" + "=" * 80)
            print("[SUCCESS] ABSA 구조 복구 완료!")
            print("=" * 80)
            print("\n다음 단계:")
            print("  python scripts/create_filtered_dataset.py")
            
        except Exception as e:
            print(f"\n[ERROR] 실패: {e}")
            conn.rollback()
            raise


if __name__ == "__main__":
    main()

