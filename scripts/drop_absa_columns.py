"""
User와 Business 테이블에서 absa_features 컬럼 제거
"""

import sys
sys.path.append('.')

from backend_web.database import engine
from sqlalchemy import text

def main():
    print("=" * 80)
    print("ABSA 컬럼 제거")
    print("=" * 80)
    
    print("\n경고: users와 businesses 테이블에서 absa_features 컬럼을 제거합니다.")
    print("이 작업은 되돌릴 수 없습니다!\n")
    
    with engine.connect() as conn:
        try:
            # Users 테이블에서 absa_features 컬럼 제거
            print("[1/2] users.absa_features 컬럼 제거 중...")
            conn.execute(text("ALTER TABLE users DROP COLUMN IF EXISTS absa_features"))
            conn.commit()
            print("  [OK] 완료")
            
            # Businesses 테이블에서 absa_features 컬럼 제거
            print("[2/2] businesses.absa_features 컬럼 제거 중...")
            conn.execute(text("ALTER TABLE businesses DROP COLUMN IF EXISTS absa_features"))
            conn.commit()
            print("  [OK] 완료")
            
            print("\n[SUCCESS] ABSA 컬럼 제거 완료!")
            print("\n이제 마이그레이션을 실행할 수 있습니다:")
            print("  python scripts/fast_copy_migration.py")
            
        except Exception as e:
            print(f"\n[ERROR] 실패: {e}")
            conn.rollback()
            raise


if __name__ == "__main__":
    main()

