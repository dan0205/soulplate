"""
DB 스키마 재생성
- 기존 테이블 유지하면서 새 테이블 추가
"""

import sys
sys.path.append('.')

from backend_web.database import engine
from backend_web import models

def main():
    print("=" * 80)
    print("DB 스키마 재생성")
    print("=" * 80)
    
    print("\n새로운 테이블 생성 중...")
    print("- user_absa_features")
    print("- business_absa_features")
    
    # 모든 테이블 생성 (이미 존재하는 테이블은 건너뜀)
    models.Base.metadata.create_all(bind=engine)
    
    print("\n[OK] 스키마 생성 완료!")
    print("\n생성된 테이블:")
    
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    for table in tables:
        print(f"  - {table}")
    
    print("\n이제 마이그레이션을 실행할 수 있습니다:")
    print("  python scripts/fast_copy_migration.py")


if __name__ == "__main__":
    main()

