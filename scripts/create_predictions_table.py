"""
user_business_predictions 테이블 생성 마이그레이션 스크립트
"""

import sys
import os

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from backend_web.database import SQLALCHEMY_DATABASE_URL
from backend_web.models import Base, UserBusinessPrediction

def create_predictions_table():
    """user_business_predictions 테이블 생성"""
    print("=" * 60)
    print("user_business_predictions 테이블 생성 시작")
    print("=" * 60)
    
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    
    try:
        # UserBusinessPrediction 테이블만 생성
        print("\n[1/1] user_business_predictions 테이블 생성 중...")
        UserBusinessPrediction.__table__.create(bind=engine, checkfirst=True)
        print("✓ user_business_predictions 테이블 생성 완료")
        
        # 테이블 확인
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'user_business_predictions'
                ORDER BY ordinal_position;
            """))
            columns = result.fetchall()
            
            print("\n생성된 컬럼 목록:")
            for col in columns:
                print(f"  - {col[0]}: {col[1]} (nullable={col[2]})")
            
            # 인덱스 확인
            result = conn.execute(text("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'user_business_predictions';
            """))
            indexes = result.fetchall()
            
            print("\n생성된 인덱스 목록:")
            for idx in indexes:
                print(f"  - {idx[0]}")
        
        print("\n" + "=" * 60)
        print("✓ 마이그레이션 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        print("\n이미 테이블이 존재하는 경우 이 메시지가 나올 수 있습니다.")
        print("테이블이 이미 존재한다면 정상입니다.")
    
    finally:
        engine.dispose()

if __name__ == "__main__":
    create_predictions_table()












