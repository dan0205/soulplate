"""
Review 테이블에 취향 테스트 관련 컬럼 추가
User 테이블에 text_embedding 컬럼 추가
"""

from sqlalchemy import text
from backend_web.database import engine, SessionLocal


def add_taste_test_columns():
    """Review 테이블에 취향 테스트 컬럼 추가"""
    print("=" * 80)
    print("데이터베이스 마이그레이션: 취향 테스트 컬럼 추가")
    print("=" * 80)
    
    session = SessionLocal()
    
    try:
        # 1. Review 테이블에 is_taste_test 컬럼 추가
        print("\n[1/5] Review.is_taste_test 컬럼 추가 중...")
        try:
            session.execute(text(
                "ALTER TABLE reviews ADD COLUMN IF NOT EXISTS is_taste_test BOOLEAN DEFAULT FALSE NOT NULL"
            ))
            session.commit()
            print("  [OK] is_taste_test 컬럼 추가 완료")
        except Exception as e:
            print(f"  [INFO] is_taste_test 컬럼이 이미 존재하거나 오류: {e}")
            session.rollback()
        
        # 2. Review 테이블에 taste_test_type 컬럼 추가
        print("\n[2/5] Review.taste_test_type 컬럼 추가 중...")
        try:
            session.execute(text(
                "ALTER TABLE reviews ADD COLUMN IF NOT EXISTS taste_test_type VARCHAR"
            ))
            session.commit()
            print("  [OK] taste_test_type 컬럼 추가 완료")
        except Exception as e:
            print(f"  [INFO] taste_test_type 컬럼이 이미 존재하거나 오류: {e}")
            session.rollback()
        
        # 3. Review 테이블에 taste_test_weight 컬럼 추가
        print("\n[3/5] Review.taste_test_weight 컬럼 추가 중...")
        try:
            session.execute(text(
                "ALTER TABLE reviews ADD COLUMN IF NOT EXISTS taste_test_weight FLOAT DEFAULT 1.0 NOT NULL"
            ))
            session.commit()
            print("  [OK] taste_test_weight 컬럼 추가 완료")
        except Exception as e:
            print(f"  [INFO] taste_test_weight 컬럼이 이미 존재하거나 오류: {e}")
            session.rollback()
        
        # 4. Review 테이블의 business_id를 nullable로 변경
        print("\n[4/5] Review.business_id nullable 설정 중...")
        try:
            session.execute(text(
                "ALTER TABLE reviews ALTER COLUMN business_id DROP NOT NULL"
            ))
            session.execute(text(
                "ALTER TABLE reviews ALTER COLUMN stars DROP NOT NULL"
            ))
            session.commit()
            print("  [OK] business_id, stars nullable 설정 완료")
        except Exception as e:
            print(f"  [INFO] 이미 nullable이거나 오류: {e}")
            session.rollback()
        
        # 5. User 테이블에 text_embedding 컬럼 추가
        print("\n[5/5] User.text_embedding 컬럼 추가 중...")
        try:
            session.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS text_embedding JSONB"
            ))
            session.commit()
            print("  [OK] text_embedding 컬럼 추가 완료")
        except Exception as e:
            print(f"  [INFO] text_embedding 컬럼이 이미 존재하거나 오류: {e}")
            session.rollback()
        
        # 검증
        print("\n[검증] 컬럼 추가 확인 중...")
        result = session.execute(text(
            """
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'reviews' 
            AND column_name IN ('is_taste_test', 'taste_test_type', 'taste_test_weight', 'business_id', 'stars')
            ORDER BY column_name
            """
        ))
        
        print("\n  Reviews 테이블 컬럼:")
        for row in result:
            print(f"    - {row[0]}: {row[1]} (nullable: {row[2]})")
        
        result = session.execute(text(
            """
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'users' 
            AND column_name = 'text_embedding'
            """
        ))
        
        print("\n  Users 테이블 컬럼:")
        for row in result:
            print(f"    - {row[0]}: {row[1]} (nullable: {row[2]})")
        
        print("\n" + "=" * 80)
        print("마이그레이션 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        raise
    
    finally:
        session.close()


if __name__ == "__main__":
    add_taste_test_columns()






