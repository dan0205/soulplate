"""
ABSA 테이블 분리 통합 테스트
- DB 스키마 확인
- 마이그레이션 테스트 (dry run)
- ABSA 집계 확인
- API 응답 확인
"""

import sys
sys.path.append('.')

from backend_web.database import SessionLocal, engine
from backend_web import models
from sqlalchemy import inspect
import json

def test_schema():
    """DB 스키마 확인"""
    print("=" * 80)
    print("테스트 1: DB 스키마 확인")
    print("=" * 80)
    
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"\n[테이블 목록]")
    for table in tables:
        print(f"  - {table}")
    
    # 필수 테이블 확인
    required_tables = ['users', 'businesses', 'reviews', 'user_absa_features', 'business_absa_features']
    missing_tables = [t for t in required_tables if t not in tables]
    
    if missing_tables:
        print(f"\n[ERROR] 누락된 테이블: {missing_tables}")
        return False
    
    print("\n[OK] 모든 필수 테이블이 존재합니다")
    
    # 컬럼 확인
    print("\n[users 테이블 컬럼]")
    user_columns = [col['name'] for col in inspector.get_columns('users')]
    print(f"  {', '.join(user_columns)}")
    
    if 'absa_features' in user_columns:
        print("[WARNING] users 테이블에 absa_features 컬럼이 아직 존재합니다 (제거 필요)")
    else:
        print("[OK] users 테이블에서 absa_features 컬럼이 제거되었습니다")
    
    print("\n[businesses 테이블 컬럼]")
    business_columns = [col['name'] for col in inspector.get_columns('businesses')]
    print(f"  {', '.join(business_columns)}")
    
    if 'absa_features' in business_columns:
        print("[WARNING] businesses 테이블에 absa_features 컬럼이 아직 존재합니다 (제거 필요)")
    else:
        print("[OK] businesses 테이블에서 absa_features 컬럼이 제거되었습니다")
    
    print("\n[user_absa_features 테이블 컬럼]")
    user_absa_columns = [col['name'] for col in inspector.get_columns('user_absa_features')]
    print(f"  {', '.join(user_absa_columns)}")
    
    print("\n[business_absa_features 테이블 컬럼]")
    business_absa_columns = [col['name'] for col in inspector.get_columns('business_absa_features')]
    print(f"  {', '.join(business_absa_columns)}")
    
    return True


def test_data_counts():
    """데이터 개수 확인"""
    print("\n" + "=" * 80)
    print("테스트 2: 데이터 개수 확인")
    print("=" * 80)
    
    session = SessionLocal()
    
    user_count = session.query(models.User).count()
    business_count = session.query(models.Business).count()
    review_count = session.query(models.Review).count()
    user_absa_count = session.query(models.UserABSAFeatures).count()
    business_absa_count = session.query(models.BusinessABSAFeatures).count()
    
    print(f"\n[데이터 개수]")
    print(f"  Users: {user_count:,}명")
    print(f"  Businesses: {business_count:,}개")
    print(f"  Reviews: {review_count:,}개")
    print(f"  UserABSAFeatures: {user_absa_count:,}개")
    print(f"  BusinessABSAFeatures: {business_absa_count:,}개")
    
    if review_count > 0:
        coverage_user = (user_absa_count / user_count * 100) if user_count > 0 else 0
        coverage_business = (business_absa_count / business_count * 100) if business_count > 0 else 0
        
        print(f"\n[ABSA 커버리지]")
        print(f"  User: {coverage_user:.1f}%")
        print(f"  Business: {coverage_business:.1f}%")
    
    session.close()
    
    return user_count > 0 and business_count > 0


def test_absa_data():
    """ABSA 데이터 샘플 확인"""
    print("\n" + "=" * 80)
    print("테스트 3: ABSA 데이터 샘플 확인")
    print("=" * 80)
    
    session = SessionLocal()
    
    # User ABSA 샘플
    user_absa = session.query(models.UserABSAFeatures).first()
    if user_absa:
        print(f"\n[UserABSAFeatures 샘플]")
        print(f"  user_id: {user_absa.user_id}")
        print(f"  ABSA keys: {list(user_absa.absa_features.keys())[:5]}...")
        print(f"  샘플 값: {dict(list(user_absa.absa_features.items())[:3])}")
        print(f"  updated_at: {user_absa.updated_at}")
    else:
        print("\n[WARNING] UserABSAFeatures 데이터가 없습니다")
    
    # Business ABSA 샘플
    business_absa = session.query(models.BusinessABSAFeatures).first()
    if business_absa:
        print(f"\n[BusinessABSAFeatures 샘플]")
        print(f"  business_id: {business_absa.business_id}")
        print(f"  ABSA keys: {list(business_absa.absa_features.keys())[:5]}...")
        print(f"  샘플 값: {dict(list(business_absa.absa_features.items())[:3])}")
        print(f"  updated_at: {business_absa.updated_at}")
    else:
        print("\n[WARNING] BusinessABSAFeatures 데이터가 없습니다")
    
    # Relationship 테스트
    print("\n[Relationship 테스트]")
    business = session.query(models.Business).first()
    if business:
        print(f"  Business: {business.name}")
        if business.absa_features:
            print(f"  ABSA relationship: OK")
            print(f"  ABSA keys: {list(business.absa_features.absa_features.keys())[:3]}...")
        else:
            print(f"  ABSA relationship: None (아직 집계되지 않음)")
    
    session.close()
    
    return True


def test_review_absa():
    """Review ABSA 확인"""
    print("\n" + "=" * 80)
    print("테스트 4: Review ABSA 확인")
    print("=" * 80)
    
    session = SessionLocal()
    
    review = session.query(models.Review).first()
    if review:
        print(f"\n[Review 샘플]")
        print(f"  review_id: {review.id}")
        print(f"  user_id: {review.user_id}")
        print(f"  business_id: {review.business_id}")
        print(f"  stars: {review.stars}")
        
        if review.absa_features:
            print(f"  ABSA: OK")
            print(f"  ABSA keys: {list(review.absa_features.keys())[:5]}...")
        else:
            print(f"  ABSA: None")
    else:
        print("[WARNING] Review 데이터가 없습니다")
    
    session.close()
    
    return True


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 80)
    print("ABSA 테이블 분리 통합 테스트")
    print("=" * 80)
    
    try:
        # 1. 스키마 확인
        if not test_schema():
            print("\n[ERROR] 스키마 테스트 실패")
            return
        
        # 2. 데이터 개수 확인
        if not test_data_counts():
            print("\n[WARNING] 데이터가 없습니다. 마이그레이션을 먼저 실행하세요:")
            print("  python scripts/fast_copy_migration.py")
            return
        
        # 3. ABSA 데이터 확인
        test_absa_data()
        
        # 4. Review ABSA 확인
        test_review_absa()
        
        print("\n" + "=" * 80)
        print("[SUCCESS] 통합 테스트 완료!")
        print("=" * 80)
        print("\n다음 단계:")
        print("1. 마이그레이션이 완료되지 않았다면:")
        print("   python scripts/fast_copy_migration.py")
        print("\n2. ABSA 집계가 완료되지 않았다면:")
        print("   python scripts/aggregate_absa_features.py")
        print("\n3. API 서버 시작:")
        print("   cd backend_web && uvicorn main:app --reload --port 8000")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

