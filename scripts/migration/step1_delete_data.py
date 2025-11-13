"""
Step 1: 기존 데이터 삭제
- Reviews -> Users -> Businesses 순서로 삭제
- 삭제 후 count 확인
"""

import sys
sys.path.append('.')

from backend_web.database import SessionLocal
from backend_web import models

print("=" * 80)
print("Step 1: 기존 데이터 삭제")
print("=" * 80)

session = SessionLocal()

try:
    # 삭제 전 count
    print("\n[삭제 전]")
    review_count = session.query(models.Review).count()
    user_count = session.query(models.User).count()
    business_count = session.query(models.Business).count()
    print(f"  Reviews: {review_count:,}개")
    print(f"  Users: {user_count:,}개")
    print(f"  Businesses: {business_count:,}개")
    
    # 삭제
    print("\n[삭제 중...]")
    
    if review_count > 0:
        print(f"  Reviews 삭제 중... ", end="", flush=True)
        session.query(models.Review).delete()
        session.commit()
        print("OK")
    
    if user_count > 0:
        print(f"  Users 삭제 중... ", end="", flush=True)
        session.query(models.User).delete()
        session.commit()
        print("OK")
    
    if business_count > 0:
        print(f"  Businesses 삭제 중... ", end="", flush=True)
        session.query(models.Business).delete()
        session.commit()
        print("OK")
    
    # 삭제 후 확인
    print("\n[삭제 후]")
    review_count = session.query(models.Review).count()
    user_count = session.query(models.User).count()
    business_count = session.query(models.Business).count()
    print(f"  Reviews: {review_count:,}개")
    print(f"  Users: {user_count:,}개")
    print(f"  Businesses: {business_count:,}개")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Step 1 완료!")
    print("=" * 80)
    print("\n다음: python scripts/migration/step2_insert_users.py")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
finally:
    session.close()





