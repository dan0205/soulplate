"""
PostgreSQL DB 상태 확인
"""
import sys
sys.path.append('.')

from backend_web.database import SessionLocal
from backend_web import models

def check_status():
    print("PostgreSQL DB 상태 확인 중...")
    
    session = SessionLocal()
    
    try:
        user_count = session.query(models.User).count()
        business_count = session.query(models.Business).count()
        review_count = session.query(models.Review).count()
        
        print(f"\n현재 데이터:")
        print(f"  Users: {user_count:,}개")
        print(f"  Businesses: {business_count:,}개")
        print(f"  Reviews: {review_count:,}개")
        
        if user_count == 0 and business_count == 0 and review_count == 0:
            print("\n상태: 데이터베이스가 비어있습니다.")
        elif user_count == 1 or user_count == 2:
            print("\n상태: 테스트 사용자만 있습니다.")
        else:
            print("\n상태: Yelp 데이터가 있습니다!")
        
    except Exception as e:
        print(f"오류: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    check_status()


