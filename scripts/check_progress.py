"""
마이그레이션 진행 상황 체크
"""

import sys
sys.path.append('.')

from backend_web.database import SessionLocal
from backend_web import models
import time

def check_progress():
    """현재 DB 데이터 개수 확인"""
    session = SessionLocal()
    
    try:
        user_count = session.query(models.User).count()
        business_count = session.query(models.Business).count()
        review_count = session.query(models.Review).count()
        
        print(f"\n[현재 DB 상태] {time.strftime('%H:%M:%S')}")
        print(f"  Users: {user_count:,}명")
        print(f"  Businesses: {business_count:,}개")
        print(f"  Reviews: {review_count:,}개")
        print(f"  합계: {user_count + business_count + review_count:,}개")
        
        return user_count, business_count, review_count
    finally:
        session.close()

if __name__ == "__main__":
    print("=" * 60)
    print("마이그레이션 진행 상황 모니터링")
    print("=" * 60)
    print("5초마다 자동으로 업데이트됩니다 (Ctrl+C로 종료)")
    
    prev_total = 0
    
    try:
        while True:
            users, businesses, reviews = check_progress()
            total = users + businesses + reviews
            
            if total > prev_total:
                diff = total - prev_total
                print(f"  -> 증가: +{diff:,}개")
            elif total == prev_total and total > 0:
                print(f"  -> 변화 없음 (완료되었을 수 있음)")
            
            prev_total = total
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n모니터링 종료")

