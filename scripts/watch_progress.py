"""
실시간 진행 상황 모니터링 (한 번만 실행)
"""

import sys
sys.path.append('.')

from backend_web.database import SessionLocal
from backend_web import models

session = SessionLocal()

user_count = session.query(models.User).count()
business_count = session.query(models.Business).count()
review_count = session.query(models.Review).count()

print(f"Users:      {user_count:>7,}명")
print(f"Businesses: {business_count:>7,}개")
print(f"Reviews:    {review_count:>7,}개")
print(f"━" * 30)
print(f"합계:       {user_count + business_count + review_count:>7,}개")

session.close()

