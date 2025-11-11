"""
리뷰 개수 분석
"""

import pandas as pd

# 데이터 로드
print("데이터 로딩 중...")
df = pd.read_csv('data/processed/review_100k_absa.csv')
print(f"총 리뷰 수: {len(df):,}개\n")

# User별 리뷰 개수
print("=" * 60)
print("User별 리뷰 분포")
print("=" * 60)
user_counts = df.groupby('user_id').size()
print(f"총 User 수: {len(user_counts):,}명")
print(f"리뷰 10개 이상인 User: {(user_counts >= 10).sum():,}명")
print(f"리뷰 5개 이상인 User: {(user_counts >= 5).sum():,}명")
print(f"리뷰 3개 이상인 User: {(user_counts >= 3).sum():,}명")
print(f"리뷰 1개인 User: {(user_counts == 1).sum():,}명")
print(f"평균 리뷰 수: {user_counts.mean():.2f}개")
print(f"최대 리뷰 수: {user_counts.max():,}개")

# Business별 리뷰 개수
print("\n" + "=" * 60)
print("Business별 리뷰 분포")
print("=" * 60)
business_counts = df.groupby('business_id').size()
print(f"총 Business 수: {len(business_counts):,}개")
print(f"리뷰 10개 이상인 Business: {(business_counts >= 10).sum():,}개")
print(f"리뷰 5개 이상인 Business: {(business_counts >= 5).sum():,}개")
print(f"리뷰 3개 이상인 Business: {(business_counts >= 3).sum():,}개")
print(f"리뷰 1개인 Business: {(business_counts == 1).sum():,}개")
print(f"평균 리뷰 수: {business_counts.mean():.2f}개")
print(f"최대 리뷰 수: {business_counts.max():,}개")

# 상위 User
print("\n" + "=" * 60)
print("리뷰를 가장 많이 작성한 User Top 10")
print("=" * 60)
top_users = user_counts.sort_values(ascending=False).head(10)
for i, (user_id, count) in enumerate(top_users.items(), 1):
    print(f"{i:2d}. {user_id}: {count:,}개")

# 상위 Business
print("\n" + "=" * 60)
print("리뷰를 가장 많이 받은 Business Top 10")
print("=" * 60)
top_businesses = business_counts.sort_values(ascending=False).head(10)
for i, (biz_id, count) in enumerate(top_businesses.items(), 1):
    print(f"{i:2d}. {biz_id}: {count:,}개")

