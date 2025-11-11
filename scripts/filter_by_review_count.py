"""
리뷰 10개 이상 받은 Business 기준으로 필터링
"""

import pandas as pd

print("데이터 로딩 중...")
df = pd.read_csv('data/processed/review_100k_absa.csv')
print(f"원본 리뷰 수: {len(df):,}개\n")

# Step 1: Business별 리뷰 개수 계산 및 필터링
print("=" * 60)
print("Step 1: Business 필터링 (리뷰 10개 이상)")
print("=" * 60)
business_counts = df.groupby('business_id').size()
businesses_10plus = business_counts[business_counts >= 10].index.tolist()
print(f"리뷰 10개 이상 받은 Business: {len(businesses_10plus):,}개")

# Step 2: 해당 Business들의 리뷰만 필터링
df_filtered = df[df['business_id'].isin(businesses_10plus)]
print(f"필터링된 리뷰 수: {len(df_filtered):,}개")

# Step 3: 해당 리뷰들에 연결된 User 추출
print("\n" + "=" * 60)
print("Step 2: User 필터링 (필터링된 Business와 연결된)")
print("=" * 60)
users_connected = df_filtered['user_id'].unique().tolist()
print(f"연결된 User 수: {len(users_connected):,}명")

# 최종 통계
print("\n" + "=" * 60)
print("최종 결과 요약")
print("=" * 60)
print(f"Business (리뷰 10개 이상): {len(businesses_10plus):,}개")
print(f"User (위 Business와 연결): {len(users_connected):,}명")
print(f"Review (위 Business와 User 연결): {len(df_filtered):,}개")

# 데이터 분포
print("\n" + "=" * 60)
print("필터링 후 분포")
print("=" * 60)
user_review_counts = df_filtered.groupby('user_id').size()
business_review_counts = df_filtered.groupby('business_id').size()

print(f"\nUser당 평균 리뷰 수: {user_review_counts.mean():.2f}개")
print(f"User당 최대 리뷰 수: {user_review_counts.max():,}개")
print(f"User당 최소 리뷰 수: {user_review_counts.min():,}개")

print(f"\nBusiness당 평균 리뷰 수: {business_review_counts.mean():.2f}개")
print(f"Business당 최대 리뷰 수: {business_review_counts.max():,}개")
print(f"Business당 최소 리뷰 수: {business_review_counts.min():,}개 (필터: 10개)")

# 원본 대비 비율
print("\n" + "=" * 60)
print("원본 대비 비율")
print("=" * 60)
print(f"Business: {len(businesses_10plus) / 14519 * 100:.1f}% 유지")
print(f"User: {len(users_connected) / 42223 * 100:.1f}% 유지")
print(f"Review: {len(df_filtered) / 100000 * 100:.1f}% 유지")

