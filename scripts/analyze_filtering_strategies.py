"""
다양한 필터링 전략 비교 (데모용 소규모 데이터셋)
"""

import pandas as pd

print("데이터 로딩 중...")
df = pd.read_csv('data/processed/review_100k_absa.csv')
print(f"원본 리뷰 수: {len(df):,}개\n")

business_counts = df.groupby('business_id').size()
user_counts = df.groupby('user_id').size()

print("=" * 80)
print("전략 비교: 다양한 필터링 옵션")
print("=" * 80)

# 전략 1: Business 리뷰 30개 이상
print("\n[전략 1] Business 리뷰 30개 이상")
print("-" * 80)
biz_30 = business_counts[business_counts >= 30].index
df_1 = df[df['business_id'].isin(biz_30)]
users_1 = df_1['user_id'].nunique()
print(f"Business: {len(biz_30):,}개 | User: {users_1:,}명 | Review: {len(df_1):,}개")
print(f"Business당 평균: {df_1.groupby('business_id').size().mean():.1f}개")

# 전략 2: Business 리뷰 50개 이상
print("\n[전략 2] Business 리뷰 50개 이상")
print("-" * 80)
biz_50 = business_counts[business_counts >= 50].index
df_2 = df[df['business_id'].isin(biz_50)]
users_2 = df_2['user_id'].nunique()
print(f"Business: {len(biz_50):,}개 | User: {users_2:,}명 | Review: {len(df_2):,}개")
print(f"Business당 평균: {df_2.groupby('business_id').size().mean():.1f}개")

# 전략 3: Business 20개 이상 + User 리뷰 3개 이상
print("\n[전략 3] Business 20개 이상 + User 리뷰 3개 이상")
print("-" * 80)
biz_20 = business_counts[business_counts >= 20].index
df_temp = df[df['business_id'].isin(biz_20)]
user_counts_filtered = df_temp.groupby('user_id').size()
users_3plus = user_counts_filtered[user_counts_filtered >= 3].index
df_3 = df_temp[df_temp['user_id'].isin(users_3plus)]
print(f"Business: {df_3['business_id'].nunique():,}개 | User: {len(users_3plus):,}명 | Review: {len(df_3):,}개")
print(f"Business당 평균: {df_3.groupby('business_id').size().mean():.1f}개")
print(f"User당 평균: {df_3.groupby('user_id').size().mean():.1f}개")

# 전략 4: Business 20개 이상 + User 리뷰 5개 이상
print("\n[전략 4] Business 20개 이상 + User 리뷰 5개 이상")
print("-" * 80)
users_5plus = user_counts_filtered[user_counts_filtered >= 5].index
df_4 = df_temp[df_temp['user_id'].isin(users_5plus)]
print(f"Business: {df_4['business_id'].nunique():,}개 | User: {len(users_5plus):,}명 | Review: {len(df_4):,}개")
print(f"Business당 평균: {df_4.groupby('business_id').size().mean():.1f}개")
print(f"User당 평균: {df_4.groupby('user_id').size().mean():.1f}개")

# 전략 5: Business Top 300 (리뷰 많은 순)
print("\n[전략 5] 리뷰가 가장 많은 Business Top 300")
print("-" * 80)
top_300_biz = business_counts.nlargest(300).index
df_5 = df[df['business_id'].isin(top_300_biz)]
users_5 = df_5['user_id'].nunique()
print(f"Business: 300개 | User: {users_5:,}명 | Review: {len(df_5):,}개")
print(f"Business당 평균: {df_5.groupby('business_id').size().mean():.1f}개")

# 전략 6: Business Top 500
print("\n[전략 6] 리뷰가 가장 많은 Business Top 500")
print("-" * 80)
top_500_biz = business_counts.nlargest(500).index
df_6 = df[df['business_id'].isin(top_500_biz)]
users_6 = df_6['user_id'].nunique()
print(f"Business: 500개 | User: {users_6:,}명 | Review: {len(df_6):,}개")
print(f"Business당 평균: {df_6.groupby('business_id').size().mean():.1f}개")

# 전략 7: Business Top 500 + User 리뷰 2개 이상 (해당 Business에서)
print("\n[전략 7] Business Top 500 + User 리뷰 2개 이상 (더 활성 사용자)")
print("-" * 80)
df_temp2 = df[df['business_id'].isin(top_500_biz)]
user_counts_top500 = df_temp2.groupby('user_id').size()
users_2plus = user_counts_top500[user_counts_top500 >= 2].index
df_7 = df_temp2[df_temp2['user_id'].isin(users_2plus)]
print(f"Business: {df_7['business_id'].nunique():,}개 | User: {len(users_2plus):,}명 | Review: {len(df_7):,}개")
print(f"Business당 평균: {df_7.groupby('business_id').size().mean():.1f}개")
print(f"User당 평균: {df_7.groupby('user_id').size().mean():.1f}개")

print("\n" + "=" * 80)
print("권장 사항")
print("=" * 80)
print("\n데모 페이지용으로 추천하는 옵션:")
print("  ✅ [전략 5] Top 300 Business: User ~42k명 (현실적)")
print("  ✅ [전략 6] Top 500 Business: User ~42k명 (더 다양한 가게)")
print("  ⭐ [전략 7] Top 500 + User 2개 이상: User 훨씬 적음 (고품질)")
print("\n💡 전략 7을 추천합니다: ")
print("   - User 수가 크게 줄어들어 DB 부담 감소")
print("   - 활성 사용자만 포함하여 추천 품질 향상")
print("   - Business당 평균 리뷰 수 높음")

