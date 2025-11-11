"""
필터링된 데이터셋 생성
- Business: 리뷰 20개 이상
- User: 위 Business에 리뷰 3개 이상 작성
- Review: 위 Business와 User에 해당하는 리뷰
"""

import pandas as pd
import os

print("=" * 80)
print("필터링된 데이터셋 생성")
print("=" * 80)

# 1. 원본 데이터 로딩
print("\n[1/7] 원본 데이터 로딩...")
review_df = pd.read_csv('data/processed/review_100k_absa.csv')
user_orig = pd.read_csv('data/processed/user_filtered.csv')
user_preprocessed = pd.read_csv('data/processed/user_preprocessed.csv')
business_orig = pd.read_csv('data/processed/business_filtered.csv')
business_preprocessed = pd.read_csv('data/processed/business_preprocessed.csv')

print(f"  Reviews: {len(review_df):,}개")
print(f"  Users: {len(user_orig):,}명")
print(f"  Businesses: {len(business_orig):,}개")

# 2. Business 필터링 (리뷰 20개 이상)
print("\n[2/7] Business 필터링 (리뷰 20개 이상)...")
business_counts = review_df.groupby('business_id').size()
businesses_20plus = business_counts[business_counts >= 20].index.tolist()
print(f"  필터링된 Business: {len(businesses_20plus):,}개")

# 3. 해당 Business의 리뷰만 필터링
print("\n[3/7] Review 1차 필터링...")
review_filtered = review_df[review_df['business_id'].isin(businesses_20plus)]
print(f"  필터링된 Review: {len(review_filtered):,}개")

# 4. User 필터링 (리뷰 3개 이상)
print("\n[4/7] User 필터링 (리뷰 3개 이상)...")
user_counts = review_filtered.groupby('user_id').size()
users_3plus = user_counts[user_counts >= 3].index.tolist()
print(f"  필터링된 User: {len(users_3plus):,}명")

# 5. 최종 Review 필터링
print("\n[5/7] Review 최종 필터링...")
review_final = review_filtered[review_filtered['user_id'].isin(users_3plus)]
print(f"  최종 Review: {len(review_final):,}개")

# 최종 Business 수 확인 (일부 Business는 User 필터링 후 제외될 수 있음)
final_businesses = review_final['business_id'].unique().tolist()
print(f"  최종 Business: {len(final_businesses):,}개")

# 6. CSV 생성 준비
print("\n[6/7] CSV 파일 생성 중...")
os.makedirs('data/filtered', exist_ok=True)

# User CSV (원본 + 전처리된 ABSA)
print("  - user_filtered_20_3.csv")
user_merged = user_orig.merge(
    user_preprocessed[['user_id'] + [c for c in user_preprocessed.columns if c.startswith('absa_')]],
    on='user_id',
    how='left'
)
user_final = user_merged[user_merged['user_id'].isin(users_3plus)]
user_final.to_csv('data/filtered/user_filtered_20_3.csv', index=False)
print(f"    저장: {len(user_final):,}명")

# Business CSV (원본 + 전처리된 ABSA)
print("  - business_filtered_20_3.csv")
business_merged = business_orig.merge(
    business_preprocessed[['business_id'] + [c for c in business_preprocessed.columns if c.startswith('absa_')]],
    on='business_id',
    how='left'
)
business_final = business_merged[business_merged['business_id'].isin(final_businesses)]
business_final.to_csv('data/filtered/business_filtered_20_3.csv', index=False)
print(f"    저장: {len(business_final):,}개")

# Review CSV
print("  - review_filtered_20_3.csv")
review_final.to_csv('data/filtered/review_filtered_20_3.csv', index=False)
print(f"    저장: {len(review_final):,}개")

# 7. 통계 출력
print("\n[7/7] 최종 통계")
print("=" * 80)
print(f"Users:      {len(user_final):,}명")
print(f"Businesses: {len(business_final):,}개")
print(f"Reviews:    {len(review_final):,}개")
print()
print(f"User당 평균 리뷰:     {len(review_final)/len(user_final):.2f}개")
print(f"Business당 평균 리뷰: {len(review_final)/len(business_final):.2f}개")

print("\n" + "=" * 80)
print("[SUCCESS] 필터링된 데이터셋 생성 완료!")
print("=" * 80)
print("\n다음 단계:")
print("  python scripts/migrate_filtered_data.py")

