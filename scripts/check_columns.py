import pandas as pd

# User 데이터 컬럼 확인
print("=" * 80)
print("User 데이터 컬럼:")
print("=" * 80)
user_df = pd.read_csv('data/processed/user_filtered.csv', nrows=0)
for i, col in enumerate(user_df.columns, 1):
    print(f"{i:2d}. {col}")
print(f"\n총 {len(user_df.columns)}개 컬럼\n")

# Business 데이터 컬럼 확인
print("=" * 80)
print("Business 데이터 컬럼:")
print("=" * 80)
business_df = pd.read_csv('data/processed/business_filtered.csv', nrows=0)
for i, col in enumerate(business_df.columns, 1):
    print(f"{i:2d}. {col}")
print(f"\n총 {len(business_df.columns)}개 컬럼\n")

# Review 데이터 컬럼 확인
print("=" * 80)
print("Review 데이터 컬럼:")
print("=" * 80)
review_df = pd.read_csv('data/processed/review_100k_translated.csv', nrows=0)
for i, col in enumerate(review_df.columns, 1):
    print(f"{i:2d}. {col}")
print(f"\n총 {len(review_df.columns)}개 컬럼\n")

