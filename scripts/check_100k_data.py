"""
100k 데이터 통계 확인 스크립트
"""

import pandas as pd
from pathlib import Path

PROCESSED_DATA_DIR = Path("data/processed")

# Load 100k data
print("Loading review_100k_translated.csv...")
df = pd.read_csv(PROCESSED_DATA_DIR / "review_100k_translated.csv")

print("\n" + "=" * 60)
print("100k Data Statistics")
print("=" * 60)

print(f"\nTotal rows: {len(df):,}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Unique businesses: {df['business_id'].nunique():,}")
print(f"\nColumns: {list(df.columns)}")

print(f"\nStar distribution:")
for star in sorted(df['stars'].unique()):
    count = (df['stars'] == star).sum()
    pct = count / len(df) * 100
    print(f"  {star} stars: {count:,} ({pct:.1f}%)")

print(f"\nDate range:")
print(f"  Earliest: {df['date'].min()}")
print(f"  Latest: {df['date'].max()}")

print("\n" + "=" * 60)



