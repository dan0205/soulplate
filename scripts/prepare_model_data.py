"""
필터링된 데이터로부터 모델 서버에 필요한 파일들을 생성
"""
import pandas as pd
import json
from pathlib import Path

# 경로 설정
DATA_DIR = Path("data")
FILTERED_DIR = DATA_DIR / "filtered"
PROCESSED_DIR = DATA_DIR / "processed"

print("=" * 80)
print("모델 서버용 데이터 파일 생성")
print("=" * 80)

# 1. users.csv 생성
print("\n[1/3] users.csv 생성...")
user_df = pd.read_csv(FILTERED_DIR / "user_filtered_20_3.csv")
# 필요한 컬럼만 선택 (model_loader가 기대하는 컬럼)
users_df = user_df[['user_id', 'review_count', 'useful', 'average_stars']].copy()
# age 컬럼 추가 (기본값 30)
users_df['age'] = 30
users_df.to_csv(PROCESSED_DIR / "users.csv", index=False)
print(f"  [OK] users.csv 생성 완료 ({len(users_df)}명)")

# 2. user_id_to_idx.json 생성
print("\n[2/3] user_id_to_idx.json 생성...")
user_id_to_idx = {row['user_id']: idx for idx, row in users_df.iterrows()}
with open(PROCESSED_DIR / "user_id_to_idx.json", 'w') as f:
    json.dump(user_id_to_idx, f)
print(f"  [OK] user_id_to_idx.json 생성 완료 ({len(user_id_to_idx)}개)")

# 3. business_id_to_idx.json 생성
print("\n[3/3] business_id_to_idx.json 생성...")
business_df = pd.read_csv(FILTERED_DIR / "business_filtered_20_3.csv")
business_id_to_idx = {row['business_id']: idx for idx, row in business_df.iterrows()}
with open(PROCESSED_DIR / "business_id_to_idx.json", 'w') as f:
    json.dump(business_id_to_idx, f)
print(f"  [OK] business_id_to_idx.json 생성 완료 ({len(business_id_to_idx)}개)")

print("\n" + "=" * 80)
print("[OK] 모든 파일 생성 완료!")
print("=" * 80)

