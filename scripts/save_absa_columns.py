"""
ABSA 컬럼 순서 저장
"""
import pandas as pd
import json

# User ABSA 컬럼 순서
user_df = pd.read_csv('data/processed/user_preprocessed.csv')
user_absa_cols = [c for c in user_df.columns if c.startswith('absa_')]

# Business ABSA 컬럼 순서  
business_df = pd.read_csv('data/processed/business_preprocessed.csv')
business_absa_cols = [c for c in business_df.columns if c.startswith('absa_')]

print(f"User ABSA columns: {len(user_absa_cols)}")
print(f"Business ABSA columns: {len(business_absa_cols)}")
print(f"\nFirst 10 User ABSA columns:")
for i, col in enumerate(user_absa_cols[:10], 1):
    print(f"  {i}. {col}")

# 저장
absa_info = {
    'user_absa_columns': user_absa_cols,
    'business_absa_columns': business_absa_cols
}

with open('models/absa_columns.json', 'w', encoding='utf-8') as f:
    json.dump(absa_info, f, ensure_ascii=False, indent=2)

print(f"\n[OK] ABSA 컬럼 순서 저장: models/absa_columns.json")







