"""
Step 2: Users 데이터 삽입
- 42k users 준비
- bulk_insert_mappings로 삽입
- 예상 시간: 1-2분
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from sqlalchemy import text
from backend_web.database import SessionLocal
from backend_web import models
from backend_web.auth import get_password_hash
from datetime import datetime, timezone

BATCH_SIZE = 10000

print("=" * 80)
print("Step 2: Users 데이터 삽입")
print("=" * 80)

# 데이터 로딩
print("\n[1/4] 데이터 로딩 중...")
user_orig = pd.read_csv("data/processed/user_filtered.csv")
user_preprocessed = pd.read_csv("data/processed/user_preprocessed.csv")
print(f"  원본: {len(user_orig):,}명")

# 병합
print("\n[2/4] 데이터 병합 중...")
merged = user_orig.merge(
    user_preprocessed[['user_id'] + [c for c in user_preprocessed.columns if c.startswith('absa_')]], 
    on='user_id', 
    how='left'
)
print(f"  병합 완료: {len(merged):,}명")

# 데이터 변환
print("\n[3/4] 데이터 변환 중...")
absa_columns = [c for c in merged.columns if c.startswith('absa_')]
user_batch = []

for idx, row in merged.iterrows():
    # ABSA JSON
    absa_dict = {}
    for col in absa_columns:
        key = col.replace('absa_', '')
        value = float(row[col]) if not pd.isna(row[col]) else 0.0
        absa_dict[key] = value
    
    user_dict = {
        'yelp_user_id': row['user_id'],
        'username': f"yelp_{row['user_id'][:8]}",
        'email': f"yelp_{row['user_id'][:8]}@yelp.com",
        'hashed_password': get_password_hash("yelp2024"),
        'review_count': int(row['review_count']) if not pd.isna(row['review_count']) else 0,
        'useful': int(row['useful'] + row.get('funny', 0) + row.get('cool', 0)) if not pd.isna(row['useful']) else 0,
        'compliment': int(sum([row.get(f'compliment_{t}', 0) for t in ['hot', 'more', 'profile', 'cute', 'list', 'note', 'plain', 'cool', 'funny', 'writer', 'photos']])),
        'fans': int(row['fans']) if not pd.isna(row['fans']) else 0,
        'average_stars': float(row['average_stars']) if not pd.isna(row['average_stars']) else 0.0,
        'yelping_since_days': 0,
        'absa_features': absa_dict,
        'created_at': datetime.now(timezone.utc)
    }
    user_batch.append(user_dict)
    
    if (idx + 1) % 10000 == 0:
        print(f"  진행: {idx+1:,} / {len(merged):,} ({(idx+1)/len(merged)*100:.1f}%)")

print(f"  변환 완료: {len(user_batch):,}개")

# Bulk Insert
print(f"\n[4/4] PostgreSQL 삽입 중 (배치: {BATCH_SIZE:,})...")
session = SessionLocal()

try:
    # 최적화 설정
    session.execute(text("SET synchronous_commit = OFF"))
    session.commit()
    
    total_inserted = 0
    for i in range(0, len(user_batch), BATCH_SIZE):
        batch = user_batch[i:i+BATCH_SIZE]
        session.bulk_insert_mappings(models.User, batch)
        session.commit()
        session.expunge_all()
        
        total_inserted += len(batch)
        print(f"  진행: {total_inserted:,} / {len(user_batch):,} ({total_inserted/len(user_batch)*100:.1f}%)")
    
    # 설정 원복
    session.execute(text("SET synchronous_commit = ON"))
    session.commit()
    
    # 확인
    final_count = session.query(models.User).count()
    
    print("\n" + "=" * 80)
    print(f"[SUCCESS] Step 2 완료!")
    print(f"  삽입: {total_inserted:,}명")
    print(f"  확인: {final_count:,}명")
    print("=" * 80)
    print("\n다음: python scripts/migration/step3_insert_businesses.py")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    session.rollback()
finally:
    session.close()



