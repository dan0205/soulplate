"""
최적화된 Bulk Insert 마이그레이션
- synchronous_commit = OFF
- 큰 배치 크기 (50,000)
- 메모리 효율적 처리
- 예상 시간: 3-5분
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from sqlalchemy import text
from backend_web.database import SessionLocal
from backend_web import models
from backend_web.auth import get_password_hash
import json
from datetime import datetime, timezone
import time

BATCH_SIZE = 50000  # 대용량 배치

print("\n" + "=" * 80)
print("PostgreSQL 초고속 데이터 마이그레이션")
print("=" * 80)
print(f"\n설정:")
print(f"  - synchronous_commit = OFF")
print(f"  - 배치 크기: {BATCH_SIZE:,}")
print(f"  - 예상 시간: 3-5분\n")

start_time = time.time()

# ============================================================================
# 1단계: 기존 데이터 삭제
# ============================================================================
print("=" * 80)
print("1/4: 기존 데이터 삭제")
print("=" * 80)

session = SessionLocal()

print("\n삭제 중...")
session.query(models.Review).delete()
session.commit()
print("  [OK] Reviews 삭제")

session.query(models.User).delete()
session.commit()
print("  [OK] Users 삭제")

session.query(models.Business).delete()
session.commit()
print("  [OK] Businesses 삭제")

# 최적화 설정
print("\nPostgreSQL 최적화 설정...")
session.execute(text("SET synchronous_commit = OFF"))
session.execute(text("SET maintenance_work_mem = '512MB'"))
session.commit()
print("  [OK] 최적화 완료")

session.close()

# ============================================================================
# 2/4: Users 마이그레이션
# ============================================================================
print("\n" + "=" * 80)
print("2/4: Users 마이그레이션")
print("=" * 80)

print("\n데이터 로딩...")
user_orig = pd.read_csv("data/processed/user_filtered.csv")
user_preprocessed = pd.read_csv("data/processed/user_preprocessed.csv")
print(f"  {len(user_orig):,}명")

print("데이터 병합...")
merged = user_orig.merge(
    user_preprocessed[['user_id'] + [c for c in user_preprocessed.columns if c.startswith('absa_')]], 
    on='user_id', 
    how='left'
)

print("데이터 변환 중...")
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
        print(f"  변환: {idx+1:,} / {len(merged):,}")

print(f"\nBulk Insert 중 ({len(user_batch):,}개)...")
session = SessionLocal()

for i in range(0, len(user_batch), BATCH_SIZE):
    batch = user_batch[i:i+BATCH_SIZE]
    session.bulk_insert_mappings(models.User, batch)
    session.commit()
    session.expunge_all()
    print(f"  진행: {min(i+BATCH_SIZE, len(user_batch)):,} / {len(user_batch):,}")

session.close()
print("[OK] Users 완료")

# ============================================================================
# 3/4: Businesses 마이그레이션
# ============================================================================
print("\n" + "=" * 80)
print("3/4: Businesses 마이그레이션")
print("=" * 80)

print("\n데이터 로딩...")
business_orig = pd.read_csv("data/processed/business_filtered.csv")
business_preprocessed = pd.read_csv("data/processed/business_preprocessed.csv")
print(f"  {len(business_orig):,}개")

print("데이터 병합...")
merged = business_orig.merge(
    business_preprocessed[['business_id'] + [c for c in business_preprocessed.columns if c.startswith('absa_')]], 
    on='business_id', 
    how='left'
)

print("데이터 변환 중...")
absa_columns = [c for c in merged.columns if c.startswith('absa_')]
business_batch = []

for idx, row in merged.iterrows():
    # ABSA JSON
    absa_dict = {}
    for col in absa_columns:
        key = col.replace('absa_', '')
        value = float(row[col]) if not pd.isna(row[col]) else 0.0
        absa_dict[key] = value
    
    business_dict = {
        'business_id': row['business_id'],
        'name': row['name'] if not pd.isna(row['name']) else "Unknown",
        'categories': row['categories'] if not pd.isna(row['categories']) else "",
        'stars': float(row['stars']) if not pd.isna(row['stars']) else 0.0,
        'review_count': int(row['review_count']) if not pd.isna(row['review_count']) else 0,
        'address': row['address'] if not pd.isna(row['address']) else "",
        'city': row['city'] if not pd.isna(row['city']) else "",
        'state': row['state'] if not pd.isna(row['state']) else "",
        'latitude': float(row['latitude']) if not pd.isna(row['latitude']) else 0.0,
        'longitude': float(row['longitude']) if not pd.isna(row['longitude']) else 0.0,
        'absa_features': absa_dict
    }
    business_batch.append(business_dict)

print(f"\nBulk Insert 중 ({len(business_batch):,}개)...")
session = SessionLocal()

for i in range(0, len(business_batch), BATCH_SIZE):
    batch = business_batch[i:i+BATCH_SIZE]
    session.bulk_insert_mappings(models.Business, batch)
    session.commit()
    session.expunge_all()
    print(f"  진행: {min(i+BATCH_SIZE, len(business_batch)):,} / {len(business_batch):,}")

session.close()
print("[OK] Businesses 완료")

# ============================================================================
# 4/4: Reviews 마이그레이션
# ============================================================================
print("\n" + "=" * 80)
print("4/4: Reviews 마이그레이션")
print("=" * 80)

print("\n데이터 로딩...")
review_absa = pd.read_csv("data/processed/review_absa_features.csv")
print(f"  {len(review_absa):,}개")

print("User/Business ID 매핑...")
session = SessionLocal()

user_map = {}
users = session.query(models.User.yelp_user_id, models.User.id).filter(models.User.yelp_user_id.isnot(None)).all()
for yelp_id, db_id in users:
    user_map[yelp_id] = db_id
print(f"  User: {len(user_map):,}개")

business_map = {}
businesses = session.query(models.Business.business_id, models.Business.id).all()
for biz_id, db_id in businesses:
    business_map[biz_id] = db_id
print(f"  Business: {len(business_map):,}개")

print("데이터 필터링...")
review_absa['db_user_id'] = review_absa['user_id'].map(user_map)
review_absa['db_business_id'] = review_absa['business_id'].map(business_map)

valid_reviews = review_absa.dropna(subset=['db_user_id', 'db_business_id'])
print(f"  유효: {len(valid_reviews):,}개")

print("원본 텍스트 로딩...")
review_orig = pd.read_csv("data/processed/review_100k_translated.csv")
valid_reviews = valid_reviews.merge(
    review_orig[['review_id', 'text', 'date']],
    on='review_id',
    how='left'
)

print("데이터 변환 중...")
absa_columns = [c for c in valid_reviews.columns if c.startswith('absa_')]
review_batch = []

for idx, row in valid_reviews.iterrows():
    # ABSA JSON
    absa_dict = {}
    for col in absa_columns:
        key = col.replace('absa_', '')
        value = float(row[col]) if not pd.isna(row[col]) else 0.0
        absa_dict[key] = value
    
    review_dict = {
        'user_id': int(row['db_user_id']),
        'business_id': int(row['db_business_id']),
        'stars': float(row['stars']),
        'text': str(row['text']) if not pd.isna(row['text']) else "",
        'date': pd.to_datetime(row['date']) if not pd.isna(row['date']) else None,
        'absa_features': absa_dict,
        'created_at': datetime.now(timezone.utc)
    }
    review_batch.append(review_dict)
    
    if (idx + 1) % 10000 == 0:
        print(f"  변환: {idx+1:,} / {len(valid_reviews):,}")

print(f"\nBulk Insert 중 ({len(review_batch):,}개)...")

for i in range(0, len(review_batch), BATCH_SIZE):
    batch = review_batch[i:i+BATCH_SIZE]
    session.bulk_insert_mappings(models.Review, batch)
    session.commit()
    session.expunge_all()
    print(f"  진행: {min(i+BATCH_SIZE, len(review_batch)):,} / {len(review_batch):,}")

# 설정 원복
session.execute(text("SET synchronous_commit = ON"))
session.commit()

session.close()
print("[OK] Reviews 완료")

# ============================================================================
# 결과 확인
# ============================================================================
print("\n" + "=" * 80)
print("결과 확인")
print("=" * 80)

session = SessionLocal()
user_count = session.query(models.User).count()
business_count = session.query(models.Business).count()
review_count = session.query(models.Review).count()
session.close()

elapsed = time.time() - start_time

print(f"\n최종 데이터:")
print(f"  Users: {user_count:,}명")
print(f"  Businesses: {business_count:,}개")
print(f"  Reviews: {review_count:,}개")

print(f"\n소요 시간: {elapsed:.1f}초 ({elapsed/60:.2f}분)")
print(f"처리 속도: {(user_count + business_count + review_count) / elapsed:.0f} rows/sec")

print("\n" + "=" * 80)
print("[SUCCESS] 마이그레이션 완료!")
print("이제 프론트엔드에서 추천을 확인하세요!")
print("=" * 80)





