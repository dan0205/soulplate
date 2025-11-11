"""
Yelp 데이터를 PostgreSQL로 고속 마이그레이션
- bulk_insert_mappings 사용 (50-100배 빠름)
- PostgreSQL 설정 최적화
- 메모리 효율적 처리
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from tqdm import tqdm
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend_web.database import SessionLocal, engine
from backend_web import models
from backend_web.auth import get_password_hash
import json
from datetime import datetime

BATCH_SIZE = 10000  # 대량 삽입용

def convert_absa_to_json(row, absa_columns):
    """ABSA 피처를 JSON으로 변환"""
    absa_dict = {}
    for col in absa_columns:
        if col.startswith('absa_'):
            key = col.replace('absa_', '')
            value = float(row[col]) if not pd.isna(row[col]) else 0.0
            absa_dict[key] = value
    return absa_dict

def optimize_db_for_bulk_insert(session):
    """대량 삽입을 위한 PostgreSQL 최적화"""
    print("\n[최적화] PostgreSQL 설정 변경 중...")
    try:
        session.execute(text("SET synchronous_commit = OFF"))
        session.execute(text("SET maintenance_work_mem = '512MB'"))
        session.commit()
        print("  ✓ 최적화 설정 적용 완료")
    except Exception as e:
        print(f"  ⚠ 최적화 설정 실패 (무시하고 진행): {e}")

def restore_db_settings(session):
    """PostgreSQL 설정 원복"""
    print("\n[복원] PostgreSQL 설정 원복 중...")
    try:
        session.execute(text("SET synchronous_commit = ON"))
        session.commit()
        print("  ✓ 설정 원복 완료")
    except Exception as e:
        print(f"  ⚠ 설정 원복 실패: {e}")

def migrate_users_fast():
    """User 데이터 고속 마이그레이션"""
    print("=" * 80)
    print("User 데이터 마이그레이션 (고속)")
    print("=" * 80)
    
    # 1. 데이터 로딩
    print("\n[1/3] 데이터 로딩 중...")
    user_orig = pd.read_csv("data/processed/user_filtered.csv")
    user_preprocessed = pd.read_csv("data/processed/user_preprocessed.csv")
    
    print(f"  원본: {len(user_orig):,}명")
    
    # 2. 데이터 병합
    print("\n[2/3] 데이터 준비 중...")
    merged = user_orig.merge(
        user_preprocessed[['user_id'] + [c for c in user_preprocessed.columns if c.startswith('absa_')]], 
        on='user_id', 
        how='left'
    )
    
    absa_columns = [c for c in merged.columns if c.startswith('absa_')]
    
    # 3. dict 리스트로 변환 (bulk_insert_mappings용)
    user_batch = []
    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="  데이터 변환"):
        absa_json = convert_absa_to_json(row, absa_columns)
        
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
            'absa_features': absa_json,
            'created_at': datetime.utcnow()
        }
        user_batch.append(user_dict)
    
    # 4. Bulk Insert
    print(f"\n[3/3] PostgreSQL 고속 삽입 중 (배치 크기: {BATCH_SIZE:,})...")
    session = SessionLocal()
    
    try:
        # 최적화 설정
        optimize_db_for_bulk_insert(session)
        
        # Bulk insert
        total_inserted = 0
        for i in range(0, len(user_batch), BATCH_SIZE):
            batch = user_batch[i:i+BATCH_SIZE]
            session.bulk_insert_mappings(models.User, batch)
            session.commit()
            session.expunge_all()  # 메모리 정리
            
            total_inserted += len(batch)
            print(f"  진행: {total_inserted:,} / {len(user_batch):,} ({total_inserted/len(user_batch)*100:.1f}%)")
        
        # 설정 원복
        restore_db_settings(session)
        
        print(f"\n✓ 완료: {total_inserted:,}명 삽입")
        return total_inserted, 0
        
    except Exception as e:
        print(f"\n✗ 오류: {e}")
        session.rollback()
        return 0, 1
    finally:
        session.close()

def migrate_businesses_fast():
    """Business 데이터 고속 마이그레이션"""
    print("\n" + "=" * 80)
    print("Business 데이터 마이그레이션 (고속)")
    print("=" * 80)
    
    # 1. 데이터 로딩
    print("\n[1/3] 데이터 로딩 중...")
    business_orig = pd.read_csv("data/processed/business_filtered.csv")
    business_preprocessed = pd.read_csv("data/processed/business_preprocessed.csv")
    
    print(f"  원본: {len(business_orig):,}개")
    
    # 2. 데이터 병합
    print("\n[2/3] 데이터 준비 중...")
    merged = business_orig.merge(
        business_preprocessed[['business_id'] + [c for c in business_preprocessed.columns if c.startswith('absa_')]], 
        on='business_id', 
        how='left'
    )
    
    absa_columns = [c for c in merged.columns if c.startswith('absa_')]
    
    # 3. dict 리스트로 변환
    business_batch = []
    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="  데이터 변환"):
        absa_json = convert_absa_to_json(row, absa_columns)
        
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
            'absa_features': absa_json
        }
        business_batch.append(business_dict)
    
    # 4. Bulk Insert
    print(f"\n[3/3] PostgreSQL 고속 삽입 중 (배치 크기: {BATCH_SIZE:,})...")
    session = SessionLocal()
    
    try:
        optimize_db_for_bulk_insert(session)
        
        total_inserted = 0
        for i in range(0, len(business_batch), BATCH_SIZE):
            batch = business_batch[i:i+BATCH_SIZE]
            session.bulk_insert_mappings(models.Business, batch)
            session.commit()
            session.expunge_all()
            
            total_inserted += len(batch)
            print(f"  진행: {total_inserted:,} / {len(business_batch):,} ({total_inserted/len(business_batch)*100:.1f}%)")
        
        restore_db_settings(session)
        
        print(f"\n✓ 완료: {total_inserted:,}개 삽입")
        return total_inserted, 0
        
    except Exception as e:
        print(f"\n✗ 오류: {e}")
        session.rollback()
        return 0, 1
    finally:
        session.close()

def migrate_reviews_fast():
    """Review 데이터 고속 마이그레이션"""
    print("\n" + "=" * 80)
    print("Review 데이터 마이그레이션 (고속)")
    print("=" * 80)
    
    # 1. 데이터 로딩
    print("\n[1/4] 데이터 로딩 중...")
    review_absa = pd.read_csv("data/processed/review_absa_features.csv")
    print(f"  리뷰: {len(review_absa):,}개")
    
    # 2. User/Business ID 매핑
    print("\n[2/4] User/Business ID 매핑 중...")
    session = SessionLocal()
    
    user_map = {}
    users = session.query(models.User.yelp_user_id, models.User.id).filter(models.User.yelp_user_id.isnot(None)).all()
    for yelp_id, db_id in users:
        user_map[yelp_id] = db_id
    print(f"  User 매핑: {len(user_map):,}개")
    
    business_map = {}
    businesses = session.query(models.Business.business_id, models.Business.id).all()
    for biz_id, db_id in businesses:
        business_map[biz_id] = db_id
    print(f"  Business 매핑: {len(business_map):,}개")
    
    # 3. 데이터 필터링 및 준비
    print("\n[3/4] 데이터 준비 중...")
    review_absa['db_user_id'] = review_absa['user_id'].map(user_map)
    review_absa['db_business_id'] = review_absa['business_id'].map(business_map)
    
    valid_reviews = review_absa.dropna(subset=['db_user_id', 'db_business_id'])
    print(f"  유효한 리뷰: {len(valid_reviews):,}개")
    
    # 원본 리뷰 텍스트 로딩
    review_orig = pd.read_csv("data/processed/review_100k_translated.csv")
    valid_reviews = valid_reviews.merge(
        review_orig[['review_id', 'text', 'date']],
        on='review_id',
        how='left'
    )
    
    absa_columns = [c for c in valid_reviews.columns if c.startswith('absa_')]
    
    # dict 리스트로 변환
    review_batch = []
    for idx, row in tqdm(valid_reviews.iterrows(), total=len(valid_reviews), desc="  데이터 변환"):
        absa_json = convert_absa_to_json(row, absa_columns)
        
        review_dict = {
            'user_id': int(row['db_user_id']),
            'business_id': int(row['db_business_id']),
            'stars': float(row['stars']),
            'text': row['text'] if not pd.isna(row['text']) else "",
            'date': pd.to_datetime(row['date']) if not pd.isna(row['date']) else None,
            'absa_features': absa_json,
            'created_at': datetime.utcnow()
        }
        review_batch.append(review_dict)
    
    # 4. Bulk Insert
    print(f"\n[4/4] PostgreSQL 고속 삽입 중 (배치 크기: {BATCH_SIZE:,})...")
    
    try:
        optimize_db_for_bulk_insert(session)
        
        total_inserted = 0
        for i in range(0, len(review_batch), BATCH_SIZE):
            batch = review_batch[i:i+BATCH_SIZE]
            session.bulk_insert_mappings(models.Review, batch)
            session.commit()
            session.expunge_all()
            
            total_inserted += len(batch)
            print(f"  진행: {total_inserted:,} / {len(review_batch):,} ({total_inserted/len(review_batch)*100:.1f}%)")
        
        restore_db_settings(session)
        
        print(f"\n✓ 완료: {total_inserted:,}개 삽입")
        return total_inserted, 0
        
    except Exception as e:
        print(f"\n✗ 오류: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        return 0, 1
    finally:
        session.close()

def verify_migration():
    """마이그레이션 결과 확인"""
    print("\n" + "=" * 80)
    print("마이그레이션 결과 확인")
    print("=" * 80)
    
    session = SessionLocal()
    
    user_count = session.query(models.User).count()
    business_count = session.query(models.Business).count()
    review_count = session.query(models.Review).count()
    
    print(f"\n✓ Users: {user_count:,}명")
    print(f"✓ Businesses: {business_count:,}개")
    print(f"✓ Reviews: {review_count:,}개")
    
    # 샘플 확인
    sample_user = session.query(models.User).first()
    if sample_user:
        print(f"\n샘플 User:")
        print(f"  - ID: {sample_user.id}")
        print(f"  - Username: {sample_user.username}")
        print(f"  - ABSA Keys: {list(sample_user.absa_features.keys())[:3] if sample_user.absa_features else 'None'}...")
    
    session.close()

def main():
    print("\n" + "=" * 80)
    print("Yelp 데이터 고속 마이그레이션")
    print("=" * 80)
    print(f"\n최적화:")
    print(f"  - bulk_insert_mappings (50-100배 빠름)")
    print(f"  - synchronous_commit = OFF")
    print(f"  - 배치 크기: {BATCH_SIZE:,}")
    print(f"  - 메모리 자동 정리\n")
    
    import time
    start_time = time.time()
    
    try:
        # 1. User 마이그레이션
        user_inserted, user_errors = migrate_users_fast()
        
        # 2. Business 마이그레이션
        business_inserted, business_errors = migrate_businesses_fast()
        
        # 3. Review 마이그레이션
        review_inserted, review_errors = migrate_reviews_fast()
        
        # 4. 결과 확인
        verify_migration()
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("[SUCCESS] 고속 마이그레이션 완료!")
        print(f"\n요약:")
        print(f"  Users: {user_inserted:,}명")
        print(f"  Businesses: {business_inserted:,}개")
        print(f"  Reviews: {review_inserted:,}개")
        print(f"\n소요 시간: {elapsed_time/60:.1f}분")
        print(f"처리 속도: {(user_inserted + business_inserted + review_inserted) / elapsed_time:.0f} rows/sec")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

