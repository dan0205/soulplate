"""
Bulk Insert를 사용한 데이터 마이그레이션
- SQLAlchemy bulk_insert_mappings 사용
- ABSA 제외하고 빠르게 삽입
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from sqlalchemy import text
from backend_web.database import SessionLocal, engine
from backend_web import models
from backend_web.auth import get_password_hash
import json
import os
from datetime import datetime, timezone
import time

def truncate_all_tables():
    """모든 테이블 데이터 삭제 (구조는 유지)"""
    print("=" * 80)
    print("1단계: 기존 데이터 삭제")
    print("=" * 80)
    
    session = SessionLocal()
    try:
        print("\n[DELETE] 모든 테이블 데이터 삭제 중...")
        
        # Foreign key 순서 고려하여 삭제
        session.query(models.Review).delete()
        session.commit()
        print(f"  [OK] Reviews 삭제 완료")
        
        session.query(models.UserABSAFeatures).delete()
        session.commit()
        print(f"  [OK] UserABSAFeatures 삭제 완료")
        
        session.query(models.User).delete()
        session.commit()
        print(f"  [OK] Users 삭제 완료")
        
        session.query(models.BusinessABSAFeatures).delete()
        session.commit()
        print(f"  [OK] BusinessABSAFeatures 삭제 완료")
        
        session.query(models.Business).delete()
        session.commit()
        print(f"  [OK] Businesses 삭제 완료")
        
        print("[OK] 기존 데이터 삭제 완료")
        
    except Exception as e:
        print(f"[ERROR] 삭제 실패: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def insert_users():
    """User 데이터 삽입"""
    print("\n" + "=" * 80)
    print("2단계: Users 데이터 삽입")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/3] 데이터 로딩...")
    user_orig = pd.read_csv("data/processed/user_filtered.csv")
    print(f"  원본: {len(user_orig):,}명")
    
    # 데이터 준비
    print("\n[2/3] 데이터 준비...")
    records = []
    
    for idx, row in user_orig.iterrows():
        record = {
            'yelp_user_id': row['user_id'],
            'username': f"yelp_{row['user_id'][:8]}",
            'email': f"yelp_{row['user_id'][:8]}@yelp.com",
            'hashed_password': get_password_hash("yelp2024"),
            'created_at': datetime.now(timezone.utc),
            'review_count': int(row['review_count']) if not pd.isna(row['review_count']) else 0,
            'useful': int(row['useful'] + row.get('funny', 0) + row.get('cool', 0)) if not pd.isna(row['useful']) else 0,
            'compliment': int(sum([row.get(f'compliment_{t}', 0) for t in ['hot', 'more', 'profile', 'cute', 'list', 'note', 'plain', 'cool', 'funny', 'writer', 'photos']])),
            'fans': int(row['fans']) if not pd.isna(row['fans']) else 0,
            'average_stars': float(row['average_stars']) if not pd.isna(row['average_stars']) else 0.0,
            'yelping_since_days': 0
        }
        records.append(record)
        
        if (idx + 1) % 10000 == 0:
            print(f"  준비: {idx+1:,} / {len(user_orig):,}")
    
    # Bulk insert
    print(f"\n[3/3] DB 삽입 중... ({len(records):,}개)")
    session = SessionLocal()
    
    try:
        batch_size = 5000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            session.bulk_insert_mappings(models.User, batch)
            session.commit()
            print(f"  진행: {min(i+batch_size, len(records)):,} / {len(records):,}")
        
        print(f"[OK] Users 삽입 완료: {len(records):,}명")
        return len(records)
        
    except Exception as e:
        print(f"[ERROR] Users 삽입 실패: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def insert_businesses():
    """Business 데이터 삽입"""
    print("\n" + "=" * 80)
    print("3단계: Businesses 데이터 삽입")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/3] 데이터 로딩...")
    business_orig = pd.read_csv("data/processed/business_filtered.csv")
    print(f"  원본: {len(business_orig):,}개")
    
    # 데이터 준비
    print("\n[2/3] 데이터 준비...")
    records = []
    
    for idx, row in business_orig.iterrows():
        record = {
            'business_id': row['business_id'],
            'name': row['name'] if not pd.isna(row['name']) else "Unknown",
            'categories': row['categories'] if not pd.isna(row['categories']) else "",
            'stars': float(row['stars']) if not pd.isna(row['stars']) else 0.0,
            'review_count': int(row['review_count']) if not pd.isna(row['review_count']) else 0,
            'address': row['address'] if not pd.isna(row['address']) else "",
            'city': row['city'] if not pd.isna(row['city']) else "",
            'state': row['state'] if not pd.isna(row['state']) else "",
            'latitude': float(row['latitude']) if not pd.isna(row['latitude']) else 0.0,
            'longitude': float(row['longitude']) if not pd.isna(row['longitude']) else 0.0
        }
        records.append(record)
    
    # Bulk insert
    print(f"\n[3/3] DB 삽입 중... ({len(records):,}개)")
    session = SessionLocal()
    
    try:
        batch_size = 5000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            session.bulk_insert_mappings(models.Business, batch)
            session.commit()
            print(f"  진행: {min(i+batch_size, len(records)):,} / {len(records):,}")
        
        print(f"[OK] Businesses 삽입 완료: {len(records):,}개")
        return len(records)
        
    except Exception as e:
        print(f"[ERROR] Businesses 삽입 실패: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def insert_reviews():
    """Review 데이터 삽입"""
    print("\n" + "=" * 80)
    print("4단계: Reviews 데이터 삽입")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/4] 데이터 로딩...")
    review_absa = pd.read_csv("data/processed/review_absa_features.csv")
    print(f"  리뷰: {len(review_absa):,}개")
    
    # User/Business ID 매핑
    print("\n[2/4] User/Business ID 매핑...")
    session = SessionLocal()
    
    # Yelp user_id -> DB id 매핑
    users = session.query(models.User.id, models.User.yelp_user_id).all()
    user_map = {yelp_id: db_id for db_id, yelp_id in users}
    print(f"  User 매핑: {len(user_map):,}개")
    
    # Business business_id -> DB id 매핑
    businesses = session.query(models.Business.id, models.Business.business_id).all()
    business_map = {biz_id: db_id for db_id, biz_id in businesses}
    print(f"  Business 매핑: {len(business_map):,}개")
    
    session.close()
    
    # 데이터 준비
    print("\n[3/4] 데이터 준비...")
    records = []
    skipped = 0
    
    # ABSA 컬럼 추출
    absa_columns = [c for c in review_absa.columns if c.startswith('absa_')]
    
    for idx, row in review_absa.iterrows():
        # User/Business 매핑 확인
        user_id = user_map.get(row['user_id'])
        business_id = business_map.get(row['business_id'])
        
        if user_id is None or business_id is None:
            skipped += 1
            continue
        
        # ABSA JSON 생성
        absa_dict = {}
        for col in absa_columns:
            key = col.replace('absa_', '')
            value = float(row[col]) if not pd.isna(row[col]) else 0.0
            absa_dict[key] = value
        
        # 날짜 파싱
        try:
            review_date = pd.to_datetime(row['date']).to_pydatetime()
        except:
            review_date = None
        
        record = {
            'user_id': user_id,
            'business_id': business_id,
            'stars': float(row['stars']),
            'text': str(row['text'])[:5000],  # 길이 제한
            'date': review_date,
            'created_at': datetime.now(timezone.utc),
            'absa_features': absa_dict
        }
        records.append(record)
        
        if (idx + 1) % 10000 == 0:
            print(f"  준비: {idx+1:,} / {len(review_absa):,} (스킵: {skipped:,})")
    
    print(f"  최종: {len(records):,}개 (스킵: {skipped:,}개)")
    
    # Bulk insert
    print(f"\n[4/4] DB 삽입 중... ({len(records):,}개)")
    session = SessionLocal()
    
    try:
        batch_size = 5000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            session.bulk_insert_mappings(models.Review, batch)
            session.commit()
            session.expunge_all()  # 메모리 정리
            print(f"  진행: {min(i+batch_size, len(records)):,} / {len(records):,}")
        
        print(f"[OK] Reviews 삽입 완료: {len(records):,}개")
        return len(records)
        
    except Exception as e:
        print(f"[ERROR] Reviews 삽입 실패: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        raise
    finally:
        session.close()


def verify_data():
    """데이터 검증"""
    print("\n" + "=" * 80)
    print("5단계: 데이터 검증")
    print("=" * 80)
    
    session = SessionLocal()
    
    user_count = session.query(models.User).count()
    business_count = session.query(models.Business).count()
    review_count = session.query(models.Review).count()
    
    print(f"\n[결과]")
    print(f"  Users: {user_count:,}명")
    print(f"  Businesses: {business_count:,}개")
    print(f"  Reviews: {review_count:,}개")
    
    # 샘플 확인
    sample_user = session.query(models.User).first()
    if sample_user:
        print(f"\n[샘플 User]")
        print(f"  username: {sample_user.username}")
        print(f"  email: {sample_user.email}")
        print(f"  review_count: {sample_user.review_count}")
    
    sample_business = session.query(models.Business).first()
    if sample_business:
        print(f"\n[샘플 Business]")
        print(f"  name: {sample_business.name}")
        print(f"  stars: {sample_business.stars}")
        print(f"  city: {sample_business.city}")
    
    sample_review = session.query(models.Review).first()
    if sample_review:
        print(f"\n[샘플 Review]")
        print(f"  stars: {sample_review.stars}")
        print(f"  text: {sample_review.text[:100]}...")
        print(f"  ABSA keys: {list(sample_review.absa_features.keys())[:5] if sample_review.absa_features else 'None'}...")
    
    session.close()
    
    return user_count, business_count, review_count


def main():
    """메인 실행"""
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("Bulk Insert 마이그레이션 시작 (ABSA 제외)")
    print("=" * 80)
    
    try:
        # 1. 기존 데이터 삭제
        truncate_all_tables()
        
        # 2. Users
        users_count = insert_users()
        
        # 3. Businesses
        businesses_count = insert_businesses()
        
        # 4. Reviews
        reviews_count = insert_reviews()
        
        # 5. 검증
        final_users, final_businesses, final_reviews = verify_data()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("[SUCCESS] Bulk Insert 마이그레이션 완료!")
        print("=" * 80)
        print(f"\n소요 시간: {elapsed:.1f}초 ({elapsed/60:.2f}분)")
        print(f"처리 속도: {(final_users + final_businesses + final_reviews) / elapsed:.0f} rows/sec")
        print("\n다음 단계:")
        print("  python scripts/aggregate_absa_features.py  # ABSA 집계 (5-10분)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

