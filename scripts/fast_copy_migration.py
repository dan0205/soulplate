"""
PostgreSQL COPY를 사용한 초고속 데이터 마이그레이션
- 예상 시간: 1-2분
- TRUNCATE로 기존 데이터 삭제
- CSV로 변환 후 COPY 명령 실행
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
        # 1. Reviews 먼저 (다른 테이블 참조)
        review_count = session.query(models.Review).count()
        if review_count > 0:
            print(f"  Reviews 삭제 중 ({review_count:,}개)...")
            session.query(models.Review).delete()
            session.commit()
            print(f"  [OK] Reviews 삭제 완료")
        
        # 2. Users 삭제
        user_count = session.query(models.User).count()
        if user_count > 0:
            print(f"  Users 삭제 중 ({user_count:,}개)...")
            session.query(models.User).delete()
            session.commit()
            print(f"  [OK] Users 삭제 완료")
        
        # 3. Businesses 삭제
        business_count = session.query(models.Business).count()
        if business_count > 0:
            print(f"  Businesses 삭제 중 ({business_count:,}개)...")
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

def prepare_users_csv():
    """User CSV 준비"""
    print("\n" + "=" * 80)
    print("2단계: Users CSV 준비")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/2] 데이터 로딩...")
    user_orig = pd.read_csv("data/processed/user_filtered.csv")
    
    print(f"  원본: {len(user_orig):,}명")
    
    # CSV 준비 (ABSA 제외)
    print("\n[2/2] CSV 준비...")
    
    csv_data = []
    for idx, row in user_orig.iterrows():
        csv_row = {
            'yelp_user_id': row['user_id'],
            'username': f"yelp_{row['user_id'][:8]}",
            'email': f"yelp_{row['user_id'][:8]}@yelp.com",
            'hashed_password': get_password_hash("yelp2024"),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'review_count': int(row['review_count']) if not pd.isna(row['review_count']) else 0,
            'useful': int(row['useful'] + row.get('funny', 0) + row.get('cool', 0)) if not pd.isna(row['useful']) else 0,
            'compliment': int(sum([row.get(f'compliment_{t}', 0) for t in ['hot', 'more', 'profile', 'cute', 'list', 'note', 'plain', 'cool', 'funny', 'writer', 'photos']])),
            'fans': int(row['fans']) if not pd.isna(row['fans']) else 0,
            'average_stars': float(row['average_stars']) if not pd.isna(row['average_stars']) else 0.0,
            'yelping_since_days': 0
        }
        csv_data.append(csv_row)
        
        if (idx + 1) % 10000 == 0:
            print(f"  진행: {idx+1:,} / {len(user_orig):,}")
    
    # CSV 저장
    df = pd.DataFrame(csv_data)
    output_path = "data/temp/users_for_copy.csv"
    os.makedirs("data/temp", exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"[OK] CSV 생성 완료: {output_path}")
    return output_path, len(df)

def prepare_businesses_csv():
    """Business CSV 준비"""
    print("\n" + "=" * 80)
    print("3단계: Businesses CSV 준비")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/2] 데이터 로딩...")
    business_orig = pd.read_csv("data/processed/business_filtered.csv")
    
    print(f"  원본: {len(business_orig):,}개")
    
    # CSV 준비 (ABSA 제외)
    print("\n[2/2] CSV 준비...")
    
    csv_data = []
    for idx, row in business_orig.iterrows():
        csv_row = {
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
        csv_data.append(csv_row)
    
    # CSV 저장
    df = pd.DataFrame(csv_data)
    output_path = "data/temp/businesses_for_copy.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"[OK] CSV 생성 완료: {output_path}")
    return output_path, len(df)

def prepare_reviews_csv():
    """Review CSV 준비"""
    print("\n" + "=" * 80)
    print("4단계: Reviews CSV 준비")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/4] 데이터 로딩...")
    review_absa = pd.read_csv("data/processed/review_absa_features.csv")
    print(f"  리뷰: {len(review_absa):,}개")
    
    # User/Business ID 매핑
    print("\n[2/4] User/Business ID 매핑...")
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
    
    session.close()
    
    # 데이터 필터링
    print("\n[3/4] 데이터 필터링...")
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
    
    # CSV 준비
    print("\n[4/4] CSV 준비...")
    absa_columns = [c for c in valid_reviews.columns if c.startswith('absa_')]
    
    csv_data = []
    for idx, row in valid_reviews.iterrows():
        # ABSA JSON 생성
        absa_dict = {}
        for col in absa_columns:
            key = col.replace('absa_', '')
            value = float(row[col]) if not pd.isna(row[col]) else 0.0
            absa_dict[key] = value
        
        csv_row = {
            'user_id': int(row['db_user_id']),
            'business_id': int(row['db_business_id']),
            'stars': float(row['stars']),
            'text': str(row['text']) if not pd.isna(row['text']) else "",
            'date': pd.to_datetime(row['date']).isoformat() if not pd.isna(row['date']) else None,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'absa_features': json.dumps(absa_dict)
        }
        csv_data.append(csv_row)
        
        if (idx + 1) % 10000 == 0:
            print(f"  진행: {idx+1:,} / {len(valid_reviews):,}")
    
    # CSV 저장
    df = pd.DataFrame(csv_data)
    output_path = "data/temp/reviews_for_copy.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"[OK] CSV 생성 완료: {output_path}")
    return output_path, len(df)

def copy_to_database(table_name, csv_path, columns):
    """PostgreSQL COPY 명령으로 데이터 삽입"""
    print(f"\n[COPY] {table_name} 테이블에 데이터 삽입 중...")
    
    session = SessionLocal()
    try:
        # 절대 경로로 변환
        abs_path = os.path.abspath(csv_path)
        
        # Windows 경로를 PostgreSQL 형식으로 변환
        abs_path = abs_path.replace('\\', '/')
        
        # COPY 명령 실행
        copy_sql = f"""
        COPY {table_name} ({', '.join(columns)})
        FROM '{abs_path}'
        WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
        """
        
        session.execute(text(copy_sql))
        session.commit()
        
        print(f"[OK] {table_name} 데이터 삽입 완료")
        
    except Exception as e:
        print(f"[ERROR] {table_name} 삽입 실패: {e}")
        print("\n대안: bulk_insert_mappings 사용...")
        
        # COPY 실패 시 bulk_insert_mappings로 폴백
        session.rollback()
        df = pd.read_csv(csv_path)
        
        # JSONB 컬럼 처리 (Review만)
        if table_name == 'reviews' and 'absa_features' in df.columns:
            df['absa_features'] = df['absa_features'].apply(json.loads)
        
        records = df.to_dict('records')
        
        # 테이블에 맞는 모델 선택
        if table_name == 'users':
            model = models.User
        elif table_name == 'businesses':
            model = models.Business
        elif table_name == 'reviews':
            model = models.Review
        
        # Bulk insert
        batch_size = 10000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            session.bulk_insert_mappings(model, batch)
            session.commit()
            session.expunge_all()
            print(f"  진행: {min(i+batch_size, len(records)):,} / {len(records):,}")
        
        print(f"[OK] {table_name} bulk insert 완료")
        
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
        print(f"  ID: {sample_user.id}")
        print(f"  Username: {sample_user.username}")
        print(f"  ABSA: {'O' if sample_user.absa_features else 'X'}")
    
    session.close()
    
    return user_count, business_count, review_count

def main():
    print("\n" + "=" * 80)
    print("PostgreSQL 초고속 데이터 마이그레이션 (COPY)")
    print("=" * 80)
    print("\n예상 시간: 1-2분\n")
    
    start_time = time.time()
    
    try:
        # 1. 기존 데이터 삭제
        truncate_all_tables()
        
        # 2-4. CSV 준비
        users_csv, users_count = prepare_users_csv()
        businesses_csv, businesses_count = prepare_businesses_csv()
        
        # 5. COPY 명령으로 삽입
        print("\n" + "=" * 80)
        print("5단계: PostgreSQL COPY 실행")
        print("=" * 80)
        
        # Users
        user_columns = ['yelp_user_id', 'username', 'email', 'hashed_password', 'created_at',
                       'review_count', 'useful', 'compliment', 'fans', 'average_stars',
                       'yelping_since_days']
        copy_to_database('users', users_csv, user_columns)
        
        # Businesses
        business_columns = ['business_id', 'name', 'categories', 'stars', 'review_count',
                           'address', 'city', 'state', 'latitude', 'longitude']
        copy_to_database('businesses', businesses_csv, business_columns)
        
        # Reviews (User와 Business가 삽입된 후)
        reviews_csv, reviews_count = prepare_reviews_csv()
        review_columns = ['user_id', 'business_id', 'stars', 'text', 'date', 'created_at', 'absa_features']
        copy_to_database('reviews', reviews_csv, review_columns)
        
        # 6. 검증
        final_users, final_businesses, final_reviews = verify_data()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("[SUCCESS] 초고속 마이그레이션 완료!")
        print("=" * 80)
        print(f"\n소요 시간: {elapsed:.1f}초 ({elapsed/60:.2f}분)")
        print(f"처리 속도: {(final_users + final_businesses + final_reviews) / elapsed:.0f} rows/sec")
        print("\n이제 프론트엔드에서 추천을 확인하세요!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

