"""
Yelp 데이터를 PostgreSQL로 마이그레이션
- User 데이터 (42k) + ABSA 피처
- Business 데이터 (14k) + ABSA 피처
- Review 데이터 (100k) + ABSA 피처
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from tqdm import tqdm
from sqlalchemy.orm import Session
from backend_web.database import SessionLocal, engine
from backend_web import models
from backend_web.auth import get_password_hash
import json

def convert_absa_to_json(row, absa_columns):
    """ABSA 피처를 JSON으로 변환"""
    absa_dict = {}
    for col in absa_columns:
        if col.startswith('absa_'):
            # absa_맛_긍정 -> 맛_긍정
            key = col.replace('absa_', '')
            value = float(row[col]) if not pd.isna(row[col]) else 0.0
            absa_dict[key] = value
    return absa_dict

def migrate_users():
    """User 데이터 마이그레이션"""
    print("=" * 80)
    print("User 데이터 마이그레이션")
    print("=" * 80)
    
    # 1. 원본 데이터 로딩
    print("\n[1/4] 데이터 로딩 중...")
    user_orig = pd.read_csv("data/processed/user_filtered.csv")
    user_preprocessed = pd.read_csv("data/processed/user_preprocessed.csv")
    
    print(f"  원본: {len(user_orig):,}명")
    print(f"  전처리: {len(user_preprocessed):,}명")
    
    # 2. 데이터 병합
    print("\n[2/4] 데이터 병합 중...")
    # user_preprocessed는 정규화된 값이므로, 원본 통계 사용
    merged = user_orig.merge(user_preprocessed[['user_id'] + [c for c in user_preprocessed.columns if c.startswith('absa_')]], 
                              on='user_id', how='left')
    
    print(f"  병합 완료: {len(merged):,}개")
    
    # 3. DB에 삽입
    print("\n[3/4] PostgreSQL에 삽입 중...")
    db = SessionLocal()
    
    inserted = 0
    errors = 0
    
    # ABSA 컬럼 추출
    absa_columns = [c for c in merged.columns if c.startswith('absa_')]
    
    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="  Users"):
        try:
            # ABSA 피처를 JSON으로 변환
            absa_json = convert_absa_to_json(row, absa_columns)
            
            # User 객체 생성
            user = models.User(
                yelp_user_id=row['user_id'],
                username=f"yelp_{row['user_id'][:8]}",  # 앞 8자만 사용
                email=f"yelp_{row['user_id'][:8]}@yelp.com",
                hashed_password=get_password_hash("yelp2024"),  # 임시 비밀번호
                review_count=int(row['review_count']) if not pd.isna(row['review_count']) else 0,
                useful=int(row['useful'] + row.get('funny', 0) + row.get('cool', 0)),
                compliment=int(sum([row.get(f'compliment_{t}', 0) for t in ['hot', 'more', 'profile', 'cute', 'list', 'note', 'plain', 'cool', 'funny', 'writer', 'photos']])),
                fans=int(row['fans']) if not pd.isna(row['fans']) else 0,
                average_stars=float(row['average_stars']) if not pd.isna(row['average_stars']) else 0.0,
                yelping_since_days=0,  # 계산 필요시 추가
                absa_features=absa_json
            )
            
            db.add(user)
            inserted += 1
            
            # 배치 커밋 (1000개마다)
            if inserted % 1000 == 0:
                db.commit()
                
        except Exception as e:
            errors += 1
            if errors < 5:  # 처음 5개만 출력
                print(f"\n  [ERROR] {row['user_id']}: {e}")
    
    # 최종 커밋
    db.commit()
    db.close()
    
    print(f"\n[4/4] 완료: {inserted:,}명 삽입, {errors}개 오류")
    return inserted, errors

def migrate_businesses():
    """Business 데이터 마이그레이션"""
    print("\n" + "=" * 80)
    print("Business 데이터 마이그레이션")
    print("=" * 80)
    
    # 1. 원본 데이터 로딩
    print("\n[1/4] 데이터 로딩 중...")
    business_orig = pd.read_csv("data/processed/business_filtered.csv")
    business_preprocessed = pd.read_csv("data/processed/business_preprocessed.csv")
    
    print(f"  원본: {len(business_orig):,}개")
    print(f"  전처리: {len(business_preprocessed):,}개")
    
    # 2. 데이터 병합
    print("\n[2/4] 데이터 병합 중...")
    merged = business_orig.merge(business_preprocessed[['business_id'] + [c for c in business_preprocessed.columns if c.startswith('absa_')]], 
                                  on='business_id', how='left')
    
    print(f"  병합 완료: {len(merged):,}개")
    
    # 3. DB에 삽입
    print("\n[3/4] PostgreSQL에 삽입 중...")
    db = SessionLocal()
    
    inserted = 0
    errors = 0
    
    # ABSA 컬럼 추출
    absa_columns = [c for c in merged.columns if c.startswith('absa_')]
    
    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="  Businesses"):
        try:
            # ABSA 피처를 JSON으로 변환
            absa_json = convert_absa_to_json(row, absa_columns)
            
            # Business 객체 생성
            business = models.Business(
                business_id=row['business_id'],
                name=row['name'] if not pd.isna(row['name']) else "Unknown",
                categories=row['categories'] if not pd.isna(row['categories']) else "",
                stars=float(row['stars']) if not pd.isna(row['stars']) else 0.0,
                review_count=int(row['review_count']) if not pd.isna(row['review_count']) else 0,
                address=row['address'] if not pd.isna(row['address']) else "",
                city=row['city'] if not pd.isna(row['city']) else "",
                state=row['state'] if not pd.isna(row['state']) else "",
                latitude=float(row['latitude']) if not pd.isna(row['latitude']) else 0.0,
                longitude=float(row['longitude']) if not pd.isna(row['longitude']) else 0.0,
                absa_features=absa_json
            )
            
            db.add(business)
            inserted += 1
            
            # 배치 커밋
            if inserted % 1000 == 0:
                db.commit()
                
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"\n  [ERROR] {row['business_id']}: {e}")
    
    # 최종 커밋
    db.commit()
    db.close()
    
    print(f"\n[4/4] 완료: {inserted:,}개 삽입, {errors}개 오류")
    return inserted, errors

def migrate_reviews():
    """Review 데이터 마이그레이션"""
    print("\n" + "=" * 80)
    print("Review 데이터 마이그레이션")
    print("=" * 80)
    
    # 1. 데이터 로딩
    print("\n[1/5] 데이터 로딩 중...")
    review_absa = pd.read_csv("data/processed/review_absa_features.csv")
    
    print(f"  리뷰: {len(review_absa):,}개")
    
    # 2. User/Business ID 매핑 (DB에서)
    print("\n[2/5] User/Business ID 매핑 중...")
    db = SessionLocal()
    
    # User 매핑
    user_map = {}
    users = db.query(models.User).all()
    for user in users:
        if user.yelp_user_id:
            user_map[user.yelp_user_id] = user.id
    print(f"  User 매핑: {len(user_map):,}개")
    
    # Business 매핑
    business_map = {}
    businesses = db.query(models.Business).all()
    for business in businesses:
        business_map[business.business_id] = business.id
    print(f"  Business 매핑: {len(business_map):,}개")
    
    # 3. 데이터 필터링 (매핑 가능한 것만)
    print("\n[3/5] 데이터 필터링 중...")
    review_absa['db_user_id'] = review_absa['user_id'].map(user_map)
    review_absa['db_business_id'] = review_absa['business_id'].map(business_map)
    
    # NaN 제거
    valid_reviews = review_absa.dropna(subset=['db_user_id', 'db_business_id'])
    print(f"  유효한 리뷰: {len(valid_reviews):,}개 / {len(review_absa):,}개")
    
    # 4. 원본 리뷰 텍스트 로딩
    print("\n[4/5] 원본 리뷰 텍스트 로딩 중...")
    review_orig = pd.read_csv("data/processed/review_100k_translated.csv")
    valid_reviews = valid_reviews.merge(
        review_orig[['review_id', 'text', 'date']],
        on='review_id',
        how='left'
    )
    
    # 5. DB에 삽입
    print("\n[5/5] PostgreSQL에 삽입 중...")
    
    inserted = 0
    errors = 0
    
    # ABSA 컬럼 추출
    absa_columns = [c for c in valid_reviews.columns if c.startswith('absa_')]
    
    for idx, row in tqdm(valid_reviews.iterrows(), total=len(valid_reviews), desc="  Reviews"):
        try:
            # ABSA 피처를 JSON으로 변환
            absa_json = convert_absa_to_json(row, absa_columns)
            
            # Review 객체 생성
            review = models.Review(
                user_id=int(row['db_user_id']),
                business_id=int(row['db_business_id']),
                stars=float(row['stars']),
                text=row['text'] if not pd.isna(row['text']) else "",
                date=pd.to_datetime(row['date']) if not pd.isna(row['date']) else None,
                absa_features=absa_json
            )
            
            db.add(review)
            inserted += 1
            
            # 배치 커밋
            if inserted % 1000 == 0:
                db.commit()
                
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"\n  [ERROR] {row['review_id']}: {e}")
    
    # 최종 커밋
    db.commit()
    db.close()
    
    print(f"\n완료: {inserted:,}개 삽입, {errors}개 오류")
    return inserted, errors

def verify_migration():
    """마이그레이션 결과 확인"""
    print("\n" + "=" * 80)
    print("마이그레이션 결과 확인")
    print("=" * 80)
    
    db = SessionLocal()
    
    user_count = db.query(models.User).count()
    business_count = db.query(models.Business).count()
    review_count = db.query(models.Review).count()
    
    print(f"\n✓ Users: {user_count:,}명")
    print(f"✓ Businesses: {business_count:,}개")
    print(f"✓ Reviews: {review_count:,}개")
    
    # 샘플 데이터 확인
    print("\n샘플 데이터:")
    sample_user = db.query(models.User).first()
    if sample_user and sample_user.absa_features:
        print(f"  User ABSA 키: {list(sample_user.absa_features.keys())[:5]} ...")
    
    sample_business = db.query(models.Business).first()
    if sample_business and sample_business.absa_features:
        print(f"  Business ABSA 키: {list(sample_business.absa_features.keys())[:5]} ...")
    
    db.close()

def main():
    print("\n" + "=" * 80)
    print("Yelp 데이터 마이그레이션 시작")
    print("=" * 80)
    
    try:
        # 1. User 마이그레이션
        user_inserted, user_errors = migrate_users()
        
        # 2. Business 마이그레이션
        business_inserted, business_errors = migrate_businesses()
        
        # 3. Review 마이그레이션
        review_inserted, review_errors = migrate_reviews()
        
        # 4. 결과 확인
        verify_migration()
        
        print("\n" + "=" * 80)
        print("[SUCCESS] 마이그레이션 완료!")
        print(f"\n요약:")
        print(f"  Users: {user_inserted:,}명 ({user_errors}개 오류)")
        print(f"  Businesses: {business_inserted:,}개 ({business_errors}개 오류)")
        print(f"  Reviews: {review_inserted:,}개 ({review_errors}개 오류)")
        print("\n다음 단계:")
        print("  - backend_model에 예측 API 추가")
        print("  - frontend UI 업데이트")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

