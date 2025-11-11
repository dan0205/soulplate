"""
단순 Insert 마이그레이션
- session.add() 방식으로 하나씩 삽입
- 가장 안정적인 방법
- ABSA 제외
"""

import sys
sys.path.append('.')
sys.stdout.flush()  # 출력 버퍼 즉시 플러시

import pandas as pd
import numpy as np
from backend_web.database import SessionLocal
from backend_web import models
from backend_web.auth import get_password_hash
from datetime import datetime, timezone
import time

def clear_all_data():
    """기존 데이터 삭제"""
    print("=" * 80)
    print("1단계: 기존 데이터 삭제")
    print("=" * 80)
    
    session = SessionLocal()
    try:
        # Foreign key 순서 고려
        review_count = session.query(models.Review).count()
        if review_count > 0:
            print(f"  Reviews 삭제 중... ({review_count:,}개)")
            session.query(models.Review).delete()
            session.commit()
        
        user_absa_count = session.query(models.UserABSAFeatures).count()
        if user_absa_count > 0:
            print(f"  UserABSAFeatures 삭제 중... ({user_absa_count:,}개)")
            session.query(models.UserABSAFeatures).delete()
            session.commit()
        
        user_count = session.query(models.User).count()
        if user_count > 0:
            print(f"  Users 삭제 중... ({user_count:,}개)")
            session.query(models.User).delete()
            session.commit()
        
        business_absa_count = session.query(models.BusinessABSAFeatures).count()
        if business_absa_count > 0:
            print(f"  BusinessABSAFeatures 삭제 중... ({business_absa_count:,}개)")
            session.query(models.BusinessABSAFeatures).delete()
            session.commit()
        
        business_count = session.query(models.Business).count()
        if business_count > 0:
            print(f"  Businesses 삭제 중... ({business_count:,}개)")
            session.query(models.Business).delete()
            session.commit()
        
        print("[OK] 기존 데이터 삭제 완료\n")
        
    except Exception as e:
        print(f"[ERROR] 삭제 실패: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def insert_users():
    """User 데이터 삽입 (하나씩)"""
    print("=" * 80)
    print("2단계: Users 데이터 삽입")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/2] 데이터 로딩...")
    user_df = pd.read_csv("data/processed/user_filtered.csv")
    print(f"  {len(user_df):,}명")
    
    # 삽입
    print(f"\n[2/2] DB 삽입 중...")
    session = SessionLocal()
    inserted = 0
    
    try:
        for idx, row in user_df.iterrows():
            user = models.User(
                yelp_user_id=row['user_id'],
                username=f"yelp_{row['user_id'][:8]}",
                email=f"yelp_{row['user_id'][:8]}@yelp.com",
                hashed_password=get_password_hash("yelp2024"),
                created_at=datetime.now(timezone.utc),
                review_count=int(row['review_count']) if not pd.isna(row['review_count']) else 0,
                useful=int(row['useful'] + row.get('funny', 0) + row.get('cool', 0)) if not pd.isna(row['useful']) else 0,
                compliment=int(sum([row.get(f'compliment_{t}', 0) for t in ['hot', 'more', 'profile', 'cute', 'list', 'note', 'plain', 'cool', 'funny', 'writer', 'photos']])),
                fans=int(row['fans']) if not pd.isna(row['fans']) else 0,
                average_stars=float(row['average_stars']) if not pd.isna(row['average_stars']) else 0.0,
                yelping_since_days=0
            )
            
            session.add(user)
            inserted += 1
            
            # 1000개마다 커밋
            if inserted % 1000 == 0:
                session.commit()
                print(f"  진행: {inserted:,} / {len(user_df):,}", flush=True)
        
        # 마지막 커밋
        session.commit()
        print(f"[OK] Users 삽입 완료: {inserted:,}명\n")
        return inserted
        
    except Exception as e:
        print(f"[ERROR] Users 삽입 실패: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def insert_businesses():
    """Business 데이터 삽입 (하나씩)"""
    print("=" * 80)
    print("3단계: Businesses 데이터 삽입")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/2] 데이터 로딩...")
    business_df = pd.read_csv("data/processed/business_filtered.csv")
    print(f"  {len(business_df):,}개")
    
    # 삽입
    print(f"\n[2/2] DB 삽입 중...")
    session = SessionLocal()
    inserted = 0
    
    try:
        for idx, row in business_df.iterrows():
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
                longitude=float(row['longitude']) if not pd.isna(row['longitude']) else 0.0
            )
            
            session.add(business)
            inserted += 1
            
            # 1000개마다 커밋
            if inserted % 1000 == 0:
                session.commit()
                print(f"  진행: {inserted:,} / {len(business_df):,}")
        
        # 마지막 커밋
        session.commit()
        print(f"[OK] Businesses 삽입 완료: {inserted:,}개\n")
        return inserted
        
    except Exception as e:
        print(f"[ERROR] Businesses 삽입 실패: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def insert_reviews():
    """Review 데이터 삽입 (하나씩)"""
    print("=" * 80)
    print("4단계: Reviews 데이터 삽입")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/4] 데이터 로딩...")
    review_df = pd.read_csv("data/processed/review_absa_features.csv")
    print(f"  {len(review_df):,}개")
    
    # User/Business 매핑
    print("\n[2/4] User/Business ID 매핑...")
    session = SessionLocal()
    
    users = session.query(models.User.id, models.User.yelp_user_id).all()
    user_map = {yelp_id: db_id for db_id, yelp_id in users}
    print(f"  User: {len(user_map):,}개")
    
    businesses = session.query(models.Business.id, models.Business.business_id).all()
    business_map = {biz_id: db_id for db_id, biz_id in businesses}
    print(f"  Business: {len(business_map):,}개")
    
    session.close()
    
    # ABSA 컬럼 추출
    absa_columns = [c for c in review_df.columns if c.startswith('absa_')]
    
    # 삽입
    print(f"\n[3/4] DB 삽입 중...")
    session = SessionLocal()
    inserted = 0
    skipped = 0
    
    try:
        for idx, row in review_df.iterrows():
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
            
            review = models.Review(
                user_id=user_id,
                business_id=business_id,
                stars=float(row['stars']),
                text=str(row['text'])[:5000],  # 길이 제한
                date=review_date,
                created_at=datetime.now(timezone.utc),
                absa_features=absa_dict
            )
            
            session.add(review)
            inserted += 1
            
            # 500개마다 커밋 (Review는 데이터가 크므로 더 자주)
            if inserted % 500 == 0:
                session.commit()
                session.expunge_all()  # 메모리 정리
                print(f"  진행: {inserted:,} / {len(review_df):,} (스킵: {skipped:,})")
        
        # 마지막 커밋
        session.commit()
        print(f"[OK] Reviews 삽입 완료: {inserted:,}개 (스킵: {skipped:,}개)\n")
        return inserted
        
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
    print("=" * 80)
    print("5단계: 데이터 검증")
    print("=" * 80)
    
    session = SessionLocal()
    
    user_count = session.query(models.User).count()
    business_count = session.query(models.Business).count()
    review_count = session.query(models.Review).count()
    
    print(f"\n[최종 결과]")
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
        print(f"  text: {sample_review.text[:80]}...")
        if sample_review.absa_features:
            print(f"  ABSA keys: {list(sample_review.absa_features.keys())[:5]}...")
    
    session.close()
    
    return user_count, business_count, review_count


def main():
    """메인 실행"""
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("단순 Insert 마이그레이션 (ABSA 제외)")
    print("=" * 80)
    print("방식: session.add() - 하나씩 삽입")
    print("=" * 80)
    
    try:
        # 1. 기존 데이터 삭제
        clear_all_data()
        
        # 2. Users
        user_count = insert_users()
        
        # 3. Businesses
        business_count = insert_businesses()
        
        # 4. Reviews
        review_count = insert_reviews()
        
        # 5. 검증
        final_users, final_businesses, final_reviews = verify_data()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("[SUCCESS] 마이그레이션 완료!")
        print("=" * 80)
        print(f"\n소요 시간: {elapsed:.1f}초 ({elapsed/60:.2f}분)")
        print(f"처리 속도: {(final_users + final_businesses + final_reviews) / elapsed:.0f} rows/sec")
        print("\n다음 단계:")
        print("  python scripts/aggregate_absa_features.py  # ABSA 집계")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

