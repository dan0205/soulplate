"""
Reviews 마이그레이션 (하나씩 커밋, 중복 skip)
"""

import sys
sys.path.append('.')

import pandas as pd
from tqdm import tqdm
from backend_web.database import SessionLocal
from backend_web import models
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
import time

def migrate_reviews():
    """Review 데이터 마이그레이션"""
    print("=" * 80)
    print("Reviews 마이그레이션")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/4] 데이터 로딩...")
    review_df = pd.read_csv('data/filtered/review_filtered_20_3.csv')
    print(f"  필터링된 리뷰: {len(review_df):,}개")
    
    # 원본 리뷰 텍스트 로딩 및 매칭
    print("\n  원본 리뷰 텍스트 로딩...")
    review_full = pd.read_csv('data/processed/review_100k_translated.csv', usecols=['review_id', 'text', 'date'])
    review_df = review_df.merge(review_full, on='review_id', how='left')
    print(f"  텍스트 매칭: {review_df['text'].notna().sum():,}개")
    
    # User/Business ID 매핑
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
    
    # 마이그레이션
    print(f"\n[3/4] DB 삽입 중 (하나씩 커밋, 중복 skip)...")
    session = SessionLocal()
    inserted = 0
    skipped = 0
    
    start_time = time.time()
    
    try:
        for idx, row in tqdm(review_df.iterrows(), total=len(review_df), desc="  Reviews"):
            try:
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
                
                # 텍스트 가져오기 (없으면 빈 문자열)
                review_text = str(row['text'])[:5000] if pd.notna(row.get('text')) else ""
                
                review = models.Review(
                    user_id=user_id,
                    business_id=business_id,
                    stars=float(row['stars']),
                    text=review_text,
                    date=review_date,
                    created_at=datetime.now(timezone.utc),
                    absa_features=absa_dict
                )
                
                session.add(review)
                session.commit()  # 하나씩 커밋
                inserted += 1
                
            except IntegrityError:
                session.rollback()
                skipped += 1
                continue
        
        elapsed = time.time() - start_time
        
        print(f"\n[4/4] Reviews 완료!")
        print(f"  삽입: {inserted:,}개")
        print(f"  스킵: {skipped:,}개")
        print(f"  소요 시간: {elapsed:.1f}초 ({elapsed/60:.2f}분)")
        
        return inserted
        
    except Exception as e:
        print(f"\n[ERROR] Reviews 실패: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    migrate_reviews()

