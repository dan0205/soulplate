"""
Businesses 마이그레이션 (하나씩 커밋, 중복 skip)
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

def migrate_businesses():
    """Business 데이터 마이그레이션"""
    print("=" * 80)
    print("Businesses 마이그레이션")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/2] 데이터 로딩...")
    business_df = pd.read_csv('data/filtered/business_filtered_20_3.csv')
    print(f"  {len(business_df):,}개")
    
    # ABSA 컬럼 추출
    absa_columns = [c for c in business_df.columns if c.startswith('absa_')]
    
    # 마이그레이션
    print(f"\n[2/2] DB 삽입 중 (하나씩 커밋, 중복 skip)...")
    session = SessionLocal()
    inserted = 0
    skipped = 0
    
    start_time = time.time()
    
    try:
        for idx, row in tqdm(business_df.iterrows(), total=len(business_df), desc="  Businesses"):
            try:
                # ABSA JSON 생성
                absa_dict = {}
                for col in absa_columns:
                    key = col.replace('absa_', '')
                    value = float(row[col]) if not pd.isna(row[col]) else 0.0
                    absa_dict[key] = value
                
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
                    absa_features=absa_dict
                )
                
                session.add(business)
                session.commit()  # 하나씩 커밋
                inserted += 1
                
            except IntegrityError:
                session.rollback()  # 중복이면 롤백
                skipped += 1
                continue
        
        elapsed = time.time() - start_time
        
        print(f"\n[OK] Businesses 완료!")
        print(f"  삽입: {inserted:,}개")
        print(f"  중복 skip: {skipped:,}개")
        print(f"  소요 시간: {elapsed:.1f}초 ({elapsed/60:.2f}분)")
        
        return inserted
        
    except Exception as e:
        print(f"\n[ERROR] Businesses 실패: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    migrate_businesses()

