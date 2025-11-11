"""
Users 마이그레이션 (하나씩 커밋, 중복 skip)
"""

import sys
sys.path.append('.')

import pandas as pd
from tqdm import tqdm
from backend_web.database import SessionLocal
from backend_web import models
from backend_web.auth import get_password_hash
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
import time

def migrate_users():
    """User 데이터 마이그레이션"""
    print("=" * 80)
    print("Users 마이그레이션")
    print("=" * 80)
    
    # 데이터 로딩
    print("\n[1/2] 데이터 로딩...")
    user_df = pd.read_csv('data/filtered/user_filtered_20_3.csv')
    print(f"  {len(user_df):,}명")
    
    # ABSA 컬럼 추출
    absa_columns = [c for c in user_df.columns if c.startswith('absa_')]
    
    # 마이그레이션
    print(f"\n[2/2] DB 삽입 중 (하나씩 커밋, 중복 skip)...")
    session = SessionLocal()
    inserted = 0
    skipped = 0
    
    start_time = time.time()
    
    try:
        for idx, row in tqdm(user_df.iterrows(), total=len(user_df), desc="  Users"):
            try:
                # ABSA JSON 생성
                absa_dict = {}
                for col in absa_columns:
                    key = col.replace('absa_', '')
                    value = float(row[col]) if not pd.isna(row[col]) else 0.0
                    absa_dict[key] = value
                
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
                    yelping_since_days=0,
                    absa_features=absa_dict
                )
                
                session.add(user)
                session.commit()  # 하나씩 커밋
                inserted += 1
                
            except IntegrityError:
                session.rollback()  # 중복이면 롤백
                skipped += 1
                continue
        
        elapsed = time.time() - start_time
        
        print(f"\n[OK] Users 완료!")
        print(f"  삽입: {inserted:,}명")
        print(f"  중복 skip: {skipped:,}명")
        print(f"  소요 시간: {elapsed:.1f}초 ({elapsed/60:.2f}분)")
        
        return inserted
        
    except Exception as e:
        print(f"\n[ERROR] Users 실패: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    migrate_users()

