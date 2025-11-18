"""
한글 리뷰를 PostgreSQL DB에 마이그레이션
- review_filtered_20_3_korean.csv → reviews 테이블
- 기존 리뷰 삭제 후 새로 입력
"""

import sys
sys.path.append('.')

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend_web.database import SessionLocal, engine
from backend_web import models
from datetime import datetime, timezone
from tqdm import tqdm

def clear_reviews(session: Session):
    """기존 리뷰 모두 삭제"""
    print("\n[준비] 기존 리뷰 삭제 중...")
    
    try:
        count = session.query(models.Review).count()
        print(f"  기존 리뷰: {count:,}개")
        
        if count > 0:
            session.query(models.Review).delete()
            session.commit()
            print(f"  [OK] {count:,}개 리뷰 삭제 완료")
        else:
            print(f"  [OK] 삭제할 리뷰 없음")
    except Exception as e:
        print(f"  [ERROR] 삭제 실패: {e}")
        session.rollback()
        raise

def create_mapping_tables(session: Session):
    """User/Business Yelp ID → DB ID 매핑 테이블 생성"""
    print("\n[1/4] 매핑 테이블 생성 중...")
    
    # User 매핑
    users = session.query(models.User.id, models.User.yelp_user_id).all()
    user_map = {yelp_id: db_id for db_id, yelp_id in users if yelp_id}
    print(f"  User 매핑: {len(user_map):,}개")
    
    # Business 매핑
    businesses = session.query(models.Business.id, models.Business.business_id).all()
    business_map = {yelp_id: db_id for db_id, yelp_id in businesses if yelp_id}
    print(f"  Business 매핑: {len(business_map):,}개")
    
    return user_map, business_map

def load_reviews():
    """한글 리뷰 CSV 로딩"""
    print("\n[2/4] CSV 파일 로딩 중...")
    
    df = pd.read_csv("data/filtered/review_filtered_20_3_korean.csv", encoding='utf-8-sig')
    print(f"  [OK] {len(df):,}개 리뷰 로드")
    
    # ABSA 컬럼 확인
    absa_columns = [col for col in df.columns if col.startswith('absa_')]
    print(f"  ABSA 피처: {len(absa_columns)}개")
    
    return df, absa_columns

def migrate_reviews(session: Session, df: pd.DataFrame, absa_columns: list, 
                   user_map: dict, business_map: dict):
    """리뷰를 DB에 삽입"""
    print("\n[3/4] 리뷰 DB 삽입 중...")
    
    inserted = 0
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  진행"):
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
            
            # 텍스트 가져오기
            review_text = str(row['text']) if pd.notna(row.get('text')) else ""
            if len(review_text) > 5000:
                review_text = review_text[:5000]
            
            # Review 객체 생성
            review = models.Review(
                user_id=user_id,
                business_id=business_id,
                stars=float(row['stars']),
                text=review_text,
                date=None,  # 날짜 정보가 없으면 None
                created_at=datetime.now(timezone.utc),
                absa_features=absa_dict
            )
            
            session.add(review)
            inserted += 1
            
            # 500개마다 커밋
            if inserted % 500 == 0:
                session.commit()
                session.expunge_all()
                
        except Exception as e:
            print(f"\n  [ERROR] {idx}번째 리뷰 실패: {e}")
            session.rollback()
            skipped += 1
            continue
    
    # 마지막 커밋
    session.commit()
    
    print(f"\n  [OK] 삽입 완료")
    print(f"    성공: {inserted:,}개")
    print(f"    스킵: {skipped:,}개")
    
    return inserted, skipped

def verify_data(session: Session):
    """데이터 검증"""
    print("\n[4/4] 데이터 검증 중...")
    
    review_count = session.query(models.Review).count()
    print(f"  총 리뷰: {review_count:,}개")
    
    # 샘플 리뷰 확인
    sample_review = session.query(models.Review).first()
    if sample_review:
        print(f"\n  샘플 리뷰:")
        print(f"    별점: {sample_review.stars}")
        print(f"    텍스트: {sample_review.text[:100]}...")
        print(f"    ABSA 피처: {len(sample_review.absa_features)}개")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("한글 리뷰 DB 마이그레이션")
    print("=" * 80)
    
    session = SessionLocal()
    
    try:
        # 0. 기존 리뷰 삭제
        clear_reviews(session)
        
        # 1. 매핑 테이블 생성
        user_map, business_map = create_mapping_tables(session)
        
        # 2. CSV 로딩
        df, absa_columns = load_reviews()
        
        # 3. 리뷰 삽입
        inserted, skipped = migrate_reviews(session, df, absa_columns, user_map, business_map)
        
        # 4. 검증
        verify_data(session)
        
        print("\n" + "=" * 80)
        print("[SUCCESS] 마이그레이션 완료!")
        print(f"\n총 {inserted:,}개 리뷰가 한글 텍스트로 업데이트되었습니다.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    main()












