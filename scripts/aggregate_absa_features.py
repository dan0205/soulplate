"""
ABSA 집계 스크립트
- Review 테이블에서 User/Business별 ABSA 평균 계산
- UserABSAFeatures, BusinessABSAFeatures 테이블에 삽입
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sqlalchemy import text
from backend_web.database import SessionLocal, engine
from backend_web import models
import json
import time

def aggregate_user_absa():
    """User별 ABSA 평균 집계"""
    print("=" * 80)
    print("1단계: User ABSA 집계")
    print("=" * 80)
    
    session = SessionLocal()
    try:
        # 1. 모든 리뷰 데이터 가져오기
        print("\n[1/4] 리뷰 데이터 로딩...")
        reviews = session.query(models.Review).all()
        print(f"  총 {len(reviews):,}개 리뷰")
        
        # 2. User별로 그룹화하여 평균 계산
        print("\n[2/4] User별 ABSA 평균 계산...")
        user_absa_dict = {}
        
        for review in reviews:
            if review.absa_features is None:
                continue
                
            user_id = review.user_id
            if user_id not in user_absa_dict:
                user_absa_dict[user_id] = []
            
            user_absa_dict[user_id].append(review.absa_features)
        
        print(f"  ABSA가 있는 User: {len(user_absa_dict):,}명")
        
        # 3. 평균 계산
        print("\n[3/4] 평균값 계산...")
        user_absa_aggregated = []
        
        for user_id, absa_list in user_absa_dict.items():
            # 모든 ABSA 특징의 평균 계산
            avg_absa = {}
            
            # 첫 번째 리뷰에서 키 목록 가져오기
            keys = absa_list[0].keys()
            
            for key in keys:
                values = [absa[key] for absa in absa_list if key in absa and absa[key] is not None]
                if values:
                    avg_absa[key] = float(np.mean(values))
                else:
                    avg_absa[key] = 0.0
            
            user_absa_aggregated.append({
                'user_id': user_id,
                'absa_features': avg_absa,
                'updated_at': datetime.now(timezone.utc)
            })
        
        # 4. UserABSAFeatures 테이블에 삽입
        print(f"\n[4/4] UserABSAFeatures 테이블에 삽입 중... ({len(user_absa_aggregated):,}개)")
        
        # 기존 데이터 삭제
        session.query(models.UserABSAFeatures).delete()
        session.commit()
        
        # Batch 삽입
        batch_size = 5000
        for i in range(0, len(user_absa_aggregated), batch_size):
            batch = user_absa_aggregated[i:i+batch_size]
            session.bulk_insert_mappings(models.UserABSAFeatures, batch)
            session.commit()
            print(f"  진행: {min(i+batch_size, len(user_absa_aggregated)):,} / {len(user_absa_aggregated):,}")
        
        print(f"[OK] User ABSA 집계 완료: {len(user_absa_aggregated):,}명")
        return len(user_absa_aggregated)
        
    except Exception as e:
        print(f"[ERROR] User ABSA 집계 실패: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        raise
    finally:
        session.close()


def aggregate_business_absa():
    """Business별 ABSA 평균 집계"""
    print("\n" + "=" * 80)
    print("2단계: Business ABSA 집계")
    print("=" * 80)
    
    session = SessionLocal()
    try:
        # 1. 모든 리뷰 데이터 가져오기
        print("\n[1/4] 리뷰 데이터 로딩...")
        reviews = session.query(models.Review).all()
        print(f"  총 {len(reviews):,}개 리뷰")
        
        # 2. Business별로 그룹화하여 평균 계산
        print("\n[2/4] Business별 ABSA 평균 계산...")
        business_absa_dict = {}
        
        for review in reviews:
            if review.absa_features is None:
                continue
                
            business_id = review.business_id
            if business_id not in business_absa_dict:
                business_absa_dict[business_id] = []
            
            business_absa_dict[business_id].append(review.absa_features)
        
        print(f"  ABSA가 있는 Business: {len(business_absa_dict):,}개")
        
        # 3. 평균 계산
        print("\n[3/4] 평균값 계산...")
        business_absa_aggregated = []
        
        for business_id, absa_list in business_absa_dict.items():
            # 모든 ABSA 특징의 평균 계산
            avg_absa = {}
            
            # 첫 번째 리뷰에서 키 목록 가져오기
            keys = absa_list[0].keys()
            
            for key in keys:
                values = [absa[key] for absa in absa_list if key in absa and absa[key] is not None]
                if values:
                    avg_absa[key] = float(np.mean(values))
                else:
                    avg_absa[key] = 0.0
            
            business_absa_aggregated.append({
                'business_id': business_id,
                'absa_features': avg_absa,
                'updated_at': datetime.now(timezone.utc)
            })
        
        # 4. BusinessABSAFeatures 테이블에 삽입
        print(f"\n[4/4] BusinessABSAFeatures 테이블에 삽입 중... ({len(business_absa_aggregated):,}개)")
        
        # 기존 데이터 삭제
        session.query(models.BusinessABSAFeatures).delete()
        session.commit()
        
        # Batch 삽입
        batch_size = 5000
        for i in range(0, len(business_absa_aggregated), batch_size):
            batch = business_absa_aggregated[i:i+batch_size]
            session.bulk_insert_mappings(models.BusinessABSAFeatures, batch)
            session.commit()
            print(f"  진행: {min(i+batch_size, len(business_absa_aggregated)):,} / {len(business_absa_aggregated):,}")
        
        print(f"[OK] Business ABSA 집계 완료: {len(business_absa_aggregated):,}개")
        return len(business_absa_aggregated)
        
    except Exception as e:
        print(f"[ERROR] Business ABSA 집계 실패: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
        raise
    finally:
        session.close()


def verify_aggregation():
    """집계 결과 검증"""
    print("\n" + "=" * 80)
    print("3단계: 집계 결과 검증")
    print("=" * 80)
    
    session = SessionLocal()
    
    user_absa_count = session.query(models.UserABSAFeatures).count()
    business_absa_count = session.query(models.BusinessABSAFeatures).count()
    
    print(f"\n[결과]")
    print(f"  UserABSAFeatures: {user_absa_count:,}개")
    print(f"  BusinessABSAFeatures: {business_absa_count:,}개")
    
    # 샘플 확인
    sample_user_absa = session.query(models.UserABSAFeatures).first()
    if sample_user_absa:
        print(f"\n[샘플 UserABSAFeatures]")
        print(f"  user_id: {sample_user_absa.user_id}")
        print(f"  ABSA keys: {list(sample_user_absa.absa_features.keys())[:5]}...")
        print(f"  updated_at: {sample_user_absa.updated_at}")
    
    sample_business_absa = session.query(models.BusinessABSAFeatures).first()
    if sample_business_absa:
        print(f"\n[샘플 BusinessABSAFeatures]")
        print(f"  business_id: {sample_business_absa.business_id}")
        print(f"  ABSA keys: {list(sample_business_absa.absa_features.keys())[:5]}...")
        print(f"  updated_at: {sample_business_absa.updated_at}")
    
    session.close()
    
    return user_absa_count, business_absa_count


def main():
    """메인 실행"""
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("ABSA 집계 스크립트 시작")
    print("=" * 80)
    
    try:
        # 1. User ABSA 집계
        user_count = aggregate_user_absa()
        
        # 2. Business ABSA 집계
        business_count = aggregate_business_absa()
        
        # 3. 검증
        verify_aggregation()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("[SUCCESS] ABSA 집계 완료!")
        print("=" * 80)
        print(f"\n소요 시간: {elapsed:.1f}초 ({elapsed/60:.2f}분)")
        print(f"  User ABSA: {user_count:,}개")
        print(f"  Business ABSA: {business_count:,}개")
        print("\n이제 API에서 ABSA 특징을 조회할 수 있습니다!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] 집계 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

