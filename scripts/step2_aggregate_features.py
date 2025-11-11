"""
Step 2: User/Business 피처 집계 및 전처리
- User별 ABSA 피처 평균 계산
- Business별 ABSA 피처 평균 계산
- 기본 피처 전처리 (통합, 변환, 정규화)
- 전처리된 데이터 CSV 저장

출력:
- data/processed/user_aggregated.csv
- data/processed/business_aggregated.csv
- data/processed/user_preprocessed.csv
- data/processed/business_preprocessed.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pickle
import os

def aggregate_absa_features():
    """ABSA 피처를 User/Business 단위로 집계"""
    print("=" * 80)
    print("Step 2: User/Business 피처 집계 및 전처리")
    print("=" * 80)
    
    # 1. ABSA 피처 로딩
    print("\n[1/6] ABSA 피처 로딩 중...")
    absa_df = pd.read_csv("data/processed/review_absa_features.csv")
    print(f"  [OK] {len(absa_df):,}개 리뷰 로딩")
    
    absa_columns = [col for col in absa_df.columns if col.startswith('absa_')]
    print(f"  ABSA 피처: {len(absa_columns)}개")
    
    # 2. User별 ABSA 평균 계산
    print("\n[2/6] User별 ABSA 피처 집계 중...")
    user_absa = absa_df.groupby('user_id')[absa_columns].mean().reset_index()
    print(f"  [OK] {len(user_absa):,}명의 User 집계 완료")
    
    # 저장
    user_absa.to_csv("data/processed/user_aggregated.csv", index=False, encoding='utf-8-sig')
    print(f"  저장: data/processed/user_aggregated.csv")
    
    # 3. Business별 ABSA 평균 계산
    print("\n[3/6] Business별 ABSA 피처 집계 중...")
    business_absa = absa_df.groupby('business_id')[absa_columns].mean().reset_index()
    print(f"  [OK] {len(business_absa):,}개의 Business 집계 완료")
    
    # 저장
    business_absa.to_csv("data/processed/business_aggregated.csv", index=False, encoding='utf-8-sig')
    print(f"  저장: data/processed/business_aggregated.csv")
    
    return user_absa, business_absa

def preprocess_user_features(user_absa):
    """User 기본 피처 전처리"""
    print("\n[4/6] User 피처 전처리 중...")
    
    # User 기본 데이터 로딩
    user_df = pd.read_csv("data/processed/user_filtered.csv")
    print(f"  원본 User 데이터: {user_df.shape}")
    
    # ABSA 피처와 병합
    user_df = user_df.merge(user_absa, on='user_id', how='left')
    
    # 1. useful 통합 (useful + funny + cool)
    user_df['useful'] = user_df['useful'] + user_df['funny'] + user_df['cool']
    
    # 2. compliment 통합 (11개 합산)
    compliment_cols = [col for col in user_df.columns if col.startswith('compliment_')]
    user_df['compliment'] = user_df[compliment_cols].sum(axis=1)
    
    # 3. yelping_since → yelping_since_days (가입 경과일)
    current_date = datetime.now()
    user_df['yelping_since_days'] = (
        current_date - pd.to_datetime(user_df['yelping_since'])
    ).dt.days
    
    # 4. 사용할 피처 선택
    feature_cols = [
        'review_count', 'useful', 'compliment', 'fans', 
        'average_stars', 'yelping_since_days'
    ]
    
    # ABSA 피처 추가
    absa_columns = [col for col in user_df.columns if col.startswith('absa_')]
    feature_cols.extend(absa_columns)
    
    # 5. 결측치 처리
    user_features = user_df[['user_id'] + feature_cols].copy()
    user_features[feature_cols] = user_features[feature_cols].fillna(0)
    
    # 6. Log Transform (편향된 분포)
    log_transform_cols = ['review_count', 'fans', 'compliment']
    for col in log_transform_cols:
        if col in user_features.columns:
            user_features[col] = np.log1p(user_features[col])  # log(1+x)
    
    # 7. Standard Scaling
    scaler = StandardScaler()
    user_features[feature_cols] = scaler.fit_transform(user_features[feature_cols])
    
    # Scaler 저장
    os.makedirs("models", exist_ok=True)
    with open("models/user_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"  [OK] User 피처 전처리 완료: {user_features.shape}")
    print(f"  피처: {len(feature_cols)}개 (기본 6개 + ABSA {len(absa_columns)}개)")
    
    # 저장
    user_features.to_csv("data/processed/user_preprocessed.csv", index=False, encoding='utf-8-sig')
    print(f"  저장: data/processed/user_preprocessed.csv")
    
    return user_features

def preprocess_business_features(business_absa):
    """Business 기본 피처 전처리"""
    print("\n[5/6] Business 피처 전처리 중...")
    
    # Business 기본 데이터 로딩
    business_df = pd.read_csv("data/processed/business_filtered.csv")
    print(f"  원본 Business 데이터: {business_df.shape}")
    
    # ABSA 피처와 병합
    business_df = business_df.merge(business_absa, on='business_id', how='left')
    
    # 1. 사용할 피처 선택
    feature_cols = [
        'stars', 'review_count', 'latitude', 'longitude'
    ]
    
    # ABSA 피처 추가
    absa_columns = [col for col in business_df.columns if col.startswith('absa_')]
    feature_cols.extend(absa_columns)
    
    # 2. 결측치 처리
    business_features = business_df[['business_id'] + feature_cols].copy()
    business_features[feature_cols] = business_features[feature_cols].fillna(0)
    
    # 3. Log Transform (편향된 분포)
    log_transform_cols = ['review_count']
    for col in log_transform_cols:
        if col in business_features.columns:
            business_features[col] = np.log1p(business_features[col])
    
    # 4. Standard Scaling
    scaler = StandardScaler()
    business_features[feature_cols] = scaler.fit_transform(business_features[feature_cols])
    
    # Scaler 저장
    with open("models/business_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"  [OK] Business 피처 전처리 완료: {business_features.shape}")
    print(f"  피처: {len(feature_cols)}개 (기본 4개 + ABSA {len(absa_columns)}개)")
    
    # 저장
    business_features.to_csv("data/processed/business_preprocessed.csv", index=False, encoding='utf-8-sig')
    print(f"  저장: data/processed/business_preprocessed.csv")
    
    return business_features

def verify_features():
    """생성된 피처 검증"""
    print("\n[6/6] 피처 검증 중...")
    
    user_df = pd.read_csv("data/processed/user_preprocessed.csv")
    business_df = pd.read_csv("data/processed/business_preprocessed.csv")
    
    print(f"\n  User 피처:")
    print(f"    Shape: {user_df.shape}")
    print(f"    컬럼: {list(user_df.columns[:7])} + {user_df.shape[1]-7}개 ABSA")
    print(f"    통계:")
    print(user_df.describe().loc[['mean', 'std']].T.head(6))
    
    print(f"\n  Business 피처:")
    print(f"    Shape: {business_df.shape}")
    print(f"    컬럼: {list(business_df.columns[:5])} + {business_df.shape[1]-5}개 ABSA")
    print(f"    통계:")
    print(business_df.describe().loc[['mean', 'std']].T.head(5))
    
    print(f"\n  [OK] 모든 피처가 정규화됨 (평균~0, 표준편차~1)")

def main():
    # 1. ABSA 피처 집계
    user_absa, business_absa = aggregate_absa_features()
    
    # 2. User 피처 전처리
    user_features = preprocess_user_features(user_absa)
    
    # 3. Business 피처 전처리
    business_features = preprocess_business_features(business_absa)
    
    # 4. 검증
    verify_features()
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Step 2 완료!")
    print("\n생성된 파일:")
    print("  - data/processed/user_aggregated.csv")
    print("  - data/processed/business_aggregated.csv")
    print("  - data/processed/user_preprocessed.csv")
    print("  - data/processed/business_preprocessed.csv")
    print("  - models/user_scaler.pkl")
    print("  - models/business_scaler.pkl")
    print("\n다음 단계: scripts/step3_create_ranking_data.py 실행")
    print("=" * 80)

if __name__ == "__main__":
    main()

