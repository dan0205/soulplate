"""
Scaler 파일 생성
- 학습 데이터에서 StandardScaler 학습
- user_scaler.pkl, business_scaler.pkl 저장
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

def create_scalers():
    """학습 데이터에서 Scaler 생성"""
    print("=" * 80)
    print("Scaler 생성")
    print("=" * 80)
    
    # 1. 학습 데이터 로딩
    print("\n[1/4] 학습 데이터 로딩 중...")
    train_df = pd.read_csv("data/processed/ranking_train.csv")
    print(f"  [OK] {len(train_df):,}개 데이터")
    print(f"  총 컬럼: {len(train_df.columns)}개")
    
    # 2. 피처 분리
    print("\n[2/4] 피처 분리 중...")
    
    # 제외할 컬럼
    exclude_cols = ['user_id', 'business_id', 'avg_stars', 'review_count']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # 전체 피처 추출
    all_features = train_df[feature_cols].values
    
    # User와 Business 피처로 분할 (절반씩)
    mid = len(feature_cols) // 2
    user_features = all_features[:, :mid]
    business_features = all_features[:, mid:]
    
    print(f"  User 피처: {user_features.shape[1]}차원")
    print(f"  Business 피처: {business_features.shape[1]}차원")
    
    # 3. StandardScaler 학습
    print("\n[3/4] StandardScaler 학습 중...")
    
    # User Scaler
    user_scaler = StandardScaler()
    user_scaler.fit(user_features)
    print(f"  [OK] User Scaler 학습 완료")
    print(f"    Mean: {user_scaler.mean_[:5]}...")
    print(f"    Std: {user_scaler.scale_[:5]}...")
    
    # Business Scaler
    business_scaler = StandardScaler()
    business_scaler.fit(business_features)
    print(f"  [OK] Business Scaler 학습 완료")
    print(f"    Mean: {business_scaler.mean_[:5]}...")
    print(f"    Std: {business_scaler.scale_[:5]}...")
    
    # 4. 저장
    print("\n[4/4] Scaler 저장 중...")
    
    os.makedirs("models", exist_ok=True)
    
    with open("models/user_scaler.pkl", 'wb') as f:
        pickle.dump(user_scaler, f)
    print(f"  [OK] User Scaler 저장: models/user_scaler.pkl")
    
    with open("models/business_scaler.pkl", 'wb') as f:
        pickle.dump(business_scaler, f)
    print(f"  [OK] Business Scaler 저장: models/business_scaler.pkl")
    
    # 검증: 변환 테스트
    print("\n[검증] Scaler 테스트...")
    sample_user = user_features[0:1]
    sample_business = business_features[0:1]
    
    scaled_user = user_scaler.transform(sample_user)
    scaled_business = business_scaler.transform(sample_business)
    
    print(f"  원본 User 피처 (처음 5개): {sample_user[0, :5]}")
    print(f"  변환 User 피처 (처음 5개): {scaled_user[0, :5]}")
    print(f"  원본 Business 피처 (처음 5개): {sample_business[0, :5]}")
    print(f"  변환 Business 피처 (처음 5개): {scaled_business[0, :5]}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Scaler 생성 완료!")
    print("\n생성된 파일:")
    print("  - models/user_scaler.pkl")
    print("  - models/business_scaler.pkl")
    print("=" * 80)

if __name__ == "__main__":
    create_scalers()

