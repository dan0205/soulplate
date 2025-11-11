"""
Step 3: 학습 데이터 생성
- 각 (user_id, business_id) 쌍의 평균 별점 계산
- 전처리된 User/Business 피처 결합
- Train/Valid/Test 분할 (80/10/10)

출력:
- data/processed/ranking_train.csv
- data/processed/ranking_valid.csv
- data/processed/ranking_test.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_user_business_pairs():
    """User-Business 쌍과 평균 별점 생성"""
    print("=" * 80)
    print("Step 3: 학습 데이터 생성 (텍스트 임베딩 포함)")
    print("=" * 80)
    
    print("\n[1/6] 리뷰 데이터 로딩 중...")
    reviews_df = pd.read_csv("data/processed/review_absa_features.csv")
    print(f"  [OK] {len(reviews_df):,}개 리뷰")
    
    # 텍스트 임베딩 로딩
    print("\n[2/6] 텍스트 임베딩 로딩 중...")
    text_embeddings = pd.read_csv("data/processed/review_text_embeddings.csv")
    print(f"  [OK] {len(text_embeddings):,}개 임베딩 ({text_embeddings.shape[1]-1}차원)")
    
    # 리뷰와 임베딩 병합
    reviews_df = reviews_df.merge(text_embeddings, on='review_id', how='left')
    print(f"  병합 후: {reviews_df.shape}")
    
    # User-Business 쌍별 평균 별점 계산
    print("\n[3/6] User-Business 쌍별 평균 별점 계산 중...")
    pairs = reviews_df.groupby(['user_id', 'business_id'])['stars'].agg(['mean', 'count']).reset_index()
    pairs.columns = ['user_id', 'business_id', 'avg_stars', 'review_count']
    
    print(f"  [OK] {len(pairs):,}개의 User-Business 쌍 생성")
    print(f"  평균 별점 분포:")
    print(pairs['avg_stars'].describe())
    
    # User-Business 쌍별 평균 텍스트 임베딩 계산
    print("\n[4/6] User-Business 쌍별 평균 텍스트 임베딩 계산 중...")
    text_cols = [col for col in reviews_df.columns if col.startswith('text_embed_')]
    
    # 각 쌍별로 평균 계산
    text_avg = reviews_df.groupby(['user_id', 'business_id'])[text_cols].mean().reset_index()
    
    # pairs에 병합
    pairs = pairs.merge(text_avg, on=['user_id', 'business_id'], how='left')
    print(f"  [OK] {len(text_cols)}개 텍스트 피처 추가")
    
    return pairs

def merge_features(pairs):
    """User와 Business 피처 병합"""
    print("\n[5/6] User/Business 피처 병합 중...")
    
    # User 피처 로딩
    user_features = pd.read_csv("data/processed/user_preprocessed.csv")
    print(f"  User 피처: {user_features.shape}")
    
    # Business 피처 로딩
    business_features = pd.read_csv("data/processed/business_preprocessed.csv")
    print(f"  Business 피처: {business_features.shape}")
    
    # 병합
    data = pairs.merge(user_features, on='user_id', how='left')
    data = data.merge(business_features, on='business_id', how='left')
    
    print(f"  [OK] 병합 완료: {data.shape}")
    
    # 결측치 확인
    missing = data.isnull().sum().sum()
    if missing > 0:
        print(f"  [WARNING] 결측치 {missing}개 발견 - 0으로 채움")
        data = data.fillna(0)
    
    return data

def split_data(data):
    """Train/Valid/Test 분할 (80/10/10)"""
    print("\n[6/6] Train/Valid/Test 분할 중...")
    
    # 먼저 Train과 Temp(Valid+Test) 분할 (80/20)
    train_data, temp_data = train_test_split(
        data, 
        test_size=0.2, 
        random_state=42,
        stratify=pd.cut(data['avg_stars'], bins=[0, 2, 3, 4, 5], labels=[1,2,3,4])  # 별점 분포 유지
    )
    
    # Temp를 Valid와 Test로 분할 (50/50 = 각각 10%)
    valid_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=42,
        stratify=pd.cut(temp_data['avg_stars'], bins=[0, 2, 3, 4, 5], labels=[1,2,3,4])
    )
    
    print(f"  Train: {len(train_data):,}개 ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Valid: {len(valid_data):,}개 ({len(valid_data)/len(data)*100:.1f}%)")
    print(f"  Test:  {len(test_data):,}개 ({len(test_data)/len(data)*100:.1f}%)")
    
    # 별점 분포 확인
    print(f"\n  별점 분포 확인:")
    print(f"    Train: {train_data['avg_stars'].describe()[['mean', 'std']]}")
    print(f"    Valid: {valid_data['avg_stars'].describe()[['mean', 'std']]}")
    print(f"    Test:  {test_data['avg_stars'].describe()[['mean', 'std']]}")
    
    return train_data, valid_data, test_data

def save_datasets(train_data, valid_data, test_data):
    """데이터셋 저장"""
    print("\n[저장] 데이터셋 저장 중...")
    
    train_data.to_csv("data/processed/ranking_train.csv", index=False, encoding='utf-8-sig')
    valid_data.to_csv("data/processed/ranking_valid.csv", index=False, encoding='utf-8-sig')
    test_data.to_csv("data/processed/ranking_test.csv", index=False, encoding='utf-8-sig')
    
    print(f"  [OK] 저장 완료:")
    print(f"    data/processed/ranking_train.csv ({train_data.shape})")
    print(f"    data/processed/ranking_valid.csv ({valid_data.shape})")
    print(f"    data/processed/ranking_test.csv ({test_data.shape})")
    
    # 컬럼 정보 출력
    print(f"\n  컬럼 구성:")
    print(f"    - ID: user_id, business_id")
    print(f"    - Target: avg_stars")
    print(f"    - Meta: review_count")
    
    # ABSA 피처 개수 계산
    absa_cols = [c for c in train_data.columns if c.startswith('absa_')]
    text_cols = [c for c in train_data.columns if c.startswith('text_embed_')]
    
    print(f"    - User 피처: 6개 기본")
    print(f"    - Business 피처: 4개 기본")
    print(f"    - ABSA 피처: {len(absa_cols)}개 (User + Business 공유)")
    print(f"    - 텍스트 임베딩: {len(text_cols)}개 (새로 추가!)")

def main():
    # 1. User-Business 쌍 생성
    pairs = create_user_business_pairs()
    
    # 2. 피처 병합
    data = merge_features(pairs)
    
    # 3. Train/Valid/Test 분할
    train_data, valid_data, test_data = split_data(data)
    
    # 4. 저장
    save_datasets(train_data, valid_data, test_data)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Step 3 완료!")
    print("\n다음 단계:")
    print("  - scripts/step4_train_deepfm.py (DeepFM 학습)")
    print("  - scripts/step5_train_multitower.py (Multi-Tower 학습)")
    print("=" * 80)

if __name__ == "__main__":
    main()

