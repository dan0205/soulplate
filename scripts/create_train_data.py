"""
Two-Tower 모델 학습용 데이터셋 생성
Positive/Negative 샘플링 및 Train/Valid/Test 분할
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

# 설정
PROCESSED_DATA_DIR = Path("data/processed")
NEGATIVE_RATIO = 4  # Positive 1개당 Negative 4개
RANDOM_SEED = 42
TEST_SIZE = 0.1
VALID_SIZE = 0.1

# 시드 고정
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_processed_data():
    """전처리된 데이터 로드"""
    print("Loading processed data...")
    
    users_df = pd.read_csv(PROCESSED_DATA_DIR / "users.csv")
    businesses_df = pd.read_csv(PROCESSED_DATA_DIR / "businesses.csv")
    reviews_df = pd.read_csv(PROCESSED_DATA_DIR / "reviews.csv")
    
    print(f"Loaded {len(users_df)} users, {len(businesses_df)} businesses, {len(reviews_df)} reviews")
    return users_df, businesses_df, reviews_df

def create_positive_samples(reviews_df):
    """Positive 샘플 생성 (실제 리뷰가 있고 평점 >= 4)"""
    print("\nCreating positive samples...")
    
    # 평점이 4 이상인 리뷰만 선택
    positive_reviews = reviews_df[reviews_df['stars'] >= 4].copy()
    
    positive_samples = positive_reviews[['user_id', 'business_id', 'stars']].copy()
    positive_samples['label'] = 1
    positive_samples['sample_type'] = 'positive'
    
    print(f"Created {len(positive_samples)} positive samples")
    return positive_samples

def create_negative_samples(reviews_df, users_df, businesses_df, num_negatives):
    """Negative 샘플 생성 (랜덤 샘플링)"""
    print(f"\nCreating {num_negatives} negative samples...")
    
    # 실제 (user, business) 쌍을 set으로 저장 (중복 방지)
    existing_pairs = set(zip(reviews_df['user_id'], reviews_df['business_id']))
    
    negative_samples = []
    user_ids = users_df['user_id'].tolist()
    business_ids = businesses_df['business_id'].tolist()
    
    attempts = 0
    max_attempts = num_negatives * 10  # 무한 루프 방지
    
    while len(negative_samples) < num_negatives and attempts < max_attempts:
        # 랜덤하게 user와 business 선택
        user_id = random.choice(user_ids)
        business_id = random.choice(business_ids)
        
        # 실제로 존재하지 않는 쌍만 선택
        if (user_id, business_id) not in existing_pairs:
            negative_samples.append({
                'user_id': user_id,
                'business_id': business_id,
                'stars': 0,  # Negative는 stars 0으로 설정
                'label': 0,
                'sample_type': 'negative'
            })
            existing_pairs.add((user_id, business_id))  # 중복 방지를 위해 추가
        
        attempts += 1
    
    negative_df = pd.DataFrame(negative_samples)
    print(f"Created {len(negative_df)} negative samples (attempts: {attempts})")
    return negative_df

def split_dataset(samples_df):
    """Train/Valid/Test 분할"""
    print("\nSplitting dataset...")
    
    # Train + Valid / Test 분할
    train_valid, test = train_test_split(
        samples_df, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=samples_df['label']
    )
    
    # Train / Valid 분할
    train, valid = train_test_split(
        train_valid,
        test_size=VALID_SIZE / (1 - TEST_SIZE),  # 전체 데이터에서 VALID_SIZE 비율이 되도록
        random_state=RANDOM_SEED,
        stratify=train_valid['label']
    )
    
    print(f"Train: {len(train)} ({len(train[train['label']==1])} pos, {len(train[train['label']==0])} neg)")
    print(f"Valid: {len(valid)} ({len(valid[valid['label']==1])} pos, {len(valid[valid['label']==0])} neg)")
    print(f"Test:  {len(test)} ({len(test[test['label']==1])} pos, {len(test[test['label']==0])} neg)")
    
    return train, valid, test

def save_datasets(train, valid, test):
    """데이터셋 저장"""
    print("\nSaving datasets...")
    
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    print(f"Saved train.csv ({len(train)} records)")
    
    valid.to_csv(PROCESSED_DATA_DIR / "valid.csv", index=False)
    print(f"Saved valid.csv ({len(valid)} records)")
    
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    print(f"Saved test.csv ({len(test)} records)")

def print_statistics(train, valid, test):
    """통계 출력"""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    total = len(train) + len(valid) + len(test)
    print(f"\nTotal samples: {total}")
    print(f"  Train: {len(train)} ({len(train)/total*100:.1f}%)")
    print(f"  Valid: {len(valid)} ({len(valid)/total*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/total*100:.1f}%)")
    
    for name, df in [("Train", train), ("Valid", valid), ("Test", test)]:
        print(f"\n[{name}]")
        print(f"  Positive: {len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.1f}%)")
        print(f"  Negative: {len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.1f}%)")
        print(f"  Unique users: {df['user_id'].nunique()}")
        print(f"  Unique businesses: {df['business_id'].nunique()}")
    
    print("\n" + "=" * 60)

def create_id_mappings(users_df, businesses_df):
    """ID 매핑 생성 및 저장"""
    print("\nCreating ID mappings...")
    
    # User ID → Index
    user_id_to_idx = {uid: idx for idx, uid in enumerate(users_df['user_id'])}
    idx_to_user_id = {idx: uid for uid, idx in user_id_to_idx.items()}
    
    # Business ID → Index
    business_id_to_idx = {bid: idx for idx, bid in enumerate(businesses_df['business_id'])}
    idx_to_business_id = {idx: bid for bid, idx in business_id_to_idx.items()}
    
    # 저장
    import json
    
    with open(PROCESSED_DATA_DIR / "user_id_to_idx.json", 'w') as f:
        json.dump(user_id_to_idx, f)
    
    with open(PROCESSED_DATA_DIR / "idx_to_user_id.json", 'w') as f:
        json.dump(idx_to_user_id, f)
    
    with open(PROCESSED_DATA_DIR / "business_id_to_idx.json", 'w') as f:
        json.dump(business_id_to_idx, f)
    
    with open(PROCESSED_DATA_DIR / "idx_to_business_id.json", 'w') as f:
        json.dump(idx_to_business_id, f)
    
    print(f"Saved ID mappings ({len(user_id_to_idx)} users, {len(business_id_to_idx)} businesses)")

def main():
    """메인 함수"""
    print("=" * 60)
    print("Creating Training Dataset for Two-Tower Model")
    print("=" * 60)
    
    # Load data
    users_df, businesses_df, reviews_df = load_processed_data()
    
    # Create positive samples
    positive_samples = create_positive_samples(reviews_df)
    
    # Create negative samples
    num_negatives = len(positive_samples) * NEGATIVE_RATIO
    negative_samples = create_negative_samples(reviews_df, users_df, businesses_df, num_negatives)
    
    # Combine
    all_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
    
    # Shuffle
    all_samples = all_samples.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"\nTotal samples: {len(all_samples)}")
    print(f"  Positive: {len(positive_samples)} ({len(positive_samples)/len(all_samples)*100:.1f}%)")
    print(f"  Negative: {len(negative_samples)} ({len(negative_samples)/len(all_samples)*100:.1f}%)")
    
    # Split dataset
    train, valid, test = split_dataset(all_samples)
    
    # Save
    save_datasets(train, valid, test)
    
    # Create ID mappings
    create_id_mappings(users_df, businesses_df)
    
    # Print statistics
    print_statistics(train, valid, test)
    
    print("\nDataset creation completed successfully!")

if __name__ == "__main__":
    main()

