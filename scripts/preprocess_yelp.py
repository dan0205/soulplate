"""
Yelp 데이터 전처리 스크립트
원본 JSON 파일을 읽어서 학습에 적합한 형태로 변환
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import random

# 설정
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_json_lines(file_path):
    """Load newline-delimited JSON file"""
    print(f"Loading {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def preprocess_users(users_data):
    """전처리: User 데이터"""
    print("Preprocessing users...")
    
    df = pd.DataFrame(users_data)
    
    # 필요한 컬럼만 선택
    cols = ['user_id', 'name', 'review_count', 'yelping_since', 
            'useful', 'funny', 'cool', 'fans', 'average_stars']
    df = df[cols]
    
    # 데모용 age와 gender 생성 (실제 Yelp 데이터에는 없음)
    np.random.seed(42)
    df['age'] = np.random.randint(18, 70, size=len(df))
    df['gender'] = np.random.choice(['M', 'F', 'Other'], size=len(df), p=[0.48, 0.48, 0.04])
    
    # yelping_since를 년도로 변환
    df['yelping_since'] = pd.to_datetime(df['yelping_since'])
    df['years_on_yelp'] = (pd.Timestamp.now() - df['yelping_since']).dt.days / 365.25
    
    # 결측치 처리
    df = df.fillna(0)
    
    print(f"Processed {len(df)} users")
    return df

def preprocess_businesses(business_data):
    """전처리: Business 데이터"""
    print("Preprocessing businesses...")
    
    df = pd.DataFrame(business_data)
    
    # 필요한 컬럼만 선택
    cols = ['business_id', 'name', 'address', 'city', 'state', 
            'postal_code', 'latitude', 'longitude', 'stars', 
            'review_count', 'is_open', 'categories']
    df = df[cols]
    
    # categories를 리스트로 변환 (null 처리)
    df['categories'] = df['categories'].fillna('')
    df['categories'] = df['categories'].apply(lambda x: x.split(', ') if x else [])
    
    # categories를 문자열로 다시 변환 (저장용)
    df['categories_str'] = df['categories'].apply(lambda x: ', '.join(x))
    
    # 결측치 처리
    df = df.fillna('')
    
    print(f"Processed {len(df)} businesses")
    return df

def preprocess_reviews(review_data):
    """전처리: Review 데이터"""
    print("Preprocessing reviews...")
    
    df = pd.DataFrame(review_data)
    
    # 필요한 컬럼만 선택
    cols = ['review_id', 'user_id', 'business_id', 'stars', 
            'useful', 'funny', 'cool', 'text', 'date']
    df = df[cols]
    
    # date를 datetime으로 변환
    df['date'] = pd.to_datetime(df['date'])
    
    # 텍스트 길이 계산
    df['text_length'] = df['text'].str.len()
    
    # 최근 2년 데이터만 사용 (성능 최적화)
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=730)
    df = df[df['date'] >= cutoff_date]
    
    # date를 문자열로 변환 (저장용)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Processed {len(df)} reviews (filtered to last 2 years)")
    return df

def create_user_features(users_df, reviews_df):
    """사용자 특징 생성"""
    print("Creating user features...")
    
    # 각 사용자의 평균 평점 계산
    user_avg_stars = reviews_df.groupby('user_id')['stars'].mean().reset_index()
    user_avg_stars.columns = ['user_id', 'user_avg_rating']
    
    # 각 사용자의 리뷰 수 계산
    user_review_count = reviews_df.groupby('user_id').size().reset_index(name='actual_review_count')
    
    # Merge
    users_df = users_df.merge(user_avg_stars, on='user_id', how='left')
    users_df = users_df.merge(user_review_count, on='user_id', how='left')
    
    # 결측치 처리
    users_df['user_avg_rating'] = users_df['user_avg_rating'].fillna(users_df['average_stars'])
    users_df['actual_review_count'] = users_df['actual_review_count'].fillna(0)
    
    return users_df

def create_business_features(businesses_df, reviews_df):
    """비즈니스 특징 생성"""
    print("Creating business features...")
    
    # 각 비즈니스의 평균 평점 계산 (최근 리뷰 기준)
    biz_avg_stars = reviews_df.groupby('business_id')['stars'].mean().reset_index()
    biz_avg_stars.columns = ['business_id', 'recent_avg_rating']
    
    # 각 비즈니스의 최근 리뷰 수
    biz_review_count = reviews_df.groupby('business_id').size().reset_index(name='recent_review_count')
    
    # Merge
    businesses_df = businesses_df.merge(biz_avg_stars, on='business_id', how='left')
    businesses_df = businesses_df.merge(biz_review_count, on='business_id', how='left')
    
    # 결측치 처리
    businesses_df['recent_avg_rating'] = businesses_df['recent_avg_rating'].fillna(businesses_df['stars'])
    businesses_df['recent_review_count'] = businesses_df['recent_review_count'].fillna(0)
    
    return businesses_df

def save_processed_data(users_df, businesses_df, reviews_df):
    """전처리된 데이터 저장"""
    print("\nSaving processed data...")
    
    users_df.to_csv(PROCESSED_DATA_DIR / "users.csv", index=False)
    print(f"Saved users.csv ({len(users_df)} records)")
    
    businesses_df.to_csv(PROCESSED_DATA_DIR / "businesses.csv", index=False)
    print(f"Saved businesses.csv ({len(businesses_df)} records)")
    
    reviews_df.to_csv(PROCESSED_DATA_DIR / "reviews.csv", index=False)
    print(f"Saved reviews.csv ({len(reviews_df)} records)")

def print_statistics(users_df, businesses_df, reviews_df):
    """데이터 통계 출력"""
    print("\n" + "=" * 60)
    print("Data Statistics")
    print("=" * 60)
    
    print("\n[Users]")
    print(f"Total users: {len(users_df)}")
    print(f"Average age: {users_df['age'].mean():.1f}")
    print(f"Gender distribution: {users_df['gender'].value_counts().to_dict()}")
    print(f"Average review count: {users_df['review_count'].mean():.1f}")
    
    print("\n[Businesses]")
    print(f"Total businesses: {len(businesses_df)}")
    print(f"Average stars: {businesses_df['stars'].mean():.2f}")
    print(f"Cities: {businesses_df['city'].nunique()}")
    print(f"Open businesses: {businesses_df['is_open'].sum()}")
    
    print("\n[Reviews]")
    print(f"Total reviews: {len(reviews_df)}")
    print(f"Average stars: {reviews_df['stars'].mean():.2f}")
    print(f"Star distribution:")
    for star in sorted(reviews_df['stars'].unique()):
        count = (reviews_df['stars'] == star).sum()
        pct = count / len(reviews_df) * 100
        print(f"  {star} stars: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 60)

def main():
    """메인 함수"""
    print("=" * 60)
    print("Yelp Data Preprocessing")
    print("=" * 60)
    
    # Load raw data
    users_data = load_json_lines(RAW_DATA_DIR / "yelp_academic_dataset_user.json")
    business_data = load_json_lines(RAW_DATA_DIR / "yelp_academic_dataset_business.json")
    review_data = load_json_lines(RAW_DATA_DIR / "yelp_academic_dataset_review.json")
    
    # Preprocess
    users_df = preprocess_users(users_data)
    businesses_df = preprocess_businesses(business_data)
    reviews_df = preprocess_reviews(review_data)
    
    # Create features
    users_df = create_user_features(users_df, reviews_df)
    businesses_df = create_business_features(businesses_df, reviews_df)
    
    # Save
    save_processed_data(users_df, businesses_df, reviews_df)
    
    # Print statistics
    print_statistics(users_df, businesses_df, reviews_df)
    
    print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()

