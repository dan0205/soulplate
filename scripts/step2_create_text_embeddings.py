"""
Step 2: 리뷰 텍스트 임베딩 생성
- TF-IDF Vectorizer로 리뷰 텍스트를 100차원 벡터로 변환
- Vectorizer 모델 저장하여 실제 서비스에서 재사용

입력:
- data/raw/review_100k_absa_with_text.csv

출력:
- data/processed/review_text_embeddings.csv (review_id + 100개 임베딩 컬럼)
- models/tfidf_vectorizer.pkl (TF-IDF Vectorizer 모델)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def create_text_embeddings():
    """TF-IDF 임베딩 생성"""
    print("=" * 80)
    print("Step 2: 리뷰 텍스트 임베딩 생성")
    print("=" * 80)
    
    # 1. 데이터 로딩
    print("\n[1/4] 리뷰 데이터 로딩 중...")
    df = pd.read_csv("data/raw/review_100k_absa_with_text.csv")
    print(f"  [OK] {len(df):,}개 리뷰 로드")
    
    # 텍스트 컬럼 확인
    if 'text' not in df.columns:
        raise ValueError("'text' 컬럼이 없습니다!")
    
    # 결측치 처리
    df['text'] = df['text'].fillna('')
    print(f"  텍스트 샘플: {df['text'].iloc[0][:100]}...")
    
    # 2. TF-IDF Vectorizer 학습
    print("\n[2/4] TF-IDF Vectorizer 학습 중...")
    vectorizer = TfidfVectorizer(
        max_features=100,  # 100차원
        min_df=5,          # 최소 5개 문서에 등장
        max_df=0.8,        # 최대 80% 문서에 등장
        ngram_range=(1, 2), # unigram, bigram
        sublinear_tf=True   # log scaling
    )
    
    # Fit & Transform
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    print(f"  [OK] TF-IDF 학습 완료")
    print(f"  형태: {tfidf_matrix.shape}")
    print(f"  어휘 크기: {len(vectorizer.vocabulary_)}")
    
    # 3. 임베딩을 DataFrame으로 변환
    print("\n[3/4] 임베딩 데이터프레임 생성 중...")
    
    # Dense matrix로 변환
    embeddings_array = tfidf_matrix.toarray()
    
    # 컬럼 이름 생성
    embedding_cols = [f'text_embed_{i}' for i in range(100)]
    
    # DataFrame 생성
    embeddings_df = pd.DataFrame(
        embeddings_array,
        columns=embedding_cols
    )
    
    # review_id 추가
    embeddings_df.insert(0, 'review_id', df['review_id'])
    
    print(f"  [OK] 형태: {embeddings_df.shape}")
    print(f"  컬럼: review_id + {len(embedding_cols)}개 임베딩")
    
    # 통계 확인
    print(f"\n  임베딩 통계:")
    print(f"    평균: {embeddings_array.mean():.6f}")
    print(f"    표준편차: {embeddings_array.std():.6f}")
    print(f"    Min: {embeddings_array.min():.6f}")
    print(f"    Max: {embeddings_array.max():.6f}")
    
    # 4. 저장
    print("\n[4/4] 저장 중...")
    
    # CSV 저장
    os.makedirs("data/processed", exist_ok=True)
    embeddings_df.to_csv("data/processed/review_text_embeddings.csv", index=False, encoding='utf-8-sig')
    print(f"  [OK] CSV 저장: data/processed/review_text_embeddings.csv")
    
    # Vectorizer 저장
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  [OK] Vectorizer 저장: models/tfidf_vectorizer.pkl")
    
    # 샘플 출력
    print(f"\n  샘플 데이터 (처음 3개):")
    print(embeddings_df.head(3))
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Step 2 완료!")
    print("\n다음 단계:")
    print("  - scripts/step3_create_ranking_data.py (학습 데이터 생성)")
    print("=" * 80)

if __name__ == "__main__":
    create_text_embeddings()


