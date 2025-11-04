"""
FAISS 인덱스 구축 스크립트
ItemTower 모델로 모든 비즈니스 아이템의 벡터를 생성하고 FAISS 인덱스에 저장
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
import faiss
import json
from pathlib import Path
from tqdm import tqdm

from backend_model.models.two_tower import ItemTower

# 설정
PROCESSED_DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 512

print(f"Using device: {DEVICE}")

def load_item_tower():
    """ItemTower 모델 로드"""
    print("Loading ItemTower model...")
    
    # 비즈니스 개수 확인
    businesses_df = pd.read_csv(PROCESSED_DATA_DIR / "businesses.csv")
    num_items = len(businesses_df)
    
    # 모델 생성
    model = ItemTower(
        num_items=num_items,
        embedding_dim=64,
        hidden_dim=256,
        output_dim=128
    )
    
    # 가중치 로드
    state_dict = torch.load(MODEL_DIR / "item_tower.pth", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded. Num items: {num_items}")
    return model, businesses_df

def load_id_mappings():
    """ID 매핑 로드"""
    with open(PROCESSED_DATA_DIR / "business_id_to_idx.json", 'r') as f:
        business_id_to_idx = json.load(f)
    
    with open(PROCESSED_DATA_DIR / "idx_to_business_id.json", 'r') as f:
        idx_to_business_id = json.load(f)
    
    return business_id_to_idx, idx_to_business_id

def generate_item_vectors(model, businesses_df, business_id_to_idx, idx_to_business_id):
    """모든 아이템의 벡터 생성"""
    print(f"\nGenerating vectors for {len(businesses_df)} businesses...")
    
    num_items = len(businesses_df)
    output_dim = 128
    item_vectors = np.zeros((num_items, output_dim), dtype=np.float32)
    
    # Set DataFrame index to business_id for easy access
    businesses_df = businesses_df.set_index('business_id')
    
    # Process in batches
    num_batches = (num_items + BATCH_SIZE - 1) // BATCH_SIZE
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, num_items)
            batch_size = end_idx - start_idx
            
            # Get business IDs for this batch
            batch_business_ids = [idx_to_business_id[str(i)] for i in range(start_idx, end_idx)]
            
            # Prepare batch data
            item_indices = torch.arange(start_idx, end_idx, dtype=torch.long).to(DEVICE)
            
            # Get item features
            item_features_list = []
            for business_id in batch_business_ids:
                row = businesses_df.loc[business_id]
                features = [
                    row['stars'] / 5.0,
                    row['review_count'] / 500.0,
                    float(row['is_open'])
                ]
                item_features_list.append(features)
            
            item_features = torch.tensor(item_features_list, dtype=torch.float32).to(DEVICE)
            
            # Generate vectors
            vectors = model(item_indices, item_features)
            
            # Store in numpy array
            item_vectors[start_idx:end_idx] = vectors.cpu().numpy()
    
    print(f"Generated {len(item_vectors)} item vectors")
    print(f"Vector shape: {item_vectors.shape}")
    print(f"Vector stats: min={item_vectors.min():.3f}, max={item_vectors.max():.3f}, mean={item_vectors.mean():.3f}")
    
    return item_vectors

def build_faiss_index(item_vectors):
    """FAISS 인덱스 구축"""
    print("\nBuilding FAISS index...")
    
    dimension = item_vectors.shape[1]
    
    # Normalize vectors for cosine similarity (Inner Product)
    faiss.normalize_L2(item_vectors)
    
    # Create index (IndexFlatIP for Inner Product / Cosine Similarity)
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors
    index.add(item_vectors)
    
    print(f"FAISS index built successfully!")
    print(f"  Dimension: {dimension}")
    print(f"  Total vectors: {index.ntotal}")
    
    return index

def save_index(index, idx_to_business_id):
    """인덱스 및 매핑 저장"""
    print("\nSaving FAISS index and mappings...")
    
    # Save FAISS index
    index_path = MODEL_DIR / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to: {index_path}")
    
    # Save ID mapping (for model API to use)
    mapping_path = MODEL_DIR / "idx_to_business_id.json"
    with open(mapping_path, 'w') as f:
        json.dump(idx_to_business_id, f)
    print(f"Saved ID mapping to: {mapping_path}")

def test_index(index, idx_to_business_id, businesses_df):
    """인덱스 테스트"""
    print("\n" + "=" * 60)
    print("Testing FAISS Index")
    print("=" * 60)
    
    businesses_df = businesses_df.set_index('business_id')
    
    # Random test query
    test_idx = 0
    test_business_id = idx_to_business_id[str(test_idx)]
    
    print(f"\nTest Query: Business Index {test_idx}")
    print(f"Business ID: {test_business_id}")
    print(f"Business Name: {businesses_df.loc[test_business_id]['name']}")
    print(f"Stars: {businesses_df.loc[test_business_id]['stars']}")
    
    # Get vector
    test_vector = index.reconstruct(test_idx).reshape(1, -1)
    
    # Search
    k = 5
    distances, indices = index.search(test_vector, k)
    
    print(f"\nTop {k} similar businesses:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        biz_id = idx_to_business_id[str(idx)]
        biz_name = businesses_df.loc[biz_id]['name']
        biz_stars = businesses_df.loc[biz_id]['stars']
        print(f"  {i+1}. (Score: {dist:.4f}) {biz_name} - {biz_stars} stars")
    
    print("\n" + "=" * 60)

def main():
    """메인 함수"""
    print("=" * 60)
    print("Building FAISS Index for Two-Tower Model")
    print("=" * 60)
    
    # Load model
    model, businesses_df = load_item_tower()
    
    # Load ID mappings
    business_id_to_idx, idx_to_business_id = load_id_mappings()
    
    # Generate item vectors
    item_vectors = generate_item_vectors(model, businesses_df, business_id_to_idx, idx_to_business_id)
    
    # Build FAISS index
    index = build_faiss_index(item_vectors)
    
    # Save
    save_index(index, idx_to_business_id)
    
    # Test
    test_index(index, idx_to_business_id, businesses_df)
    
    print("\n" + "=" * 60)
    print("FAISS index building completed successfully!")
    print("=" * 60)
    print(f"Files saved to: {MODEL_DIR}")
    print(f"  - index.faiss: FAISS index file")
    print(f"  - idx_to_business_id.json: Index to Business ID mapping")
    print("\nYou can now use these files in the Model API (Tier 3)")

if __name__ == "__main__":
    main()

