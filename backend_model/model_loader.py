"""
모델 및 FAISS 인덱스 로더
"""

import torch
import faiss
import json
import pandas as pd
from pathlib import Path
import logging

from backend_model.models.two_tower import UserTower

logger = logging.getLogger(__name__)

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

class ModelLoader:
    """모델 및 데이터 로더 클래스"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_tower = None
        self.faiss_index = None
        self.idx_to_business_id = None
        self.business_id_to_idx = None
        self.users_df = None
        self.user_id_to_idx = None
        
        logger.info(f"Using device: {self.device}")
    
    def load_user_tower(self):
        """UserTower 모델 로드"""
        logger.info("Loading UserTower model...")
        
        try:
            # 사용자 수 확인
            users_df = pd.read_csv(DATA_DIR / "users.csv")
            num_users = len(users_df)
            
            # 모델 생성
            model = UserTower(
                num_users=num_users,
                embedding_dim=64,
                hidden_dim=256,
                output_dim=128
            )
            
            # 가중치 로드
            state_dict = torch.load(
                MODEL_DIR / "user_tower.pth", 
                map_location=self.device,
                weights_only=True
            )
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            
            self.user_tower = model
            logger.info(f"UserTower loaded successfully (num_users: {num_users})")
            
        except Exception as e:
            logger.error(f"Failed to load UserTower: {e}")
            raise
    
    def load_faiss_index(self):
        """FAISS 인덱스 및 매핑 로드"""
        logger.info("Loading FAISS index...")
        
        try:
            # FAISS 인덱스 로드
            index = faiss.read_index(str(MODEL_DIR / "index.faiss"))
            self.faiss_index = index
            
            # ID 매핑 로드
            with open(MODEL_DIR / "idx_to_business_id.json", 'r') as f:
                self.idx_to_business_id = json.load(f)
            
            with open(DATA_DIR / "business_id_to_idx.json", 'r') as f:
                self.business_id_to_idx = json.load(f)
            
            logger.info(f"FAISS index loaded (total vectors: {index.ntotal})")
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
    
    def load_user_data(self):
        """사용자 데이터 로드"""
        logger.info("Loading user data...")
        
        try:
            self.users_df = pd.read_csv(DATA_DIR / "users.csv")
            
            with open(DATA_DIR / "user_id_to_idx.json", 'r') as f:
                self.user_id_to_idx = json.load(f)
            
            logger.info(f"User data loaded (num_users: {len(self.users_df)})")
            
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
            raise
    
    def load_all(self):
        """모든 리소스 로드"""
        logger.info("Loading all resources...")
        
        self.load_user_tower()
        self.load_faiss_index()
        self.load_user_data()
        
        logger.info("All resources loaded successfully!")
    
    def get_user_vector(self, user_id: str, user_features: dict = None):
        """사용자 벡터 생성"""
        if self.user_tower is None:
            raise ValueError("UserTower not loaded")
        
        # User index
        if user_id not in self.user_id_to_idx:
            # 새로운 사용자인 경우 랜덤 인덱스 사용 (실제로는 cold start 처리 필요)
            logger.warning(f"Unknown user_id: {user_id}, using random index")
            user_idx = 0
        else:
            user_idx = self.user_id_to_idx[user_id]
        
        # User features
        if user_features is None:
            # 데이터베이스에서 가져오기
            if user_id in self.users_df['user_id'].values:
                user_row = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
                features = [
                    user_row['age'] / 100.0,
                    user_row['review_count'] / 100.0,
                    user_row['useful'] / 50.0,
                    user_row['average_stars'] / 5.0
                ]
            else:
                # 기본값 사용
                features = [0.3, 0.5, 0.4, 0.8]  # 기본 features
        else:
            # 제공된 features 사용
            features = [
                user_features.get('age', 30.0) / 100.0,
                user_features.get('review_count', 50.0) / 100.0,
                user_features.get('useful', 20.0) / 50.0,
                user_features.get('average_stars', 4.0) / 5.0
            ]
        
        # Tensor 생성
        user_idx_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        user_features_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)
        
        # 벡터 생성
        with torch.no_grad():
            user_vector = self.user_tower(user_idx_tensor, user_features_tensor)
        
        return user_vector.cpu().numpy()
    
    def search_similar_items(self, user_vector, top_k=10):
        """유사한 아이템 검색"""
        if self.faiss_index is None:
            raise ValueError("FAISS index not loaded")
        
        # FAISS 검색
        distances, indices = self.faiss_index.search(user_vector, top_k)
        
        # Index를 business_id로 변환
        business_ids = [self.idx_to_business_id[str(idx)] for idx in indices[0]]
        scores = distances[0].tolist()
        
        return business_ids, scores

# 전역 로더 인스턴스
model_loader = ModelLoader()

def get_model_loader():
    """모델 로더 인스턴스 반환"""
    return model_loader

