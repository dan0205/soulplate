"""
Two-Tower 모델 정의
User Tower와 Item Tower를 각각 정의하고, 학습용 Combined Model도 포함
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    """User Tower: 사용자 임베딩 생성"""
    
    def __init__(self, num_users, embedding_dim=64, hidden_dim=256, output_dim=128):
        super(UserTower, self).__init__()
        
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        # User ID 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # MLP layers
        self.fc1 = nn.Linear(embedding_dim + 4, hidden_dim)  # +4 for user features
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, user_idx, user_features):
        """
        Args:
            user_idx: (batch_size,) - User index
            user_features: (batch_size, 4) - [age, review_count, useful, average_stars]
        
        Returns:
            user_vector: (batch_size, output_dim)
        """
        # User embedding
        user_emb = self.user_embedding(user_idx)  # (batch_size, embedding_dim)
        
        # Concatenate with features
        x = torch.cat([user_emb, user_features], dim=1)  # (batch_size, embedding_dim + 4)
        
        # MLP
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
        return x

class ItemTower(nn.Module):
    """Item Tower: 아이템(비즈니스) 임베딩 생성"""
    
    def __init__(self, num_items, embedding_dim=64, hidden_dim=256, output_dim=128):
        super(ItemTower, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        # Item ID 임베딩
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.fc1 = nn.Linear(embedding_dim + 3, hidden_dim)  # +3 for item features
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, item_idx, item_features):
        """
        Args:
            item_idx: (batch_size,) - Item index
            item_features: (batch_size, 3) - [stars, review_count, is_open]
        
        Returns:
            item_vector: (batch_size, output_dim)
        """
        # Item embedding
        item_emb = self.item_embedding(item_idx)  # (batch_size, embedding_dim)
        
        # Concatenate with features
        x = torch.cat([item_emb, item_features], dim=1)  # (batch_size, embedding_dim + 3)
        
        # MLP
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
        return x

class TwoTowerModel(nn.Module):
    """
    Two-Tower Model: User Tower + Item Tower + Dot Product
    학습 전용 모델
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=256, output_dim=128):
        super(TwoTowerModel, self).__init__()
        
        self.user_tower = UserTower(num_users, embedding_dim, hidden_dim, output_dim)
        self.item_tower = ItemTower(num_items, embedding_dim, hidden_dim, output_dim)
        
        # Temperature parameter for scaling dot product
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, user_idx, user_features, item_idx, item_features):
        """
        Args:
            user_idx: (batch_size,)
            user_features: (batch_size, 4)
            item_idx: (batch_size,)
            item_features: (batch_size, 3)
        
        Returns:
            scores: (batch_size,) - Dot product scores
        """
        # Get embeddings from both towers
        user_vec = self.user_tower(user_idx, user_features)  # (batch_size, output_dim)
        item_vec = self.item_tower(item_idx, item_features)  # (batch_size, output_dim)
        
        # Dot product (cosine similarity since vectors are normalized)
        scores = torch.sum(user_vec * item_vec, dim=1)  # (batch_size,)
        
        # Scale by temperature
        scores = scores / self.temperature
        
        return scores
    
    def predict(self, user_idx, user_features, item_idx, item_features):
        """예측 (sigmoid 적용)"""
        scores = self.forward(user_idx, user_features, item_idx, item_features)
        return torch.sigmoid(scores)

def create_two_tower_model(num_users, num_items, device='cpu'):
    """Two-Tower 모델 생성 헬퍼 함수"""
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64,
        hidden_dim=256,
        output_dim=128
    )
    model = model.to(device)
    return model

if __name__ == "__main__":
    # 테스트
    print("Testing Two-Tower Model...")
    
    num_users = 1000
    num_items = 500
    batch_size = 32
    
    model = create_two_tower_model(num_users, num_items)
    
    # 더미 입력
    user_idx = torch.randint(0, num_users, (batch_size,))
    user_features = torch.randn(batch_size, 4)
    item_idx = torch.randint(0, num_items, (batch_size,))
    item_features = torch.randn(batch_size, 3)
    
    # Forward pass
    scores = model(user_idx, user_features, item_idx, item_features)
    print(f"Scores shape: {scores.shape}")
    print(f"Scores range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Test individual towers
    user_vec = model.user_tower(user_idx, user_features)
    item_vec = model.item_tower(item_idx, item_features)
    print(f"User vector shape: {user_vec.shape}")
    print(f"Item vector shape: {item_vec.shape}")
    print(f"User vector norm: {torch.norm(user_vec, dim=1).mean():.3f}")
    print(f"Item vector norm: {torch.norm(item_vec, dim=1).mean():.3f}")
    
    print("\nModel test passed!")

