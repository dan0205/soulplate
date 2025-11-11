"""
Multi-Tower 랭킹 모델
- User Tower: User 피처 → 임베딩
- Business Tower: Business 피처 → 임베딩
- Interaction Layer: 두 임베딩 결합 → 별점 예측
"""

import torch
import torch.nn as nn

class MultiTowerModel(nn.Module):
    def __init__(self, user_input_dim, business_input_dim, 
                 tower_dims=[128, 64], interaction_dims=[64, 32]):
        """
        Multi-Tower 모델
        
        Args:
            user_input_dim: User 피처 차원
            business_input_dim: Business 피처 차원
            tower_dims: 각 타워의 hidden 차원들
            interaction_dims: Interaction 레이어 차원들
        """
        super(MultiTowerModel, self).__init__()
        
        self.user_input_dim = user_input_dim
        self.business_input_dim = business_input_dim
        
        # User Tower
        user_layers = []
        prev_dim = user_input_dim
        for dim in tower_dims:
            user_layers.append(nn.Linear(prev_dim, dim))
            user_layers.append(nn.BatchNorm1d(dim))
            user_layers.append(nn.ReLU())
            user_layers.append(nn.Dropout(0.3))
            prev_dim = dim
        self.user_tower = nn.Sequential(*user_layers)
        user_embed_dim = tower_dims[-1]
        
        # Business Tower
        business_layers = []
        prev_dim = business_input_dim
        for dim in tower_dims:
            business_layers.append(nn.Linear(prev_dim, dim))
            business_layers.append(nn.BatchNorm1d(dim))
            business_layers.append(nn.ReLU())
            business_layers.append(nn.Dropout(0.3))
            prev_dim = dim
        self.business_tower = nn.Sequential(*business_layers)
        business_embed_dim = tower_dims[-1]
        
        # Interaction Layer
        interaction_layers = []
        prev_dim = user_embed_dim + business_embed_dim
        for dim in interaction_dims:
            interaction_layers.append(nn.Linear(prev_dim, dim))
            interaction_layers.append(nn.BatchNorm1d(dim))
            interaction_layers.append(nn.ReLU())
            interaction_layers.append(nn.Dropout(0.2))
            prev_dim = dim
        
        self.interaction_layers = nn.Sequential(*interaction_layers)
        
        # 최종 출력
        self.final_linear = nn.Linear(prev_dim, 1)
        
    def forward(self, user_features, business_features):
        """
        Forward pass
        
        Args:
            user_features: (batch_size, user_input_dim)
            business_features: (batch_size, business_input_dim)
            
        Returns:
            predictions: (batch_size,) - 별점 예측
        """
        # User Tower
        user_embed = self.user_tower(user_features)  # (batch, 64)
        
        # Business Tower
        business_embed = self.business_tower(business_features)  # (batch, 64)
        
        # Concatenate
        combined = torch.cat([user_embed, business_embed], dim=1)  # (batch, 128)
        
        # Interaction
        interaction = self.interaction_layers(combined)
        
        # Final prediction
        output = self.final_linear(interaction)  # (batch, 1)
        
        # 별점 범위로 제한 (1~5)
        output = torch.sigmoid(output) * 4 + 1  # [0,1] -> [1,5]
        
        return output.squeeze()  # (batch,)
    
    def get_user_embedding(self, user_features):
        """User 임베딩만 추출"""
        self.eval()
        with torch.no_grad():
            return self.user_tower(user_features)
    
    def get_business_embedding(self, business_features):
        """Business 임베딩만 추출"""
        self.eval()
        with torch.no_grad():
            return self.business_tower(business_features)

class MultiTowerTrainer:
    """Multi-Tower 학습 클래스"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    def train_epoch(self, train_loader):
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0
        
        for user_features, business_features, targets in train_loader:
            user_features = user_features.to(self.device)
            business_features = business_features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            predictions = self.model(user_features, business_features)
            loss = self.criterion(predictions, targets)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for user_features, business_features, targets in val_loader:
                user_features = user_features.to(self.device)
                business_features = business_features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(user_features, business_features)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
        
        rmse = (total_loss / len(val_loader)) ** 0.5
        return rmse
    
    def predict(self, user_features, business_features):
        """예측"""
        self.model.eval()
        with torch.no_grad():
            user_features = torch.FloatTensor(user_features).to(self.device)
            business_features = torch.FloatTensor(business_features).to(self.device)
            predictions = self.model(user_features, business_features)
        return predictions.cpu().numpy()

