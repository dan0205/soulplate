"""
DeepFM 랭킹 모델
- FM Layer: 피처 조합 자동 학습
- Deep Layer: 비선형 패턴 학습
- 출력: 별점 예측 (1~5)
"""

import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, input_dim, embed_dim=16, hidden_dims=[128, 64, 32]):
        """
        DeepFM 모델
        
        Args:
            input_dim: 입력 피처 차원
            embed_dim: FM 임베딩 차원
            hidden_dims: Deep 레이어 차원들
        """
        super(DeepFM, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # FM Part: 1차 + 2차 상호작용
        # 1차: 선형 변환
        self.fm_linear = nn.Linear(input_dim, 1)
        
        # 2차: 임베딩을 통한 피처 조합
        self.fm_embeddings = nn.Linear(input_dim, embed_dim)
        
        # Deep Part: DNN
        deep_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            deep_layers.append(nn.BatchNorm1d(hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        self.deep_layers = nn.Sequential(*deep_layers)
        
        # 최종 출력
        self.final_linear = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            predictions: (batch_size, 1) - 별점 예측
        """
        # FM Part
        # 1차 항
        fm_linear_part = self.fm_linear(x)  # (batch, 1)
        
        # 2차 항: sum(embed)^2 - sum(embed^2)
        embeddings = self.fm_embeddings(x)  # (batch, embed_dim)
        sum_of_square = torch.pow(embeddings, 2).sum(dim=1, keepdim=True)  # (batch, 1)
        square_of_sum = torch.pow(embeddings.sum(dim=1, keepdim=True), 2)  # (batch, 1)
        fm_cross_part = 0.5 * (square_of_sum - sum_of_square)  # (batch, 1)
        
        fm_output = fm_linear_part + fm_cross_part  # (batch, 1)
        
        # Deep Part
        deep_output = self.deep_layers(x)
        deep_output = self.final_linear(deep_output)  # (batch, 1)
        
        # 결합
        output = fm_output + deep_output  # (batch, 1)
        
        # 별점 범위로 제한 (1~5) - sigmoid 사용 후 스케일링
        output = torch.sigmoid(output) * 4 + 1  # [0,1] -> [1,5]
        
        return output.squeeze()  # (batch,)

class DeepFMTrainer:
    """DeepFM 학습 클래스"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    def train_epoch(self, train_loader):
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
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
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
        
        rmse = (total_loss / len(val_loader)) ** 0.5
        return rmse
    
    def predict(self, x):
        """예측"""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).to(self.device)
            predictions = self.model(x)
        return predictions.cpu().numpy()

