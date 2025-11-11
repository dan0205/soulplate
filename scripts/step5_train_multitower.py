"""
Step 5: Multi-Tower 모델 학습 (코랩용)
- User Tower: User 피처 → 임베딩
- Business Tower: Business 피처 → 임베딩
- Interaction Layer: 두 임베딩 결합 → 별점 예측
- Google Drive에서 데이터 로딩
"""

import sys
import os

# 코랩 환경 체크 및 Google Drive 마운트
try:
    from google.colab import drive
    IN_COLAB = True
    print("🔍 코랩 환경 감지됨")
except:
    IN_COLAB = False
    print("🔍 로컬 환경에서 실행 중")

# 경로 설정 및 프로젝트 루트 추가
if IN_COLAB:
    # Google Drive 마운트
    try:
        drive.mount('/content/drive')
        print("✅ Google Drive 마운트 완료")
    except:
        print("⚠️ Google Drive 이미 마운트됨")
    
    # 프로젝트 루트 경로 추가
    PROJECT_ROOT = "/content/drive/MyDrive/yelp_dataset"
    sys.path.insert(0, PROJECT_ROOT)
    print(f"📁 프로젝트 루트: {PROJECT_ROOT}")
    
    DATA_PATH = f"{PROJECT_ROOT}/train"
    MODEL_PATH = f"{PROJECT_ROOT}/models"
else:
    sys.path.append('.')
    DATA_PATH = "data/processed"
    MODEL_PATH = "models"

print(f"📂 데이터 경로: {DATA_PATH}")
print(f"📂 모델 저장 경로: {MODEL_PATH}")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==================== Multi-Tower 모델 정의 ====================
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
        self.final_linear = nn.Linear(prev_dim, 1)
        
    def forward(self, user_features, business_features):
        # User Tower
        user_embed = self.user_tower(user_features)
        
        # Business Tower
        business_embed = self.business_tower(business_features)
        
        # Concatenate
        combined = torch.cat([user_embed, business_embed], dim=1)
        
        # Interaction
        interaction = self.interaction_layers(combined)
        
        # Final prediction
        output = self.final_linear(interaction)
        output = torch.sigmoid(output) * 4 + 1  # [0,1] -> [1,5]
        
        return output.squeeze()

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
            
            predictions = self.model(user_features, business_features)
            loss = self.criterion(predictions, targets)
            
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

# ==================== 데이터셋 정의 ====================
class MultiTowerDataset(Dataset):
    """Multi-Tower 랭킹 데이터셋"""
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        
        # 피처와 타겟 분리
        feature_cols = [col for col in self.data.columns 
                       if col not in ['user_id', 'business_id', 'avg_stars', 'review_count']]
        
        all_features = self.data[feature_cols].values.astype(np.float32)
        
        # 전체 피처를 User와 Business로 분할 (절반씩)
        mid = len(feature_cols) // 2
        self.user_features = all_features[:, :mid]
        self.business_features = all_features[:, mid:]
        
        self.targets = self.data['avg_stars'].values.astype(np.float32)
        
        print(f"  User 피처 차원: {self.user_features.shape[1]}")
        print(f"  Business 피처 차원: {self.business_features.shape[1]}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.user_features[idx], self.business_features[idx], self.targets[idx]

def train_multitower():
    """Multi-Tower 모델 학습"""
    print("=" * 80)
    print("Step 5: Multi-Tower 모델 학습")
    print("=" * 80)
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n디바이스: {device}")
    
    # 데이터 로딩
    print("\n[1/5] 데이터 로딩 중...")
    train_dataset = MultiTowerDataset(f"{DATA_PATH}/ranking_train.csv")
    valid_dataset = MultiTowerDataset(f"{DATA_PATH}/ranking_valid.csv")
    test_dataset = MultiTowerDataset(f"{DATA_PATH}/ranking_test.csv")
    
    print(f"  Train: {len(train_dataset):,}개")
    print(f"  Valid: {len(valid_dataset):,}개")
    print(f"  Test:  {len(test_dataset):,}개")
    
    # DataLoader
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델 생성
    print("\n[2/5] Multi-Tower 모델 생성 중...")
    user_dim = train_dataset.user_features.shape[1]
    business_dim = train_dataset.business_features.shape[1]
    
    model = MultiTowerModel(
        user_input_dim=user_dim,
        business_input_dim=business_dim,
        tower_dims=[128, 64],
        interaction_dims=[64, 32]
    )
    
    print(f"  User Tower 입력: {user_dim}차원")
    print(f"  Business Tower 입력: {business_dim}차원")
    print(f"  Tower 구조: [128, 64]")
    print(f"  Interaction 구조: [64, 32]")
    print(f"  총 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
    
    # Trainer 생성
    trainer = MultiTowerTrainer(model, device=device)
    
    # 학습
    print("\n[3/5] 모델 학습 중...")
    epochs = 20
    best_valid_rmse = float('inf')
    train_losses = []
    valid_rmses = []
    
    for epoch in range(epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Validation
        valid_rmse = trainer.evaluate(valid_loader)
        
        train_losses.append(train_loss)
        valid_rmses.append(valid_rmse)
        
        print(f"  Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Valid RMSE: {valid_rmse:.4f}")
        
        # Best model 저장
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            os.makedirs(MODEL_PATH, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_PATH}/multitower_ranking.pth")
            print(f"    [BEST] 모델 저장!")
    
    # Best model 로딩
    print("\n[4/5] Best 모델 로딩 중...")
    model.load_state_dict(torch.load(f"{MODEL_PATH}/multitower_ranking.pth"))
    trainer.model = model.to(device)
    
    # Test 평가
    print("\n[5/5] Test 평가 중...")
    test_rmse = trainer.evaluate(test_loader)
    print(f"  Test RMSE: {test_rmse:.4f}")
    
    # 학습 곡선 저장
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(valid_rmses, label='Valid RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    plt.savefig(f"{MODEL_PATH}/multitower_training_curve.png", dpi=100, bbox_inches='tight')
    print(f"\n  학습 곡선 저장: {MODEL_PATH}/multitower_training_curve.png")
    
    # 샘플 예측 확인
    print("\n  샘플 예측 확인 (Test set 처음 5개):")
    sample_user_features = test_dataset.user_features[:5]
    sample_business_features = test_dataset.business_features[:5]
    sample_targets = test_dataset.targets[:5]
    predictions = trainer.predict(sample_user_features, sample_business_features)
    
    for i in range(5):
        print(f"    실제: {sample_targets[i]:.2f} | 예측: {predictions[i]:.2f} | 오차: {abs(sample_targets[i]-predictions[i]):.2f}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Multi-Tower 학습 완료!")
    print(f"\nBest Valid RMSE: {best_valid_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"\n모델 저장: {MODEL_PATH}/multitower_ranking.pth")
    print("=" * 80)

if __name__ == "__main__":
    train_multitower()

