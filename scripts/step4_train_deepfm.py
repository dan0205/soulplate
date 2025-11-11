"""
Step 4: DeepFM 모델 학습 (코랩용)
- FM Layer + Deep Layer
- MSE Loss, RMSE Metric
- Batch Size: 512, Epochs: 20
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

# ==================== DeepFM 모델 정의 ====================
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
        self.fm_linear = nn.Linear(input_dim, 1)
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
        self.final_linear = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        # FM Part
        fm_linear_part = self.fm_linear(x)
        embeddings = self.fm_embeddings(x)
        sum_of_square = torch.pow(embeddings, 2).sum(dim=1, keepdim=True)
        square_of_sum = torch.pow(embeddings.sum(dim=1, keepdim=True), 2)
        fm_cross_part = 0.5 * (square_of_sum - sum_of_square)
        fm_output = fm_linear_part + fm_cross_part
        
        # Deep Part
        deep_output = self.deep_layers(x)
        deep_output = self.final_linear(deep_output)
        
        # 결합
        output = fm_output + deep_output
        output = torch.sigmoid(output) * 4 + 1  # [0,1] -> [1,5]
        
        return output.squeeze()

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
            
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
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

# ==================== 데이터셋 정의 ====================
class RankingDataset(Dataset):
    """랭킹 데이터셋"""
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        
        # 피처와 타겟 분리
        feature_cols = [col for col in self.data.columns 
                       if col not in ['user_id', 'business_id', 'avg_stars', 'review_count']]
        
        self.features = self.data[feature_cols].values.astype(np.float32)
        self.targets = self.data['avg_stars'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def train_deepfm():
    """DeepFM 모델 학습"""
    print("=" * 80)
    print("Step 4: DeepFM 모델 학습")
    print("=" * 80)
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n디바이스: {device}")
    
    # 데이터 로딩
    print("\n[1/5] 데이터 로딩 중...")
    train_dataset = RankingDataset(f"{DATA_PATH}/ranking_train.csv")
    valid_dataset = RankingDataset(f"{DATA_PATH}/ranking_valid.csv")
    test_dataset = RankingDataset(f"{DATA_PATH}/ranking_test.csv")
    
    print(f"  Train: {len(train_dataset):,}개")
    print(f"  Valid: {len(valid_dataset):,}개")
    print(f"  Test:  {len(test_dataset):,}개")
    print(f"  입력 차원: {train_dataset.features.shape[1]}")
    
    # DataLoader
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델 생성
    print("\n[2/5] DeepFM 모델 생성 중...")
    input_dim = train_dataset.features.shape[1]
    print(f"  입력 차원: {input_dim} (User 피처 + Business 피처 + ABSA 피처 + 텍스트 임베딩 100차원)")
    
    model = DeepFM(
        input_dim=input_dim,
        embed_dim=16,
        hidden_dims=[256, 128, 64]
    )
    
    print(f"  FM 임베딩 차원: 16")
    print(f"  Deep 레이어: [256, 128, 64]")
    print(f"  총 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
    
    # Trainer 생성
    trainer = DeepFMTrainer(model, device=device)
    
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
            torch.save(model.state_dict(), f"{MODEL_PATH}/deepfm_ranking.pth")
            print(f"    [BEST] 모델 저장!")
    
    # Best model 로딩
    print("\n[4/5] Best 모델 로딩 중...")
    model.load_state_dict(torch.load(f"{MODEL_PATH}/deepfm_ranking.pth"))
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
    plt.savefig(f"{MODEL_PATH}/deepfm_training_curve.png", dpi=100, bbox_inches='tight')
    print(f"\n  학습 곡선 저장: {MODEL_PATH}/deepfm_training_curve.png")
    
    # 샘플 예측 확인
    print("\n  샘플 예측 확인 (Test set 처음 5개):")
    test_data = pd.read_csv(f"{DATA_PATH}/ranking_test.csv")
    sample_features = test_dataset.features[:5]
    sample_targets = test_dataset.targets[:5]
    predictions = trainer.predict(sample_features)
    
    for i in range(5):
        print(f"    실제: {sample_targets[i]:.2f} | 예측: {predictions[i]:.2f} | 오차: {abs(sample_targets[i]-predictions[i]):.2f}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] DeepFM 학습 완료!")
    print(f"\nBest Valid RMSE: {best_valid_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"\n모델 저장: {MODEL_PATH}/deepfm_ranking.pth")
    print("다음 단계: scripts/step5_train_multitower.py 실행")
    print("=" * 80)

if __name__ == "__main__":
    train_deepfm()

