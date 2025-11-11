"""
Step 4: DeepFM 모델 학습
- FM Layer + Deep Layer
- MSE Loss, RMSE Metric
- Batch Size: 512, Epochs: 20
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from backend_model.models.deepfm_ranking import DeepFM, DeepFMTrainer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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
    train_dataset = RankingDataset("data/processed/ranking_train.csv")
    valid_dataset = RankingDataset("data/processed/ranking_valid.csv")
    test_dataset = RankingDataset("data/processed/ranking_test.csv")
    
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
    model = DeepFM(
        input_dim=input_dim,
        embed_dim=16,
        hidden_dims=[256, 128, 64]
    )
    
    print(f"  입력 차원: {input_dim}")
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
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/deepfm_ranking.pth")
            print(f"    [BEST] 모델 저장!")
    
    # Best model 로딩
    print("\n[4/5] Best 모델 로딩 중...")
    model.load_state_dict(torch.load("models/deepfm_ranking.pth"))
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
    
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/deepfm_training_curve.png", dpi=100, bbox_inches='tight')
    print(f"\n  학습 곡선 저장: models/deepfm_training_curve.png")
    
    # 샘플 예측 확인
    print("\n  샘플 예측 확인 (Test set 처음 5개):")
    test_data = pd.read_csv("data/processed/ranking_test.csv")
    sample_features = test_dataset.features[:5]
    sample_targets = test_dataset.targets[:5]
    predictions = trainer.predict(sample_features)
    
    for i in range(5):
        print(f"    실제: {sample_targets[i]:.2f} | 예측: {predictions[i]:.2f} | 오차: {abs(sample_targets[i]-predictions[i]):.2f}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] DeepFM 학습 완료!")
    print(f"\nBest Valid RMSE: {best_valid_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"\n모델 저장: models/deepfm_ranking.pth")
    print("다음 단계: scripts/step5_train_multitower.py 실행")
    print("=" * 80)

if __name__ == "__main__":
    train_deepfm()

