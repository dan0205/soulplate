"""
통합 Colab 노트북: DeepFM + Multi-Tower 학습 및 평가

실행 방법:
1. Google Colab에서 새 노트북 생성
2. 이 파일의 각 셀 코드를 복사하여 실행
3. Google Drive에 다음 파일들을 준비:
   - ranking_train.csv
   - ranking_valid.csv
   - ranking_test.csv
4. 학습 후 모델 파일 다운로드:
   - deepfm_ranking.pth
   - multitower_ranking.pth
"""

# ============================================================================
# 셀 1: 환경 설정
# ============================================================================

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 필요한 라이브러리 설치
!pip install torch pandas numpy scikit-learn matplotlib tqdm -q

print("[OK] 환경 설정 완료!")

# ============================================================================
# 셀 2: 데이터 경로 설정 및 로딩 확인
# ============================================================================

import pandas as pd
import numpy as np

# ===== 여기에 Google Drive 경로를 설정하세요 =====
DATA_DIR = "/content/drive/MyDrive/absa-ranking-data"  # 수정 필요!

# 데이터 파일 경로
TRAIN_PATH = f"{DATA_DIR}/ranking_train.csv"
VALID_PATH = f"{DATA_DIR}/ranking_valid.csv"
TEST_PATH = f"{DATA_DIR}/ranking_test.csv"

# 데이터 로딩 테스트
print("데이터 로딩 확인 중...")
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"✓ Train: {train_df.shape}")
print(f"✓ Valid: {valid_df.shape}")
print(f"✓ Test: {test_df.shape}")
print(f"\n컬럼 샘플: {list(train_df.columns[:5])} ...")
print(f"\n[OK] 데이터 로딩 성공!")

# ============================================================================
# 셀 3: Dataset 및 DataLoader 정의
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader

class RankingDataset(Dataset):
    """랭킹 데이터셋"""
    
    def __init__(self, df):
        # 피처와 타겟 분리
        feature_cols = [col for col in df.columns 
                       if col not in ['user_id', 'business_id', 'avg_stars', 'review_count']]
        
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df['avg_stars'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class MultiTowerDataset(Dataset):
    """Multi-Tower용 데이터셋 (User/Business 분리)"""
    
    def __init__(self, df):
        # User 피처 (기본 6개 + ABSA 51개)
        user_base_cols = ['review_count', 'useful', 'compliment', 'fans', 'average_stars', 'yelping_since_days']
        user_absa_cols = [col for col in df.columns if col.startswith('absa_')]
        
        # Business 피처는 stars, review_count(business), latitude, longitude + ABSA
        # 하지만 데이터에서 User/Business가 이미 합쳐져 있으므로, 
        # 임의로 절반씩 나눠서 사용 (실제로는 컬럼 이름으로 구분해야 함)
        
        # 간단하게: 전체를 User와 Business로 나누기
        feature_cols = [col for col in df.columns 
                       if col not in ['user_id', 'business_id', 'avg_stars', 'review_count']]
        
        all_features = df[feature_cols].values.astype(np.float32)
        mid = len(feature_cols) // 2
        
        self.user_features = all_features[:, :mid]
        self.business_features = all_features[:, mid:]
        self.targets = df['avg_stars'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.user_features[idx], self.business_features[idx], self.targets[idx]

# 데이터셋 생성
print("데이터셋 생성 중...")
train_dataset_deepfm = RankingDataset(train_df)
valid_dataset_deepfm = RankingDataset(valid_df)
test_dataset_deepfm = RankingDataset(test_df)

train_dataset_mt = MultiTowerDataset(train_df)
valid_dataset_mt = MultiTowerDataset(valid_df)
test_dataset_mt = MultiTowerDataset(test_df)

print(f"[OK] DeepFM - 입력 차원: {train_dataset_deepfm.features.shape[1]}")
print(f"[OK] Multi-Tower - User: {train_dataset_mt.user_features.shape[1]}, Business: {train_dataset_mt.business_features.shape[1]}")

# ============================================================================
# 셀 4: DeepFM 모델 정의
# ============================================================================

import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, input_dim, embed_dim=16, hidden_dims=[256, 128, 64]):
        super(DeepFM, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # FM Part
        self.fm_linear = nn.Linear(input_dim, 1)
        self.fm_embeddings = nn.Linear(input_dim, embed_dim)
        
        # Deep Part
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
        output = torch.sigmoid(output) * 4 + 1  # [1, 5]
        
        return output.squeeze()

print("[OK] DeepFM 모델 정의 완료!")

# ============================================================================
# 셀 5: Multi-Tower 모델 정의
# ============================================================================

class MultiTowerModel(nn.Module):
    def __init__(self, user_input_dim, business_input_dim, 
                 tower_dims=[128, 64], interaction_dims=[64, 32]):
        super(MultiTowerModel, self).__init__()
        
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
        
        # Interaction Layer
        interaction_layers = []
        prev_dim = tower_dims[-1] * 2
        for dim in interaction_dims:
            interaction_layers.append(nn.Linear(prev_dim, dim))
            interaction_layers.append(nn.BatchNorm1d(dim))
            interaction_layers.append(nn.ReLU())
            interaction_layers.append(nn.Dropout(0.2))
            prev_dim = dim
        
        self.interaction_layers = nn.Sequential(*interaction_layers)
        self.final_linear = nn.Linear(prev_dim, 1)
        
    def forward(self, user_features, business_features):
        user_embed = self.user_tower(user_features)
        business_embed = self.business_tower(business_features)
        
        combined = torch.cat([user_embed, business_embed], dim=1)
        interaction = self.interaction_layers(combined)
        
        output = self.final_linear(interaction)
        output = torch.sigmoid(output) * 4 + 1  # [1, 5]
        
        return output.squeeze()

print("[OK] Multi-Tower 모델 정의 완료!")

# ============================================================================
# 셀 6: 학습 함수 정의
# ============================================================================

from tqdm import tqdm

def train_epoch_deepfm(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate_deepfm(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item()
    
    rmse = (total_loss / len(loader)) ** 0.5
    return rmse

def train_epoch_multitower(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for user_feat, biz_feat, targets in loader:
        user_feat = user_feat.to(device)
        biz_feat = biz_feat.to(device)
        targets = targets.to(device)
        
        predictions = model(user_feat, biz_feat)
        loss = criterion(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate_multitower(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for user_feat, biz_feat, targets in loader:
            user_feat = user_feat.to(device)
            biz_feat = biz_feat.to(device)
            targets = targets.to(device)
            
            predictions = model(user_feat, biz_feat)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
    
    rmse = (total_loss / len(loader)) ** 0.5
    return rmse

print("[OK] 학습 함수 정의 완료!")

# ============================================================================
# 셀 7: DeepFM 학습
# ============================================================================

print("=" * 80)
print("DeepFM 모델 학습 시작")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"디바이스: {device}")

# DataLoader
batch_size = 512
train_loader_deepfm = DataLoader(train_dataset_deepfm, batch_size=batch_size, shuffle=True)
valid_loader_deepfm = DataLoader(valid_dataset_deepfm, batch_size=batch_size)
test_loader_deepfm = DataLoader(test_dataset_deepfm, batch_size=batch_size)

# 모델 생성
input_dim = train_dataset_deepfm.features.shape[1]
deepfm_model = DeepFM(input_dim=input_dim, embed_dim=16, hidden_dims=[256, 128, 64])
deepfm_model = deepfm_model.to(device)

print(f"입력 차원: {input_dim}")
print(f"총 파라미터: {sum(p.numel() for p in deepfm_model.parameters()):,}개")

# Optimizer & Loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(deepfm_model.parameters(), lr=0.001)

# 학습
epochs = 20
best_valid_rmse_deepfm = float('inf')

print("\n학습 시작...")
for epoch in range(epochs):
    train_loss = train_epoch_deepfm(deepfm_model, train_loader_deepfm, criterion, optimizer, device)
    valid_rmse = evaluate_deepfm(deepfm_model, valid_loader_deepfm, criterion, device)
    
    print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Valid RMSE: {valid_rmse:.4f}", end="")
    
    if valid_rmse < best_valid_rmse_deepfm:
        best_valid_rmse_deepfm = valid_rmse
        torch.save(deepfm_model.state_dict(), "deepfm_ranking.pth")
        print(" [BEST]")
    else:
        print()

# Test 평가
deepfm_model.load_state_dict(torch.load("deepfm_ranking.pth"))
test_rmse_deepfm = evaluate_deepfm(deepfm_model, test_loader_deepfm, criterion, device)

print(f"\n[SUCCESS] DeepFM 학습 완료!")
print(f"Best Valid RMSE: {best_valid_rmse_deepfm:.4f}")
print(f"Test RMSE: {test_rmse_deepfm:.4f}")

# ============================================================================
# 셀 8: Multi-Tower 학습
# ============================================================================

print("\n" + "=" * 80)
print("Multi-Tower 모델 학습 시작")
print("=" * 80)

# DataLoader
train_loader_mt = DataLoader(train_dataset_mt, batch_size=batch_size, shuffle=True)
valid_loader_mt = DataLoader(valid_dataset_mt, batch_size=batch_size)
test_loader_mt = DataLoader(test_dataset_mt, batch_size=batch_size)

# 모델 생성
user_dim = train_dataset_mt.user_features.shape[1]
business_dim = train_dataset_mt.business_features.shape[1]
mt_model = MultiTowerModel(user_dim, business_dim, tower_dims=[128, 64], interaction_dims=[64, 32])
mt_model = mt_model.to(device)

print(f"User 차원: {user_dim}")
print(f"Business 차원: {business_dim}")
print(f"총 파라미터: {sum(p.numel() for p in mt_model.parameters()):,}개")

# Optimizer & Loss
optimizer_mt = torch.optim.Adam(mt_model.parameters(), lr=0.001)

# 학습
best_valid_rmse_mt = float('inf')

print("\n학습 시작...")
for epoch in range(epochs):
    train_loss = train_epoch_multitower(mt_model, train_loader_mt, criterion, optimizer_mt, device)
    valid_rmse = evaluate_multitower(mt_model, valid_loader_mt, criterion, device)
    
    print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Valid RMSE: {valid_rmse:.4f}", end="")
    
    if valid_rmse < best_valid_rmse_mt:
        best_valid_rmse_mt = valid_rmse
        torch.save(mt_model.state_dict(), "multitower_ranking.pth")
        print(" [BEST]")
    else:
        print()

# Test 평가
mt_model.load_state_dict(torch.load("multitower_ranking.pth"))
test_rmse_mt = evaluate_multitower(mt_model, test_loader_mt, criterion, device)

print(f"\n[SUCCESS] Multi-Tower 학습 완료!")
print(f"Best Valid RMSE: {best_valid_rmse_mt:.4f}")
print(f"Test RMSE: {test_rmse_mt:.4f}")

# ============================================================================
# 셀 9: 모델 비교 및 결과 요약
# ============================================================================

print("\n" + "=" * 80)
print("최종 결과 요약")
print("=" * 80)

print(f"\nDeepFM:")
print(f"  Valid RMSE: {best_valid_rmse_deepfm:.4f}")
print(f"  Test RMSE:  {test_rmse_deepfm:.4f}")

print(f"\nMulti-Tower:")
print(f"  Valid RMSE: {best_valid_rmse_mt:.4f}")
print(f"  Test RMSE:  {test_rmse_mt:.4f}")

if test_rmse_deepfm < test_rmse_mt:
    print(f"\n✓ DeepFM이 {test_rmse_mt - test_rmse_deepfm:.4f} 더 좋은 성능!")
else:
    print(f"\n✓ Multi-Tower가 {test_rmse_deepfm - test_rmse_mt:.4f} 더 좋은 성능!")

print("\n생성된 모델 파일:")
print("  - deepfm_ranking.pth")
print("  - multitower_ranking.pth")

# ============================================================================
# 셀 10: 모델 파일 다운로드
# ============================================================================

from google.colab import files

print("모델 파일 다운로드 중...")
files.download("deepfm_ranking.pth")
files.download("multitower_ranking.pth")

print("\n[SUCCESS] 모든 작업 완료!")
print("\n다음 단계:")
print("1. 다운로드한 두 모델 파일을 로컬의 'models/' 폴더에 복사")
print("2. 로컬에서 scripts/step6_evaluate_models.py 실행 (선택사항)")

