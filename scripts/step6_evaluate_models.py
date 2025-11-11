"""
Step 6: 모델 평가 및 비교
- DeepFM vs Multi-Tower 성능 비교
- Test Set RMSE 계산
- 예측 샘플 확인
- 앙상블 성능 측정 (옵션)
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from backend_model.models.deepfm_ranking import DeepFM
from backend_model.models.multitower_ranking import MultiTowerModel
import matplotlib.pyplot as plt

class RankingDataset(Dataset):
    """DeepFM용 데이터셋"""
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        
        feature_cols = [col for col in self.data.columns 
                       if col not in ['user_id', 'business_id', 'avg_stars', 'review_count']]
        
        self.features = self.data[feature_cols].values.astype(np.float32)
        self.targets = self.data['avg_stars'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class MultiTowerDataset(Dataset):
    """Multi-Tower용 데이터셋"""
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        
        feature_cols = [col for col in self.data.columns 
                       if col not in ['user_id', 'business_id', 'avg_stars', 'review_count']]
        
        all_features = self.data[feature_cols].values.astype(np.float32)
        mid = len(feature_cols) // 2
        
        self.user_features = all_features[:, :mid]
        self.business_features = all_features[:, mid:]
        self.targets = self.data['avg_stars'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.user_features[idx], self.business_features[idx], self.targets[idx]

def calculate_rmse(predictions, targets):
    """RMSE 계산"""
    return np.sqrt(np.mean((predictions - targets) ** 2))

def calculate_mae(predictions, targets):
    """MAE 계산"""
    return np.mean(np.abs(predictions - targets))

def evaluate_deepfm(model, loader, device):
    """DeepFM 평가"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            predictions = model(batch_x)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.numpy())
    
    return np.array(all_predictions), np.array(all_targets)

def evaluate_multitower(model, loader, device):
    """Multi-Tower 평가"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for user_feat, biz_feat, targets in loader:
            user_feat = user_feat.to(device)
            biz_feat = biz_feat.to(device)
            predictions = model(user_feat, biz_feat)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    return np.array(all_predictions), np.array(all_targets)

def plot_predictions(targets, deepfm_preds, mt_preds, save_path="models/predictions_comparison.png"):
    """예측 결과 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # DeepFM
    axes[0].scatter(targets, deepfm_preds, alpha=0.3, s=10)
    axes[0].plot([1, 5], [1, 5], 'r--', linewidth=2)
    axes[0].set_xlabel('Actual Stars')
    axes[0].set_ylabel('Predicted Stars')
    axes[0].set_title('DeepFM Predictions')
    axes[0].set_xlim(0.5, 5.5)
    axes[0].set_ylim(0.5, 5.5)
    axes[0].grid(True, alpha=0.3)
    
    # Multi-Tower
    axes[1].scatter(targets, mt_preds, alpha=0.3, s=10)
    axes[1].plot([1, 5], [1, 5], 'r--', linewidth=2)
    axes[1].set_xlabel('Actual Stars')
    axes[1].set_ylabel('Predicted Stars')
    axes[1].set_title('Multi-Tower Predictions')
    axes[1].set_xlim(0.5, 5.5)
    axes[1].set_ylim(0.5, 5.5)
    axes[1].grid(True, alpha=0.3)
    
    # 오차 분포
    deepfm_errors = np.abs(targets - deepfm_preds)
    mt_errors = np.abs(targets - mt_preds)
    
    axes[2].hist(deepfm_errors, bins=50, alpha=0.5, label='DeepFM', color='blue')
    axes[2].hist(mt_errors, bins=50, alpha=0.5, label='Multi-Tower', color='orange')
    axes[2].set_xlabel('Absolute Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Error Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"  시각화 저장: {save_path}")

def main():
    print("=" * 80)
    print("Step 6: 모델 평가 및 비교")
    print("=" * 80)
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n디바이스: {device}")
    
    # 데이터 로딩
    print("\n[1/5] 테스트 데이터 로딩 중...")
    test_dataset_deepfm = RankingDataset("data/processed/ranking_test.csv")
    test_dataset_mt = MultiTowerDataset("data/processed/ranking_test.csv")
    
    test_loader_deepfm = DataLoader(test_dataset_deepfm, batch_size=512)
    test_loader_mt = DataLoader(test_dataset_mt, batch_size=512)
    
    print(f"  Test 샘플: {len(test_dataset_deepfm):,}개")
    
    # DeepFM 모델 로딩
    print("\n[2/5] DeepFM 모델 로딩 중...")
    input_dim = test_dataset_deepfm.features.shape[1]
    deepfm_model = DeepFM(input_dim=input_dim, embed_dim=16, hidden_dims=[256, 128, 64])
    deepfm_model.load_state_dict(torch.load("models/deepfm_ranking.pth", map_location=device))
    deepfm_model = deepfm_model.to(device)
    print(f"  [OK] DeepFM 로딩 완료")
    
    # Multi-Tower 모델 로딩
    print("\n[3/5] Multi-Tower 모델 로딩 중...")
    user_dim = test_dataset_mt.user_features.shape[1]
    business_dim = test_dataset_mt.business_features.shape[1]
    mt_model = MultiTowerModel(user_dim, business_dim, tower_dims=[128, 64], interaction_dims=[64, 32])
    mt_model.load_state_dict(torch.load("models/multitower_ranking.pth", map_location=device))
    mt_model = mt_model.to(device)
    print(f"  [OK] Multi-Tower 로딩 완료")
    
    # 평가
    print("\n[4/5] 모델 평가 중...")
    print("\n  DeepFM 평가...")
    deepfm_preds, targets = evaluate_deepfm(deepfm_model, test_loader_deepfm, device)
    deepfm_rmse = calculate_rmse(deepfm_preds, targets)
    deepfm_mae = calculate_mae(deepfm_preds, targets)
    
    print("\n  Multi-Tower 평가...")
    mt_preds, _ = evaluate_multitower(mt_model, test_loader_mt, device)
    mt_rmse = calculate_rmse(mt_preds, targets)
    mt_mae = calculate_mae(mt_preds, targets)
    
    # 앙상블 (평균)
    print("\n  앙상블 평가...")
    ensemble_preds = (deepfm_preds + mt_preds) / 2
    ensemble_rmse = calculate_rmse(ensemble_preds, targets)
    ensemble_mae = calculate_mae(ensemble_preds, targets)
    
    # 결과 출력
    print("\n[5/5] 결과 요약")
    print("\n" + "=" * 80)
    print("평가 결과 (Test Set)")
    print("=" * 80)
    
    print(f"\nDeepFM:")
    print(f"  RMSE: {deepfm_rmse:.4f}")
    print(f"  MAE:  {deepfm_mae:.4f}")
    
    print(f"\nMulti-Tower:")
    print(f"  RMSE: {mt_rmse:.4f}")
    print(f"  MAE:  {mt_mae:.4f}")
    
    print(f"\n앙상블 (평균):")
    print(f"  RMSE: {ensemble_rmse:.4f}")
    print(f"  MAE:  {ensemble_mae:.4f}")
    
    # 최고 성능 모델
    print("\n" + "-" * 80)
    best_model = "DeepFM" if deepfm_rmse < mt_rmse else "Multi-Tower"
    best_rmse = min(deepfm_rmse, mt_rmse)
    
    if ensemble_rmse < best_rmse:
        print(f"[BEST] 최고 성능: 앙상블 (RMSE: {ensemble_rmse:.4f})")
    else:
        print(f"[BEST] 최고 성능: {best_model} (RMSE: {best_rmse:.4f})")
    
    # 샘플 예측 확인
    print("\n샘플 예측 (처음 10개):")
    print("-" * 80)
    print(f"{'실제':>6} {'DeepFM':>8} {'Multi-T':>8} {'앙상블':>8} {'오차(D)':>8} {'오차(M)':>8} {'오차(E)':>8}")
    print("-" * 80)
    
    for i in range(min(10, len(targets))):
        actual = targets[i]
        pred_d = deepfm_preds[i]
        pred_m = mt_preds[i]
        pred_e = ensemble_preds[i]
        err_d = abs(actual - pred_d)
        err_m = abs(actual - pred_m)
        err_e = abs(actual - pred_e)
        
        print(f"{actual:6.2f} {pred_d:8.2f} {pred_m:8.2f} {pred_e:8.2f} {err_d:8.3f} {err_m:8.3f} {err_e:8.3f}")
    
    # 시각화
    print("\n시각화 생성 중...")
    plot_predictions(targets, deepfm_preds, mt_preds)
    
    # 결과 저장
    results_df = pd.DataFrame({
        'actual': targets,
        'deepfm_pred': deepfm_preds,
        'multitower_pred': mt_preds,
        'ensemble_pred': ensemble_preds
    })
    results_df.to_csv("models/test_predictions.csv", index=False, encoding='utf-8-sig')
    print(f"  예측 결과 저장: models/test_predictions.csv")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] 모든 평가 완료!")
    print("=" * 80)
    
    return {
        'deepfm_rmse': deepfm_rmse,
        'multitower_rmse': mt_rmse,
        'ensemble_rmse': ensemble_rmse
    }

if __name__ == "__main__":
    main()

