"""
Two-Tower 모델 학습 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time

from backend_model.models.two_tower import TwoTowerModel

# 설정
PROCESSED_DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3

print(f"Using device: {DEVICE}")

class YelpDataset(Dataset):
    """Yelp 데이터셋"""
    
    def __init__(self, data_df, users_df, businesses_df, user_id_to_idx, business_id_to_idx):
        self.data = data_df.reset_index(drop=True)
        self.users_df = users_df.set_index('user_id')
        self.businesses_df = businesses_df.set_index('business_id')
        self.user_id_to_idx = user_id_to_idx
        self.business_id_to_idx = business_id_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        user_id = row['user_id']
        business_id = row['business_id']
        label = row['label']
        
        # User index
        user_idx = self.user_id_to_idx[user_id]
        
        # User features
        user_row = self.users_df.loc[user_id]
        user_features = [
            user_row['age'] / 100.0,  # Normalize
            user_row['review_count'] / 100.0,
            user_row['useful'] / 50.0,
            user_row['average_stars'] / 5.0
        ]
        
        # Item index
        item_idx = self.business_id_to_idx[business_id]
        
        # Item features
        item_row = self.businesses_df.loc[business_id]
        item_features = [
            item_row['stars'] / 5.0,
            item_row['review_count'] / 500.0,
            float(item_row['is_open'])
        ]
        
        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'user_features': torch.tensor(user_features, dtype=torch.float32),
            'item_idx': torch.tensor(item_idx, dtype=torch.long),
            'item_features': torch.tensor(item_features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

def load_data():
    """데이터 로드"""
    print("Loading data...")
    
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    valid_df = pd.read_csv(PROCESSED_DATA_DIR / "valid.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    users_df = pd.read_csv(PROCESSED_DATA_DIR / "users.csv")
    businesses_df = pd.read_csv(PROCESSED_DATA_DIR / "businesses.csv")
    
    with open(PROCESSED_DATA_DIR / "user_id_to_idx.json", 'r') as f:
        user_id_to_idx = json.load(f)
    
    with open(PROCESSED_DATA_DIR / "business_id_to_idx.json", 'r') as f:
        business_id_to_idx = json.load(f)
    
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    print(f"Users: {len(users_df)}, Businesses: {len(businesses_df)}")
    
    return train_df, valid_df, test_df, users_df, businesses_df, user_id_to_idx, business_id_to_idx

def create_dataloaders(train_df, valid_df, test_df, users_df, businesses_df, 
                       user_id_to_idx, business_id_to_idx):
    """DataLoader 생성"""
    print("Creating dataloaders...")
    
    train_dataset = YelpDataset(train_df, users_df, businesses_df, user_id_to_idx, business_id_to_idx)
    valid_dataset = YelpDataset(valid_df, users_df, businesses_df, user_id_to_idx, business_id_to_idx)
    test_dataset = YelpDataset(test_df, users_df, businesses_df, user_id_to_idx, business_id_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, valid_loader, test_loader

def train_epoch(model, dataloader, optimizer, criterion):
    """1 에폭 학습"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        user_idx = batch['user_idx'].to(DEVICE)
        user_features = batch['user_features'].to(DEVICE)
        item_idx = batch['item_idx'].to(DEVICE)
        item_features = batch['item_features'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        
        # Forward pass
        scores = model(user_idx, user_features, item_idx, item_features)
        loss = criterion(scores, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record
        total_loss += loss.item()
        
        # For AUC calculation
        with torch.no_grad():
            preds = torch.sigmoid(scores)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc

def evaluate(model, dataloader, criterion):
    """평가"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            user_idx = batch['user_idx'].to(DEVICE)
            user_features = batch['user_features'].to(DEVICE)
            item_idx = batch['item_idx'].to(DEVICE)
            item_features = batch['item_features'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Forward pass
            scores = model(user_idx, user_features, item_idx, item_features)
            loss = criterion(scores, labels)
            
            total_loss += loss.item()
            
            # For AUC calculation
            preds = torch.sigmoid(scores)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc

def save_model(model, path, metadata=None):
    """모델 저장"""
    print(f"Saving model to {path}...")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }
    torch.save(checkpoint, path)

def train_model():
    """전체 학습 파이프라인"""
    print("=" * 60)
    print("Two-Tower Model Training")
    print("=" * 60)
    
    # Load data
    train_df, valid_df, test_df, users_df, businesses_df, user_id_to_idx, business_id_to_idx = load_data()
    
    # Create dataloaders
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_df, valid_df, test_df, users_df, businesses_df, 
        user_id_to_idx, business_id_to_idx
    )
    
    # Create model
    print("\nCreating model...")
    num_users = len(users_df)
    num_items = len(businesses_df)
    
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64,
        hidden_dim=256,
        output_dim=128
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training Start")
    print("=" * 60)
    
    best_valid_auc = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        valid_loss, valid_auc = evaluate(model, valid_loader, criterion)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}, Valid AUC: {valid_auc:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Early stopping
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            patience_counter = 0
            
            # Save best model
            save_model(
                model, 
                MODEL_DIR / "best_two_tower_model.pth",
                metadata={
                    'epoch': epoch + 1,
                    'valid_auc': valid_auc,
                    'num_users': num_users,
                    'num_items': num_items
                }
            )
            print(f"  [BEST] New best model saved! (Valid AUC: {valid_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("\nEarly stopping triggered!")
                break
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("Test Evaluation")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(MODEL_DIR / "best_two_tower_model.pth", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_auc = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save individual towers
    print("\n" + "=" * 60)
    print("Saving Individual Towers")
    print("=" * 60)
    
    torch.save(model.user_tower.state_dict(), MODEL_DIR / "user_tower.pth")
    print(f"Saved user_tower.pth")
    
    torch.save(model.item_tower.state_dict(), MODEL_DIR / "item_tower.pth")
    print(f"Saved item_tower.pth")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Best Validation AUC: {best_valid_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Models saved to: {MODEL_DIR}")

if __name__ == "__main__":
    train_model()

