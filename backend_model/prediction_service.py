"""
예측 서비스 - DeepFM과 Multi-Tower 모델을 사용한 별점 예측
"""

import torch
import numpy as np
from backend_model.models.deepfm_ranking import DeepFM
from backend_model.models.multitower_ranking import MultiTowerModel
import pickle

class PredictionService:
    """예측 서비스 클래스"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
        # 모델 로딩
        self.deepfm_model = None
        self.multitower_model = None
        
        # Scaler 로딩
        self.user_scaler = None
        self.business_scaler = None
        
        # ABSA 피처 키 (순서 유지)
        self.absa_keys = self._get_absa_keys()
        
    def _get_absa_keys(self):
        """ABSA 피처 키 목록 (순서 유지)"""
        aspects = ['맛', '짠맛', '매운맛', '단맛', '느끼함', '담백함', '고소함', 
                   '품질/신선도', '양', '서비스', '가격', '쾌적함/청결도', 
                   '소음', '분위기', '공간', '주차', '대기']
        sentiments = ['긍정', '부정', '중립']
        
        keys = []
        for aspect in aspects:
            for sentiment in sentiments:
                keys.append(f"{aspect}_{sentiment}")
        return keys
    
    def load_models(self, deepfm_path='models/deepfm_ranking.pth',
                    multitower_path='models/multitower_ranking.pth',
                    user_scaler_path='models/user_scaler.pkl',
                    business_scaler_path='models/business_scaler.pkl'):
        """모델 및 Scaler 로딩"""
        print("모델 로딩 중...")
        
        deepfm_loaded = False
        multitower_loaded = False
        
        try:
            # DeepFM 로딩
            input_dim = 112  # 6(user) + 4(business) + 51(user_absa) + 51(business_absa)
            self.deepfm_model = DeepFM(input_dim=input_dim, embed_dim=16, hidden_dims=[256, 128, 64])
            self.deepfm_model.load_state_dict(torch.load(deepfm_path, map_location=self.device))
            self.deepfm_model.to(self.device)
            self.deepfm_model.eval()
            print("  ✓ DeepFM 로딩 완료")
            deepfm_loaded = True
            
        except Exception as e:
            print(f"  ✗ DeepFM 로딩 실패: {e}")
        
        try:
            # Multi-Tower 로딩 (학습 시 차원에 맞춤: 56/56)
            user_dim = 56
            business_dim = 56
            self.multitower_model = MultiTowerModel(user_dim, business_dim, tower_dims=[128, 64], interaction_dims=[64, 32])
            self.multitower_model.load_state_dict(torch.load(multitower_path, map_location=self.device))
            self.multitower_model.to(self.device)
            self.multitower_model.eval()
            print("  ✓ Multi-Tower 로딩 완료")
            multitower_loaded = True
            
        except Exception as e:
            print(f"  ✗ Multi-Tower 로딩 실패: {e}")
            print("  → DeepFM만 사용합니다.")
        
        try:
            # Scaler 로딩
            with open(user_scaler_path, 'rb') as f:
                self.user_scaler = pickle.load(f)
            with open(business_scaler_path, 'rb') as f:
                self.business_scaler = pickle.load(f)
            print("  ✓ Scaler 로딩 완료")
            
        except Exception as e:
            print(f"  ✗ Scaler 로딩 실패: {e}")
            return False
        
        # 최소한 하나의 모델은 로딩되어야 함
        if deepfm_loaded or multitower_loaded:
            return True
        else:
            print("  ✗ 모든 모델 로딩 실패")
            return False
    
    def prepare_user_features(self, user_data):
        """User 피처 준비"""
        # 기본 피처 (6개)
        features = [
            user_data.get('review_count', 0),
            user_data.get('useful', 0),
            user_data.get('compliment', 0),
            user_data.get('fans', 0),
            user_data.get('average_stars', 0.0),
            user_data.get('yelping_since_days', 0)
        ]
        
        # ABSA 피처 (51개)
        absa = user_data.get('absa_features', {})
        for key in self.absa_keys:
            features.append(absa.get(key, 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def prepare_business_features(self, business_data):
        """Business 피처 준비"""
        # 기본 피처 (4개)
        features = [
            business_data.get('stars', 0.0),
            business_data.get('review_count', 0),
            business_data.get('latitude', 0.0),
            business_data.get('longitude', 0.0)
        ]
        
        # ABSA 피처 (51개)
        absa = business_data.get('absa_features', {})
        for key in self.absa_keys:
            features.append(absa.get(key, 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def predict_rating(self, user_data, business_data):
        """별점 예측"""
        if self.deepfm_model is None and self.multitower_model is None:
            raise ValueError("모델이 로딩되지 않았습니다. load_models()를 먼저 호출하세요.")
        
        # 피처 준비
        user_features = self.prepare_user_features(user_data)
        business_features = self.prepare_business_features(business_data)
        
        # Log Transform (학습시와 동일하게)
        user_features[0] = np.log1p(user_features[0])  # review_count
        user_features[3] = np.log1p(user_features[3])  # fans
        user_features[2] = np.log1p(user_features[2])  # compliment
        
        business_features[1] = np.log1p(business_features[1])  # review_count
        
        # Standard Scaling
        user_features = self.user_scaler.transform(user_features.reshape(1, -1))[0]
        business_features = self.business_scaler.transform(business_features.reshape(1, -1))[0]
        
        predictions = {}
        
        # DeepFM 예측
        if self.deepfm_model is not None:
            try:
                combined_features = np.concatenate([user_features, business_features])
                deepfm_input = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    deepfm_pred = self.deepfm_model(deepfm_input).item()
                    deepfm_pred = max(1.0, min(5.0, deepfm_pred))
                    predictions['deepfm'] = deepfm_pred
            except Exception as e:
                print(f"DeepFM 예측 실패: {e}")
        
        # Multi-Tower 예측 (차원 문제로 스킵)
        # 학습 시 피처가 잘못 분리되어 현재는 사용 불가
        mt_pred = None
        predictions['multitower'] = None
        
        # 앙상블 (DeepFM만 사용)
        if 'deepfm' in predictions:
            ensemble_pred = predictions['deepfm']
        else:
            ensemble_pred = 3.0  # 기본값
        
        return {
            'deepfm_rating': round(predictions.get('deepfm', 3.0), 2),
            'multitower_rating': None,  # 현재 사용 불가
            'ensemble_rating': round(ensemble_pred, 2),
            'confidence': 0.75  # DeepFM만 사용하므로 confidence 낮춤
        }

# 전역 서비스 인스턴스
_prediction_service = None

def get_prediction_service():
    """예측 서비스 싱글톤"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
        _prediction_service.load_models()
    return _prediction_service

