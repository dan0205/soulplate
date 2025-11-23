"""
예측 서비스 - DeepFM과 Multi-Tower 모델을 사용한 별점 예측 (309차원)
"""

import torch
import numpy as np
from models.deepfm_ranking import DeepFM
from models.multitower_ranking import MultiTowerModel
from model_loader import ensure_model_file
from utils.text_embedding import get_text_embedding_service
import json
import os
import logging

logger = logging.getLogger(__name__)

class PredictionService:
    """예측 서비스 클래스 (309차원)"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
        # 모델
        self.deepfm_model = None
        self.multitower_model = None
        
        # Scaler 파라미터 (mean, std)
        self.scaler_params = None
        self._load_scaler_params()
        
        # 전역 평균 임베딩 (신규 유저/가게 fallback용)
        self.global_user_avg = None
        self.global_business_avg = None
        self._load_global_avg_embeddings()
        
        # ABSA 피처 키 (순서 유지)
        self.absa_keys = self._get_absa_keys()
        
        # 텍스트 임베딩 서비스 (옵셔널)
        self.text_embedding_service = None
        self._load_text_embedding_service()
    
    def _load_global_avg_embeddings(self):
        """전역 평균 임베딩 로딩"""
        user_path = 'data/global_avg_user_embedding.npy'
        business_path = 'data/global_avg_business_embedding.npy'
        
        try:
            if os.path.exists(user_path):
                self.global_user_avg = np.load(user_path)
                logger.info(f"[Global Avg] User 임베딩 로딩: {self.global_user_avg.shape}")
            else:
                self.global_user_avg = np.zeros(100, dtype=np.float32)
                logger.warning(f"[Global Avg] User 임베딩 없음, 0 벡터")
            
            if os.path.exists(business_path):
                self.global_business_avg = np.load(business_path)
                logger.info(f"[Global Avg] Business 임베딩 로딩: {self.global_business_avg.shape}")
            else:
                self.global_business_avg = np.zeros(100, dtype=np.float32)
                logger.warning(f"[Global Avg] Business 임베딩 없음, 0 벡터")
        except Exception as e:
            logger.error(f"[Global Avg] 로딩 실패: {e}")
            self.global_user_avg = np.zeros(100, dtype=np.float32)
            self.global_business_avg = np.zeros(100, dtype=np.float32)
    
    def _load_text_embedding_service(self):
        """텍스트 임베딩 서비스 로딩 (옵셔널)"""
        try:
            self.text_embedding_service = get_text_embedding_service()
            logger.info("[Text Embedding] ✅ 서비스 로딩 완료")
        except Exception as e:
            logger.warning(f"[Text Embedding] ⚠️ 로딩 실패: {e}")
            logger.warning("[Text Embedding] → 0 벡터로 fallback")
            self.text_embedding_service = None
    
    def _load_scaler_params(self):
        """Scaler 파라미터 로딩 (309d 버전)"""
        scaler_path = 'models/scaler_params_309d.json'
        
        logger.info(f"[Scaler] 로딩: {scaler_path}")
        
        # HuggingFace에서 다운로드 시도 (루트 디렉토리에서)
        hf_path = ensure_model_file("scaler_params_309d.json", scaler_path)
        
        if hf_path and os.path.exists(hf_path):
            with open(hf_path, 'r') as f:
                self.scaler_params = json.load(f)
            logger.info(f"[Scaler] ✅ HuggingFace에서 로딩: {hf_path}")
        elif os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                self.scaler_params = json.load(f)
            logger.info(f"[Scaler] ✅ 로컬에서 로딩: {scaler_path}")
        else:
            logger.error(f"[Scaler] ❌ 파일 없음: {scaler_path}")
            self.scaler_params = None
    
    def _get_absa_keys(self):
        """ABSA 피처 키 목록 (51개)"""
        aspects = ['맛', '짠맛', '매운맛', '단맛', '느끼함', '담백함', '고소함', 
                   '품질/신선도', '양', '서비스', '가격', '쾌적함/청결도', 
                   '소음', '분위기', '공간', '주차', '대기']
        sentiments = ['긍정', '부정', '중립']
        return [f"{a}_{s}" for a in aspects for s in sentiments]
    
    def load_models(self):
        """모델 로딩 (DeepFM 309d, Multi-Tower 309d)"""
        logger.info("=" * 60)
        logger.info("모델 로딩 시작 (309차원)")
        logger.info("=" * 60)
        
        deepfm_loaded = False
        multitower_loaded = False
        
        # DeepFM 로딩
        try:
            deepfm_path = 'models/deepfm_ranking_309d.pth'
            hf_path = ensure_model_file("deepfm_ranking_309d.pth", deepfm_path)
            actual_path = hf_path if hf_path else deepfm_path
            
            self.deepfm_model = DeepFM(input_dim=309, embed_dim=16, hidden_dims=[256, 128, 64])
            self.deepfm_model.load_state_dict(torch.load(actual_path, map_location=self.device))
            self.deepfm_model.to(self.device)
            self.deepfm_model.eval()
            deepfm_loaded = True
            logger.info(f"✓ DeepFM 로딩 완료: {actual_path}")
        except Exception as e:
            logger.error(f"✗ DeepFM 로딩 실패: {e}")
        
        # Multi-Tower 로딩
        try:
            mt_path = 'models/multitower_ranking_309d.pth'
            hf_path = ensure_model_file("multitower_ranking_309d.pth", mt_path)
            actual_path = hf_path if hf_path else mt_path
            
            self.multitower_model = MultiTowerModel(
                user_input_dim=156,  # User: 100 + 5 + 51
                business_input_dim=153,  # Business: 100 + 2 + 51
                tower_dims=[128, 64],
                interaction_dims=[64, 32]
            )
            self.multitower_model.load_state_dict(torch.load(actual_path, map_location=self.device))
            self.multitower_model.to(self.device)
            self.multitower_model.eval()
            multitower_loaded = True
            logger.info(f"✓ Multi-Tower 로딩 완료: {actual_path}")
        except Exception as e:
            logger.error(f"✗ Multi-Tower 로딩 실패: {e}")
        
        if deepfm_loaded or multitower_loaded:
            logger.info("=" * 60)
            return True
        else:
            logger.error("모든 모델 로딩 실패")
            return False
    
    def prepare_combined_features(self, user_data, business_data, review_text=None):
        """
        309차원 피처 준비 (User-Business 분리)
        
        === User 피처 (156차원) ===
        1. User 텍스트 임베딩 (100)
        2. User 통계 (5): review_count, useful, compliment, fans, average_stars (log+scaled)
        3. User ABSA (51)
        
        === Business 피처 (153차원) ===
        4. Business 텍스트 임베딩 (100)
        5. Business 통계 (2): review_count, stars (log+scaled)
        6. Business ABSA (51)
        
        User Tower: 0-155 (156차원)
        Business Tower: 156-308 (153차원)
        """
        features = []
        
        # === User 피처 (156) ===
        # 1. User 텍스트 임베딩 (100)
        if 'text_embedding' in user_data and user_data['text_embedding']:
            user_text_emb = np.array(user_data['text_embedding'], dtype=np.float32)
        else:
            user_text_emb = self.global_user_avg
        features.extend(user_text_emb.tolist())
        
        # 2. User 통계 (5) - Log + Scaled
        user_review_count = user_data.get('review_count', 0)
        useful = user_data.get('useful', 0)
        compliment = user_data.get('compliment', 0)
        fans = user_data.get('fans', 0)
        average_stars = user_data.get('average_stars', 0.0)
        
        user_review_count_log = np.log1p(user_review_count)
        useful_log = np.log1p(useful)
        compliment_log = np.log1p(compliment)
        fans_log = np.log1p(fans)
        
        if not self.scaler_params:
            raise ValueError("scaler_params가 없습니다")
        
        user_params = self.scaler_params['user']
        user_review_count_scaled = (user_review_count_log - user_params['user_review_count_log']['mean']) / user_params['user_review_count_log']['std']
        useful_scaled = (useful_log - user_params['useful_log']['mean']) / user_params['useful_log']['std']
        compliment_scaled = (compliment_log - user_params['compliment_log']['mean']) / user_params['compliment_log']['std']
        fans_scaled = (fans_log - user_params['fans_log']['mean']) / user_params['fans_log']['std']
        average_stars_scaled = (average_stars - user_params['average_stars']['mean']) / user_params['average_stars']['std']
        
        features.extend([
            user_review_count_scaled,
            useful_scaled,
            compliment_scaled,
            fans_scaled,
            average_stars_scaled
        ])
        
        # 3. User ABSA (51)
        user_absa = user_data.get('absa_features', {})
        for key in self.absa_keys:
            features.append(user_absa.get(key, 0.0))
        
        # === Business 피처 (153) ===
        # 4. Business 텍스트 임베딩 (100)
        if 'text_embedding' in business_data and business_data['text_embedding']:
            business_text_emb = np.array(business_data['text_embedding'], dtype=np.float32)
        else:
            business_text_emb = self.global_business_avg
        features.extend(business_text_emb.tolist())
        
        # 5. Business 통계 (2) - Log + Scaled
        business_review_count = business_data.get('review_count', 0)
        stars = business_data.get('stars', 0.0)
        
        business_review_count_log = np.log1p(business_review_count)
        
        business_params = self.scaler_params['business']
        business_review_count_scaled = (business_review_count_log - business_params['business_review_count_log']['mean']) / business_params['business_review_count_log']['std']
        stars_scaled = (stars - business_params['business_stars']['mean']) / business_params['business_stars']['std']
        
        features.extend([
            business_review_count_scaled,
            stars_scaled
        ])
        
        # 6. Business ABSA (51)
        business_absa = business_data.get('absa_features', {})
        for key in self.absa_keys:
            features.append(business_absa.get(key, 0.0))
        
        final_features = np.array(features, dtype=np.float32)
        logger.info(f"[피처] 309차원 생성 완료 (User: 156, Business: 153)")
        
        return final_features
    
    def predict_rating(self, user_data, business_data, review_text=None):
        """별점 예측 (309차원)"""
        if not self.deepfm_model and not self.multitower_model:
            raise ValueError("모델이 로딩되지 않았습니다")
        
        # 309차원 피처 준비
        combined_features = self.prepare_combined_features(user_data, business_data, review_text)
        
        predictions = {}
        
        # DeepFM 예측
        if self.deepfm_model:
            try:
                deepfm_input = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    deepfm_pred = self.deepfm_model(deepfm_input).item()
                    predictions['deepfm'] = max(1.0, min(5.0, deepfm_pred))
            except Exception as e:
                logger.error(f"DeepFM 예측 실패: {e}")
        
        # Multi-Tower 예측 (156 + 153)
        if self.multitower_model:
            try:
                user_tower_input = combined_features[:156]
                business_tower_input = combined_features[156:]
                
                mt_user_input = torch.FloatTensor(user_tower_input).unsqueeze(0).to(self.device)
                mt_business_input = torch.FloatTensor(business_tower_input).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    mt_pred = self.multitower_model(mt_user_input, mt_business_input).item()
                    predictions['multitower'] = max(1.0, min(5.0, mt_pred))
            except Exception as e:
                logger.error(f"Multi-Tower 예측 실패: {e}")
        
        # 앙상블
        if 'deepfm' in predictions and 'multitower' in predictions:
            ensemble = (predictions['deepfm'] * 0.5) + (predictions['multitower'] * 0.5)
            confidence = 0.9
        elif 'deepfm' in predictions:
            ensemble = predictions['deepfm']
            confidence = 0.7
        elif 'multitower' in predictions:
            ensemble = predictions['multitower']
            confidence = 0.7
        else:
            raise ValueError("모든 모델 예측 실패")
        
        return {
            'deepfm_rating': predictions.get('deepfm'),
            'multitower_rating': predictions.get('multitower'),
            'ensemble_rating': ensemble,
            'confidence': confidence
        }

# ==================== 싱글톤 ====================
_prediction_service = None

def get_prediction_service():
    """예측 서비스 싱글톤 (309차원)"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
        _prediction_service.load_models()
    return _prediction_service

