"""
예측 서비스 - DeepFM과 Multi-Tower 모델을 사용한 별점 예측
- 텍스트 임베딩 포함(체크)
"""

import torch
import numpy as np
from models.deepfm_ranking import DeepFM
from models.multitower_ranking import MultiTowerModel
from utils.text_embedding import TextEmbeddingService
from model_loader import ensure_model_file
import pickle
import json
import os
import logging

logger = logging.getLogger(__name__)

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
        
        # Scaler 파라미터 (mean, std)
        self.scaler_params = None
        self._load_scaler_params()
        
        # 텍스트 임베딩 서비스
        self.text_embedding_service = None
        
        # ABSA 피처 키 (순서 유지)
        self.absa_keys = self._get_absa_keys()
    
    def _load_scaler_params(self):
        """Scaler 파라미터 로딩 (mean, std)"""
        scaler_path = 'models/scaler_params.json'
        
        # HuggingFace에서 다운로드 시도
        hf_path = ensure_model_file("models/scaler_params.json", scaler_path)
        
        if hf_path and os.path.exists(hf_path):
            with open(hf_path, 'r') as f:
                self.scaler_params = json.load(f)
        elif os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                self.scaler_params = json.load(f)
        else:
            print(f"  [WARNING] Scaler params 파일 없음: {scaler_path}")
            self.scaler_params = None
        
    def _get_absa_keys(self):
        """ABSA 피처 키 목록 (순서 유지) - 학습 데이터와 동일한 순서"""
        # 저장된 ABSA 컬럼 순서 로드
        absa_file = 'models/absa_columns.json'
        if os.path.exists(absa_file):
            with open(absa_file, 'r', encoding='utf-8') as f:
                absa_info = json.load(f)
                # "absa_" prefix 제거
                keys = [col.replace('absa_', '') for col in absa_info['user_absa_columns']]
                return keys
        
        # 파일이 없으면 기본값 사용 (하드코딩)
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
                    business_scaler_path='models/business_scaler.pkl',
                    vectorizer_path='models/tfidf_vectorizer.pkl'):
        """모델 및 Scaler 로딩 (HuggingFace Hub에서 자동 다운로드)"""
        print("모델 로딩 중...")
        
        deepfm_loaded = False
        multitower_loaded = False
        
        try:
            # DeepFM 로딩 - HuggingFace에서 다운로드 시도
            logger.info("DeepFM 모델 로딩 시도...")
            deepfm_hf_path = ensure_model_file("models/deepfm_ranking.pth", deepfm_path)
            actual_deepfm_path = deepfm_hf_path if deepfm_hf_path else deepfm_path
            
            if not os.path.exists(actual_deepfm_path):
                raise FileNotFoundError(f"DeepFM 모델 파일을 찾을 수 없습니다: {actual_deepfm_path}")
            
            # 모델이 212차원으로 학습됨 (실제 데이터는 210이지만 모델 재학습 필요)
            input_dim = 212
            self.deepfm_model = DeepFM(input_dim=input_dim, embed_dim=16, hidden_dims=[256, 128, 64])
            self.deepfm_model.load_state_dict(torch.load(actual_deepfm_path, map_location=self.device))
            self.deepfm_model.to(self.device)
            self.deepfm_model.eval()
            print("  [OK] DeepFM 로딩 완료 (입력 차원: 212, 패딩 2개 추가 필요)")
            deepfm_loaded = True
            
        except Exception as e:
            print(f"  [ERROR] DeepFM 로딩 실패: {e}")
        
        try:
            # Multi-Tower 로딩 - HuggingFace에서 다운로드 시도
            logger.info("Multi-Tower 모델 로딩 시도...")
            multitower_hf_path = ensure_model_file("models/multitower_ranking.pth", multitower_path)
            actual_multitower_path = multitower_hf_path if multitower_hf_path else multitower_path
            
            if not os.path.exists(actual_multitower_path):
                raise FileNotFoundError(f"Multi-Tower 모델 파일을 찾을 수 없습니다: {actual_multitower_path}")
            
            # 모델이 106, 106으로 학습됨 (실제는 105, 105이지만 모델 재학습 필요)
            user_dim = 106
            business_dim = 106
            self.multitower_model = MultiTowerModel(user_dim, business_dim, tower_dims=[128, 64], interaction_dims=[64, 32])
            self.multitower_model.load_state_dict(torch.load(actual_multitower_path, map_location=self.device))
            self.multitower_model.to(self.device)
            self.multitower_model.eval()
            print("  [OK] Multi-Tower 로딩 완료 (User: 106차원, Business: 106차원, 패딩 1개씩 추가 필요)")
            multitower_loaded = True
            
        except Exception as e:
            print(f"  [ERROR] Multi-Tower 로딩 실패: {e}")
            print("  [INFO] DeepFM만 사용합니다.")
        
        try:
            # Scaler 로딩 (선택적)
            if os.path.exists(user_scaler_path) and os.path.exists(business_scaler_path):
                with open(user_scaler_path, 'rb') as f:
                    self.user_scaler = pickle.load(f)
                with open(business_scaler_path, 'rb') as f:
                    self.business_scaler = pickle.load(f)
                print("  [OK] Scaler 로딩 완료")
            else:
                print("  [WARNING] Scaler 파일 없음 - 스케일링 건너뜀")
                self.user_scaler = None
                self.business_scaler = None
            
        except Exception as e:
            print(f"  [WARNING] Scaler 로딩 실패: {e} - 스케일링 건너뜀")
            self.user_scaler = None
            self.business_scaler = None
        
        try:
            # 텍스트 임베딩 서비스 로딩 - HuggingFace에서 다운로드 시도
            logger.info("텍스트 임베딩 서비스 로딩 시도...")
            vectorizer_hf_path = ensure_model_file("models/tfidf_vectorizer.pkl", vectorizer_path)
            actual_vectorizer_path = vectorizer_hf_path if vectorizer_hf_path else vectorizer_path
            
            self.text_embedding_service = TextEmbeddingService(actual_vectorizer_path)
            self.text_embedding_service.load_vectorizer()
            
        except Exception as e:
            print(f"  [ERROR] 텍스트 임베딩 서비스 로딩 실패: {e}")
            print("  [WARNING] 텍스트 피처 없이 진행합니다.")
            self.text_embedding_service = None
        
        # 최소한 하나의 모델은 로딩되어야 함
        if deepfm_loaded or multitower_loaded:
            return True
        else:
            print("  [ERROR] 모든 모델 로딩 실패")
            return False
    
    def prepare_combined_features(self, user_data, business_data, review_text=None):
        """
        학습 데이터와 동일한 형식으로 전체 피처 준비
        
        학습 데이터 구조:
        [텍스트 임베딩 100개] + [useful, compliment, fans, average_stars, yelping_since_days, 
         stars, latitude, longitude] + [User ABSA 51개 + Business ABSA 51개] = 210개
        
        Args:
            user_data: User 데이터 (dict)
            business_data: Business 데이터 (dict)
            review_text: User-Business 쌍의 리뷰 텍스트 (optional)
        
        Returns:
            전체 210개 피처
        """
        features = []
        
        # 1. 텍스트 임베딩 (100개) - 맨 앞!
        if self.text_embedding_service is not None:
            # 방법 1: 리뷰 텍스트가 제공된 경우
            if review_text:
                text_emb = self.text_embedding_service.get_average_embedding([review_text])
            # 방법 2: User의 평균 임베딩 사용 (fallback)
            elif 'text_embedding' in user_data and user_data['text_embedding'] is not None:
                text_emb = np.array(user_data['text_embedding'], dtype=np.float32)
            # 방법 3: 텍스트가 없으면 0 벡터
            else:
                text_emb = np.zeros(100, dtype=np.float32)
            
            features.extend(text_emb.tolist())
        else:
            # 텍스트 임베딩 서비스가 없으면 0 벡터
            features.extend([0.0] * 100)
        
        # 2. 기본 피처 (8개) - review_count 제외! + Log Transform + Standard Scaling
        # User 피처 (5개)
        useful = user_data.get('useful', 0)
        compliment = user_data.get('compliment', 0)
        fans = user_data.get('fans', 0)
        average_stars = user_data.get('average_stars', 0.0)
        yelping_since_days = user_data.get('yelping_since_days', 0)
        
        # Log Transform (학습 시와 동일)
        compliment_log = np.log1p(compliment)
        fans_log = np.log1p(fans)
        
        # Standard Scaling
        if self.scaler_params:
            user_params = self.scaler_params['user']
            useful_scaled = (useful - user_params['useful']['mean']) / user_params['useful']['std']
            compliment_scaled = (compliment_log - user_params['compliment']['mean']) / user_params['compliment']['std']
            fans_scaled = (fans_log - user_params['fans']['mean']) / user_params['fans']['std']
            average_stars_scaled = (average_stars - user_params['average_stars']['mean']) / user_params['average_stars']['std']
            yelping_since_days_scaled = (yelping_since_days - user_params['yelping_since_days']['mean']) / user_params['yelping_since_days']['std']
        else:
            # Scaler 없으면 원본 값 사용 (비권장)
            useful_scaled = useful
            compliment_scaled = compliment_log
            fans_scaled = fans_log
            average_stars_scaled = average_stars
            yelping_since_days_scaled = yelping_since_days
        
        # Business 피처 (3개) - review_count 제외!
        stars = business_data.get('stars', 0.0)
        latitude = business_data.get('latitude', 0.0)
        longitude = business_data.get('longitude', 0.0)
        
        # Standard Scaling
        if self.scaler_params:
            business_params = self.scaler_params['business']
            stars_scaled = (stars - business_params['stars']['mean']) / business_params['stars']['std']
            latitude_scaled = (latitude - business_params['latitude']['mean']) / business_params['latitude']['std']
            longitude_scaled = (longitude - business_params['longitude']['mean']) / business_params['longitude']['std']
        else:
            # Scaler 없으면 원본 값 사용 (비권장)
            stars_scaled = stars
            latitude_scaled = latitude
            longitude_scaled = longitude
        
        features.extend([
            useful_scaled, compliment_scaled, fans_scaled, average_stars_scaled, yelping_since_days_scaled,
            stars_scaled, latitude_scaled, longitude_scaled
        ])
        
        # 3. User ABSA 피처 (51개)
        user_absa = user_data.get('absa_features', {})
        for key in self.absa_keys:
            features.append(user_absa.get(key, 0.0))
        
        # 4. Business ABSA 피처 (51개)
        business_absa = business_data.get('absa_features', {})
        for key in self.absa_keys:
            features.append(business_absa.get(key, 0.0))
        
        # 5. 패딩 (모델이 212차원으로 학습됨, 현재 210개이므로 2개 추가)
        # TODO: 모델을 210차원으로 재학습하면 이 부분 제거 가능
        features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def predict_rating(self, user_data, business_data, review_text=None):
        """
        별점 예측
        
        Args:
            user_data: User 데이터 (dict)
                필수: review_count, useful, compliment, fans, average_stars, yelping_since_days, absa_features
            business_data: Business 데이터 (dict)
                필수: stars, review_count, latitude, longitude, absa_features
            review_text: User-Business 쌍의 리뷰 텍스트 (optional)
        
        Returns:
            dict: 예측 결과
                - deepfm_rating: DeepFM 예측값
                - multitower_rating: Multi-Tower 예측값 (없으면 None)
                - ensemble_rating: 앙상블 예측값
                - confidence: 신뢰도
        """
        if self.deepfm_model is None and self.multitower_model is None:
            raise ValueError("모델이 로딩되지 않았습니다. load_models()를 먼저 호출하세요.")
        
        # 전체 피처 준비 (212개)
        combined_features = self.prepare_combined_features(user_data, business_data, review_text)
        
        print(f"[DEBUG] Combined features shape: {combined_features.shape}")
        print(f"[DEBUG] Combined features stats: min={combined_features.min():.4f}, max={combined_features.max():.4f}, mean={combined_features.mean():.4f}")
        print(f"[DEBUG] Non-zero features: {np.count_nonzero(combined_features)}/212")
        print(f"[DEBUG] First 10 features: {combined_features[:10]}")
        
        predictions = {}
        
        # DeepFM 예측 (전체 212개 피처 사용)
        if self.deepfm_model is not None:
            try:
                deepfm_input = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    deepfm_pred = self.deepfm_model(deepfm_input).item()
                    deepfm_pred = max(1.0, min(5.0, deepfm_pred))
                    predictions['deepfm'] = deepfm_pred
                    print(f"[DEBUG] DeepFM 예측: {deepfm_pred:.2f}")
            except Exception as e:
                print(f"DeepFM 예측 실패: {e}")
                import traceback
                traceback.print_exc()
        
        # Multi-Tower 예측 (212개 피처를 절반으로 나눔)
        if self.multitower_model is not None:
            try:
                # 전체 212개 피처를 절반으로 나눔 (106, 106)
                mid = len(combined_features) // 2
                user_tower_input = combined_features[:mid]
                business_tower_input = combined_features[mid:]
                
                mt_user_input = torch.FloatTensor(user_tower_input).unsqueeze(0).to(self.device)
                mt_business_input = torch.FloatTensor(business_tower_input).unsqueeze(0).to(self.device)
                
                print(f"[DEBUG] Multi-Tower input shapes: user={mt_user_input.shape}, business={mt_business_input.shape}")
                
                with torch.no_grad():
                    mt_pred = self.multitower_model(mt_user_input, mt_business_input).item()
                    mt_pred = max(1.0, min(5.0, mt_pred))
                    predictions['multitower'] = mt_pred
                    print(f"[DEBUG] Multi-Tower 예측: {mt_pred:.2f}")
            except Exception as e:
                print(f"Multi-Tower 예측 실패: {e}")
                import traceback
                traceback.print_exc()
                predictions['multitower'] = None
        else:
            predictions['multitower'] = None
        
        # 앙상블 (두 모델의 평균)
        if 'deepfm' in predictions and predictions.get('multitower') is not None:
            ensemble_pred = (predictions['deepfm'] + predictions['multitower']) / 2
            confidence = 0.95
        elif 'deepfm' in predictions:
            ensemble_pred = predictions['deepfm']
            confidence = 0.75
        elif predictions.get('multitower') is not None:
            ensemble_pred = predictions['multitower']
            confidence = 0.75
        else:
            ensemble_pred = 3.0  # 기본값
            confidence = 0.5
        
        return {
            'deepfm_rating': round(predictions.get('deepfm', 3.0), 2),
            'multitower_rating': round(predictions['multitower'], 2) if predictions.get('multitower') is not None else None,
            'ensemble_rating': round(ensemble_pred, 2),
            'confidence': confidence
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

