"""
ABSA (Aspect-Based Sentiment Analysis) 서비스
- BERT 기반 모델을 사용하여 리뷰 텍스트 분석
- 17개 aspect × 3개 sentiment = 51개 클래스 예측
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
import os
import json
import logging
from model_loader import ensure_absa_model

logger = logging.getLogger(__name__)


class ABSAService:
    """ABSA 분석 서비스 클래스"""
    
    def __init__(self, model_path='models/absa'):
        """
        Args:
            model_path: ABSA 모델 파일 경로
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.device = None
        
    def load_model(self):
        """ABSA 모델 및 토크나이저 로딩 (HuggingFace Hub에서 자동 다운로드)"""
        print(f"[ABSA] 모델 로딩 중... ({self.model_path})")
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ABSA] 디바이스: {self.device}")
        if torch.cuda.is_available():
            print(f"[ABSA] GPU: {torch.cuda.get_device_name(0)}")
        
        # HuggingFace에서 모델 다운로드 시도
        logger.info("ABSA 모델 HuggingFace에서 다운로드 시도...")
        hf_model_path = ensure_absa_model(self.model_path)
        actual_model_path = hf_model_path if hf_model_path else self.model_path
        
        if not os.path.exists(actual_model_path):
            raise FileNotFoundError(f"ABSA 모델 디렉토리를 찾을 수 없습니다: {actual_model_path}")
        
        print(f"[ABSA] 모델 경로: {actual_model_path}")
        
        # 모델 로딩
        self.model = BertForSequenceClassification.from_pretrained(actual_model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 토크나이저 로딩
        self.tokenizer = BertTokenizer.from_pretrained(actual_model_path)
        
        # Label 정보 로딩
        config_path = os.path.join(actual_model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.id2label = config["id2label"]
        
        print(f"[ABSA] 모델 로딩 완료 ({len(self.id2label)}개 클래스)")
        return True
    
    def analyze_review(self, text):
        """
        단일 리뷰 텍스트 ABSA 분석
        
        Args:
            text: 리뷰 텍스트 (str)
            
        Returns:
            dict: aspect-sentiment별 확률값
                예: {"맛_긍정": 0.95, "맛_부정": 0.02, "맛_중립": 0.03, ...}
        """
        import time
        
        service_start = time.time()
        text_sample = text[:30] + "..." if len(text) > 30 else text
        logger.info(f"🔍 [ABSA Service] 분석 시작 (텍스트: \"{text_sample}\")")
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("모델이 로딩되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        # 텍스트가 비어있으면 중립 값 반환
        if not text or text.strip() == '':
            logger.info(f"  ⚠️  빈 텍스트 - 중립값 반환")
            return self._get_neutral_absa()
        
        # 단계 1: 토크나이징
        tokenize_start = time.time()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        tokenize_time = time.time() - tokenize_start
        logger.info(f"  ⏱️  토크나이징: {tokenize_time:.3f}s")
        
        # 단계 2: 모델 추론 (BERT) - 가장 중요한 측정 지점!
        inference_start = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        inference_time = time.time() - inference_start
        logger.info(f"  ⏱️  모델 추론 (BERT): {inference_time:.3f}s ⚠️ 핵심 지점!")
        
        # 단계 3: 후처리 (sigmoid)
        postprocess_start = time.time()
        probs = torch.sigmoid(logits)
        postprocess_time = time.time() - postprocess_start
        logger.info(f"  ⏱️  후처리 (sigmoid): {postprocess_time:.3f}s")
        
        # 단계 4: 결과 변환 (dict)
        convert_start = time.time()
        probs_array = probs.cpu().numpy()[0]
        absa_dict = {}
        
        for i, prob in enumerate(probs_array):
            label = self.id2label[str(i)]
            absa_dict[label] = float(prob)
        convert_time = time.time() - convert_start
        logger.info(f"  ⏱️  결과 변환 (dict): {convert_time:.3f}s")
        
        # 전체 소요 시간
        total_time = time.time() - service_start
        logger.info(f"✅ [ABSA Service] 완료 - 총 소요: {total_time:.3f}s")
        
        return absa_dict
    
    def _get_neutral_absa(self):
        """빈 텍스트에 대한 중립 ABSA 반환"""
        neutral_dict = {}
        for i in range(len(self.id2label)):
            label = self.id2label[str(i)]
            # 긍정/부정은 낮게, 중립은 높게
            if "_중립" in label:
                neutral_dict[label] = 0.7
            else:
                neutral_dict[label] = 0.15
        return neutral_dict
    
    def analyze_batch(self, texts, batch_size=32):
        """
        배치 단위로 ABSA 분석 (다중 리뷰 처리)
        
        Args:
            texts: 리뷰 텍스트 리스트 (list of str)
            batch_size: 배치 크기
            
        Returns:
            list of dict: 각 텍스트에 대한 ABSA 결과
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("모델이 로딩되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 토크나이징
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits)
            
            # 결과 변환
            probs_array = probs.cpu().numpy()
            
            for prob_row in probs_array:
                absa_dict = {}
                for j, prob in enumerate(prob_row):
                    label = self.id2label[str(j)]
                    absa_dict[label] = float(prob)
                results.append(absa_dict)
        
        return results


# 전역 서비스 인스턴스 (싱글톤)
_absa_service = None


def get_absa_service():
    """ABSA 서비스 싱글톤"""
    global _absa_service
    if _absa_service is None:
        _absa_service = ABSAService()
        _absa_service.load_model()
    return _absa_service



