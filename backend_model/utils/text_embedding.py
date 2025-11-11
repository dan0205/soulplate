"""
텍스트 임베딩 유틸리티
- TF-IDF Vectorizer를 사용하여 리뷰 텍스트를 벡터로 변환
- 학습과 실제 서비스에서 공통으로 사용
"""

import pickle
import numpy as np
import os

class TextEmbeddingService:
    """텍스트 임베딩 서비스 클래스"""
    
    def __init__(self, vectorizer_path='models/tfidf_vectorizer.pkl'):
        """
        Args:
            vectorizer_path: TF-IDF Vectorizer 파일 경로
        """
        self.vectorizer_path = vectorizer_path
        self.vectorizer = None
        
    def load_vectorizer(self):
        """Vectorizer 로딩"""
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer 파일을 찾을 수 없습니다: {self.vectorizer_path}")
        
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"[OK] TF-IDF Vectorizer 로딩 완료 (어휘 크기: {len(self.vectorizer.vocabulary_)})")
        return True
    
    def transform_text(self, text):
        """
        단일 텍스트를 임베딩으로 변환
        
        Args:
            text: 리뷰 텍스트 (str)
            
        Returns:
            embedding: 100차원 벡터 (numpy array)
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer가 로딩되지 않았습니다. load_vectorizer()를 먼저 호출하세요.")
        
        # 텍스트가 None이거나 빈 문자열이면 0 벡터 반환
        if not text or text.strip() == '':
            return np.zeros(100, dtype=np.float32)
        
        # TF-IDF 변환
        tfidf_vector = self.vectorizer.transform([text])
        
        # Dense array로 변환
        embedding = tfidf_vector.toarray()[0].astype(np.float32)
        
        return embedding
    
    def transform_texts(self, texts):
        """
        여러 텍스트를 임베딩으로 변환
        
        Args:
            texts: 리뷰 텍스트 리스트 (list of str)
            
        Returns:
            embeddings: (N, 100) 형태의 numpy array
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer가 로딩되지 않았습니다. load_vectorizer()를 먼저 호출하세요.")
        
        # 빈 텍스트 처리
        processed_texts = [text if text and text.strip() else '' for text in texts]
        
        # TF-IDF 변환
        tfidf_matrix = self.vectorizer.transform(processed_texts)
        
        # Dense array로 변환
        embeddings = tfidf_matrix.toarray().astype(np.float32)
        
        return embeddings
    
    def get_average_embedding(self, texts):
        """
        여러 텍스트의 평균 임베딩 계산
        
        Args:
            texts: 리뷰 텍스트 리스트 (list of str)
            
        Returns:
            avg_embedding: 100차원 평균 벡터 (numpy array)
        """
        if not texts or len(texts) == 0:
            return np.zeros(100, dtype=np.float32)
        
        embeddings = self.transform_texts(texts)
        avg_embedding = embeddings.mean(axis=0)
        
        return avg_embedding

# 전역 서비스 인스턴스
_text_embedding_service = None

def get_text_embedding_service():
    """텍스트 임베딩 서비스 싱글톤"""
    global _text_embedding_service
    if _text_embedding_service is None:
        _text_embedding_service = TextEmbeddingService()
        _text_embedding_service.load_vectorizer()
    return _text_embedding_service


