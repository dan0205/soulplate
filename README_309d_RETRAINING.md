# 309차원 모델 재학습 가이드

## 개요

이 문서는 추천 모델을 309차원으로 재학습하고 배포하는 전체 과정을 설명합니다.

## 피처 구성 (309차원)

### 1. User 텍스트 임베딩 (100차원)
- 유저가 작성한 모든 리뷰의 TF-IDF 평균
- 신규 유저: 전역 평균 임베딩 사용

### 2. Business 텍스트 임베딩 (100차원)
- 가게에 달린 모든 리뷰의 TF-IDF 평균
- 신규 가게: 전역 평균 임베딩 사용

### 3. User 통계 (5차원)
- `review_count` (log + scaled)
- `useful` (log + scaled)
- `compliment` (log + scaled)
- `fans` (log + scaled)
- `average_stars` (scaled)

### 4. Business 통계 (2차원)
- `review_count` (log + scaled)
- `stars` (scaled)

### 5. User ABSA (51차원)
- 17개 aspect × 3개 sentiment = 51개

### 6. Business ABSA (51차원)
- 17개 aspect × 3개 sentiment = 51개

**총 309차원** (100 + 100 + 5 + 2 + 51 + 51)

## 실행 순서

### Phase 1: 데이터베이스 준비

#### 1. User 테이블에 age, gender 컬럼 추가
```bash
python scripts/add_age_gender_columns.py
```

**작업 내용:**
- `users` 테이블에 `age` (INTEGER), `gender` (VARCHAR(10)) 컬럼 추가
- `businesses` 테이블에 `text_embedding` (JSONB) 컬럼 추가
- 기존 데이터는 NULL로 유지 (현재 모델 학습에 미사용)

#### 2. Business 텍스트 임베딩 생성
```bash
python scripts/generate_business_text_embeddings.py
```

**작업 내용:**
- 모든 Business의 리뷰 텍스트 수집
- TF-IDF 벡터화 (100차원)
- Business.text_embedding 업데이트
- 예상 소요 시간: ~2시간 (14,000개 Business)

#### 3. 전역 평균 임베딩 계산
```bash
python scripts/calculate_global_avg_embeddings.py
```

**작업 내용:**
- User 텍스트 임베딩 평균 → `data/global_avg_user_embedding.npy`
- Business 텍스트 임베딩 평균 → `data/global_avg_business_embedding.npy`
- 신규 유저/가게 예측 시 fallback으로 사용

### Phase 2: 학습 데이터 생성

#### 4. 309차원 학습 데이터 생성
```bash
python scripts/create_training_data_309d.py
```

**입력:**
- `data/raw/review_100k_absa_with_text.csv`
- `data/processed/user_filtered.csv`
- `data/processed/business_filtered.csv`

**출력:**
- `data/training/ranking_train_309d.csv`
- `data/training/ranking_valid_309d.csv`
- `data/training/ranking_test_309d.csv`
- `data/training/scaler_params_309d.json`
- `data/training/tfidf_vectorizer_309d.pkl`

**작업 내용:**
- User-Business 쌍별 피처 집계
- Log 변환 및 Standard Scaling
- Train(70%) / Valid(15%) / Test(15%) 분할

### Phase 3: 모델 학습 (Google Colab)

#### 5. DeepFM 학습
```bash
# Google Colab에서 실행
# scripts/colab_train_deepfm_309d.py 업로드 및 실행
```

**모델 구조:**
- Input: 309차원
- FM Embedding: 16차원
- Deep Layers: [256, 128, 64]
- Output: 1~5 별점

**하이퍼파라미터:**
- Batch Size: 512
- Learning Rate: 0.001
- Epochs: 100 (Early Stopping: patience=10)

**출력:**
- `models/deepfm_ranking_309d.pth`
- `models/deepfm_309d_training_curve.png`

#### 6. Multi-Tower 학습
```bash
# Google Colab에서 실행
# scripts/colab_train_multitower_309d.py 업로드 및 실행
```

**모델 구조:**
- User Tower Input: 154차원
- Business Tower Input: 155차원
- Tower Layers: [128, 64]
- Interaction Layers: [64, 32]
- Output: 1~5 별점

**하이퍼파라미터:**
- Batch Size: 512
- Learning Rate: 0.001
- Epochs: 100 (Early Stopping: patience=10)

**출력:**
- `models/multitower_ranking_309d.pth`
- `models/multitower_309d_training_curve.png`

### Phase 4: HuggingFace 업로드

학습 스크립트 실행 후 HuggingFace 업로드 프롬프트에서 'y' 입력

**업로드 파일:**
1. `deepfm_ranking_309d.pth`
2. `multitower_ranking_309d.pth`
3. `scaler_params_309d.json`
4. `tfidf_vectorizer_309d.pkl`

### Phase 5: 배포 및 검증

#### 7. 모델 검증
```bash
python scripts/validate_309d_model.py
```

**검증 항목:**
- 309차원 피처 생성 확인
- DeepFM/Multi-Tower 예측 테스트
- 예측값 범위 검증 (1.0~5.0)

#### 8. 예측 서비스 업데이트

**새 파일:**
- `backend_model/prediction_service_309d.py` (309차원용)

**기존 서비스 대체 방법:**
```python
# backend_model/__init__.py 또는 main.py에서
from backend_model.prediction_service_309d import PredictionService

pred_service = PredictionService()
pred_service.load_models()
```

## 주요 변경사항

### 1. 패딩 제거
- **이전:** 210차원 → 212차원 (2개 패딩)
- **현재:** 309차원 (패딩 없음)

### 2. Business 텍스트 임베딩 추가
- **이전:** User 텍스트 임베딩만 사용
- **현재:** User + Business 텍스트 임베딩 (대칭 구조)

### 3. 피처 스케일링 개선
- **이전:** `useful` - 스케일링만
- **현재:** `useful` - Log 변환 + 스케일링

### 4. 제외된 피처
- `yelping_since_days` (가입 경과일) - 예측 기여도 낮음
- `latitude`, `longitude` (위도, 경도) - 학습에 무의미

### 5. 신규 유저/가게 처리
- **이전:** 0 벡터 사용
- **현재:** 전역 평균 임베딩 사용 (더 나은 Cold Start 성능)

## 파일 구조

```
.
├── backend_web/
│   └── models.py                          # User/Business 테이블 (age, gender, text_embedding 추가)
├── backend_model/
│   ├── prediction_service_309d.py         # 309차원 예측 서비스
│   ├── model_loader.py                    # HuggingFace 다운로드
│   └── models/
│       ├── deepfm_ranking.py             # DeepFM 모델 정의
│       └── multitower_ranking.py         # Multi-Tower 모델 정의
├── scripts/
│   ├── add_age_gender_columns.py         # [Phase 1-1] DB 스키마 업데이트
│   ├── generate_business_text_embeddings.py  # [Phase 1-2] Business 임베딩 생성
│   ├── calculate_global_avg_embeddings.py    # [Phase 1-3] 전역 평균 계산
│   ├── create_training_data_309d.py      # [Phase 2] 학습 데이터 생성
│   ├── colab_train_deepfm_309d.py        # [Phase 3-1] DeepFM 학습 (Colab)
│   ├── colab_train_multitower_309d.py    # [Phase 3-2] Multi-Tower 학습 (Colab)
│   └── validate_309d_model.py            # [Phase 5] 검증
├── data/
│   ├── global_avg_user_embedding.npy     # 전역 평균 User 임베딩
│   ├── global_avg_business_embedding.npy # 전역 평균 Business 임베딩
│   └── training/
│       ├── ranking_train_309d.csv
│       ├── ranking_valid_309d.csv
│       ├── ranking_test_309d.csv
│       ├── scaler_params_309d.json       # 스케일링 파라미터
│       └── tfidf_vectorizer_309d.pkl     # TF-IDF 벡터라이저
└── models/
    ├── deepfm_ranking_309d.pth           # DeepFM 모델
    └── multitower_ranking_309d.pth       # Multi-Tower 모델
```

## 트러블슈팅

### 1. DB 연결 실패
```bash
# .env 파일에서 DATABASE_URL 확인
# postgresql://postgres:CIirVNgnCPOhbljkayXakvLFttNSodnu@interchange.proxy.rlwy.net:52092/railway
```

### 2. 메모리 부족
```python
# create_training_data_309d.py 수정
# 배치 처리 크기 줄이기
batch_size = 100  # 기본값: 500
```

### 3. HuggingFace 업로드 실패
```bash
# 토큰 확인: https://huggingface.co/settings/tokens
# 수동 업로드: https://huggingface.co/yidj/soulplate-models/upload/main
```

### 4. 피처 차원 불일치
```bash
# 검증 스크립트로 확인
python scripts/validate_309d_model.py

# 로그에서 각 세그먼트 차원 확인
# [  0:100] User 텍스트 임베딩 (100개)
# [100:200] Business 텍스트 임베딩 (100개)
# [200:205] User 통계 (5개)
# [205:207] Business 통계 (2개)
# [207:258] User ABSA (51개)
# [258:309] Business ABSA (51개)
```

## 성능 비교

### 예상 성능 (학습 후 업데이트)

| 모델 | 이전 (212d) | 현재 (309d) | 개선 |
|------|------------|------------|------|
| DeepFM RMSE | TBD | TBD | TBD |
| Multi-Tower RMSE | TBD | TBD | TBD |
| Ensemble RMSE | TBD | TBD | TBD |

### 추론 속도

- DeepFM: ~5ms/request
- Multi-Tower: ~7ms/request
- Ensemble: ~12ms/request

## 향후 계획

### 1. age, gender 활용
- 충분한 데이터 수집 후 모델에 추가
- 예상 차원: 309 + 2 = 311차원

### 2. 지역 정보 개선
- 위도/경도 대신 지역 카테고리 임베딩
- 예: '강남', '홍대', '이태원' 등

### 3. 시계열 피처
- 리뷰 작성 시간 패턴
- 최근 리뷰 가중치

## 참고 자료

- [DeepFM 논문](https://arxiv.org/abs/1703.04247)
- [Multi-Tower 아키텍처](https://developers.google.com/machine-learning/recommendation/dnn/multi-tower)
- [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

## 문의

이슈 발생 시 로그 파일과 함께 문의:
- `backend_model/logs/prediction_service.log`
- `scripts/logs/training.log`

