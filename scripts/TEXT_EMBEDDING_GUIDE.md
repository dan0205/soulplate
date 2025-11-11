# 텍스트 임베딩 추가 가이드

리뷰 텍스트를 TF-IDF로 임베딩하여 DeepFM과 Multi-Tower 모델 학습 및 서비스에 추가

## 📋 개요

- **목적**: 리뷰 텍스트를 모델 학습에 포함하여 예측 성능 향상
- **방법**: TF-IDF Vectorizer로 100차원 벡터 변환
- **적용 모델**: DeepFM, Multi-Tower

## 🔄 실행 순서

### 1. TF-IDF 임베딩 생성
```bash
python scripts/step2_create_text_embeddings.py
```

**입력**:
- `data/raw/review_100k_absa_with_text.csv`

**출력**:
- `data/processed/review_text_embeddings.csv` (review_id + 100개 임베딩 컬럼)
- `models/tfidf_vectorizer.pkl` (TF-IDF Vectorizer 모델)

**처리 내용**:
- 10만개 리뷰 텍스트를 TF-IDF로 변환
- max_features=100 (100차원)
- min_df=5 (최소 5개 문서에 등장)
- ngram_range=(1, 2) (unigram + bigram)

---

### 2. 학습 데이터 생성
```bash
python scripts/step3_create_ranking_data.py
```

**입력**:
- `data/processed/review_absa_features.csv`
- `data/processed/review_text_embeddings.csv` (새로 생성됨)
- `data/processed/user_preprocessed.csv`
- `data/processed/business_preprocessed.csv`

**출력**:
- `data/processed/ranking_train.csv` (80%)
- `data/processed/ranking_valid.csv` (10%)
- `data/processed/ranking_test.csv` (10%)

**변경 사항**:
- User-Business 쌍별 평균 텍스트 임베딩 계산
- 기존 피처 + 텍스트 임베딩 100차원 추가
- 총 컬럼: user_id, business_id, avg_stars, review_count + 피처들 + text_embed_0~99

---

### 3. DeepFM 모델 학습 (코랩)
```bash
python scripts/step4_train_deepfm.py
```

**입력 차원 변경**:
- 기존: 112차원 (6 User + 4 Business + 51 User ABSA + 51 Business ABSA)
- **신규: 212차원** (기존 112 + 텍스트 임베딩 100)

**출력**:
- `models/deepfm_ranking.pth`
- `models/deepfm_training_curve.png`

---

### 4. Multi-Tower 모델 학습 (코랩)
```bash
python scripts/step5_train_multitower.py
```

**입력 차원 변경**:
- **User Tower**: 106차원 (6 User + 51 ABSA + 50 텍스트)
- **Business Tower**: 105차원 (4 Business + 51 ABSA + 50 텍스트)
  - 참고: 학습 데이터에서 전체 피처를 절반씩 분할

**출력**:
- `models/multitower_ranking.pth`
- `models/multitower_training_curve.png`

---

### 5. 실제 서비스 사용

#### 5.1 모델 로딩
```python
from backend_model.prediction_service import get_prediction_service

service = get_prediction_service()
# 자동으로 TF-IDF Vectorizer도 로딩됨
```

#### 5.2 예측 방법

**방법 1: 미리 계산된 임베딩 사용 (권장)**
```python
user_data = {
    'review_count': 10,
    'useful': 5,
    'compliment': 2,
    'fans': 1,
    'average_stars': 4.2,
    'yelping_since_days': 1000,
    'absa_features': {...},
    'text_embedding': [0.1, 0.2, ...]  # 100차원 미리 계산된 값
}

business_data = {
    'stars': 4.5,
    'review_count': 100,
    'latitude': 37.5,
    'longitude': -122.4,
    'absa_features': {...},
    'text_embedding': [0.3, 0.4, ...]  # 100차원 미리 계산된 값
}

result = service.predict_rating(user_data, business_data)
```

**방법 2: 실시간 텍스트 임베딩 계산**
```python
user_data = {
    'review_count': 10,
    'useful': 5,
    'compliment': 2,
    'fans': 1,
    'average_stars': 4.2,
    'yelping_since_days': 1000,
    'absa_features': {...},
    'review_texts': ['맛있어요', '서비스 좋아요', ...]  # 리뷰 텍스트 리스트
}

business_data = {
    'stars': 4.5,
    'review_count': 100,
    'latitude': 37.5,
    'longitude': -122.4,
    'absa_features': {...},
    'review_texts': ['분위기 좋음', '가격 적당', ...]
}

result = service.predict_rating(user_data, business_data)
```

#### 5.3 예측 결과
```python
{
    'deepfm_rating': 4.32,
    'multitower_rating': 4.28,
    'ensemble_rating': 4.30,
    'confidence': 0.95
}
```

---

## 📊 피처 구성

### DeepFM (212차원)
```
User 피처 (6개):
  - review_count, useful, compliment, fans, average_stars, yelping_since_days

Business 피처 (4개):
  - stars, review_count, latitude, longitude

ABSA 피처 (102개):
  - User ABSA (51개)
  - Business ABSA (51개)

텍스트 임베딩 (100개):
  - User-Business 쌍의 평균 TF-IDF 벡터
```

### Multi-Tower

**User Tower (106차원)**:
```
- 기본 User 피처 (6개)
- User ABSA (51개)
- User 평균 텍스트 임베딩 (50개)
  * 실제로는 전체 212차원을 절반으로 분할
```

**Business Tower (105차원)**:
```
- 기본 Business 피처 (4개)
- Business ABSA (51개)
- Business 평균 텍스트 임베딩 (50개)
  * 실제로는 전체 212차원을 절반으로 분할
```

---

## 🔧 텍스트 임베딩 유틸리티

### 독립적으로 사용하기
```python
from backend_model.utils.text_embedding import TextEmbeddingService

# 서비스 초기화
text_service = TextEmbeddingService('models/tfidf_vectorizer.pkl')
text_service.load_vectorizer()

# 단일 텍스트 변환
text = "맛있어요 분위기도 좋고"
embedding = text_service.transform_text(text)
print(embedding.shape)  # (100,)

# 여러 텍스트 변환
texts = ["맛있어요", "서비스 좋아요", "가격 저렴"]
embeddings = text_service.transform_texts(texts)
print(embeddings.shape)  # (3, 100)

# 평균 임베딩 계산
avg_embedding = text_service.get_average_embedding(texts)
print(avg_embedding.shape)  # (100,)
```

---

## 📁 파일 구조

```
project/
├── data/
│   ├── raw/
│   │   └── review_100k_absa_with_text.csv  # 입력
│   └── processed/
│       ├── review_text_embeddings.csv      # [NEW] 생성됨
│       ├── ranking_train.csv               # 업데이트됨
│       ├── ranking_valid.csv               # 업데이트됨
│       └── ranking_test.csv                # 업데이트됨
├── models/
│   ├── tfidf_vectorizer.pkl                # [NEW] 생성됨
│   ├── deepfm_ranking.pth                  # 업데이트됨
│   └── multitower_ranking.pth              # 업데이트됨
├── scripts/
│   ├── step2_create_text_embeddings.py     # [NEW]
│   ├── step3_create_ranking_data.py        # 수정됨
│   ├── step4_train_deepfm.py               # 수정됨
│   └── step5_train_multitower.py           # [NEW]
└── backend_model/
    ├── prediction_service.py               # 수정됨
    └── utils/
        ├── __init__.py                     # [NEW]
        └── text_embedding.py               # [NEW]
```

---

## ⚠️ 주의사항

### 1. 코랩에서 학습 시
- GPU 사용 가능 시 자동으로 CUDA 사용
- 배치 크기: 512
- 에폭: 20

### 2. 차원 불일치 주의
- DeepFM: 정확히 212차원 입력 필요
- Multi-Tower: User 106차원, Business 105차원

### 3. 텍스트가 없는 경우
- 0 벡터로 자동 처리
- 서비스에 영향 없음

### 4. DB 스키마 권장사항
```sql
-- User 테이블에 평균 텍스트 임베딩 저장 (선택)
ALTER TABLE users ADD COLUMN text_embedding FLOAT[];

-- Business 테이블에 평균 텍스트 임베딩 저장 (선택)
ALTER TABLE businesses ADD COLUMN text_embedding FLOAT[];
```

미리 계산해서 저장하면 실시간 예측 속도가 빨라집니다.

---

## 🚀 성능 향상 팁

1. **텍스트 임베딩 캐싱**: DB에 미리 계산해서 저장
2. **배치 예측**: 여러 예측을 한 번에 처리
3. **GPU 사용**: 코랩에서 학습 시 GPU 런타임 사용

---

## 📝 체크리스트

학습 전:
- [ ] `data/raw/review_100k_absa_with_text.csv` 파일 존재 확인
- [ ] `data/processed/review_absa_features.csv` 파일 존재 확인

학습 후:
- [ ] `models/tfidf_vectorizer.pkl` 생성 확인
- [ ] `models/deepfm_ranking.pth` 업데이트 확인
- [ ] `models/multitower_ranking.pth` 생성 확인

서비스 배포 전:
- [ ] 모든 모델 파일을 서버로 복사
- [ ] prediction_service 테스트
- [ ] 텍스트 임베딩 로딩 확인

---

## 🐛 문제 해결

### "Vectorizer 파일을 찾을 수 없습니다"
→ `step2_create_text_embeddings.py`를 먼저 실행하세요.

### "입력 차원이 맞지 않습니다"
→ 모델을 재학습하거나, 피처 차원을 확인하세요.

### "텍스트 임베딩이 0 벡터입니다"
→ `review_texts` 또는 `text_embedding`을 제공하지 않은 경우 정상입니다.

---

**작성일**: 2025-11-11
**버전**: 1.0

