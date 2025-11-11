<!-- dc299d1c-36b6-4f9d-a8d7-d3628c5145da 23ae83a8-c6dc-44bb-8771-31a9d8ea7c95 -->
# DeepFM/Multi-Tower 별점 예측 모델 문제 해결

## 발견된 문제들

### 1. Multi-Tower 모델이 완전히 비활성화됨

`prediction_service.py` 169-172번 줄에서 Multi-Tower 예측이 하드코딩으로 스킵됨

### 2. Multi-Tower 학습 시 피처 분리 문제

`train_models_colab.py` 98-100번 줄:

- 전체 피처를 단순히 절반으로 나누어 User/Business로 분리
- 이는 잘못된 방식! User 피처와 Business 피처가 섞임

### 3. ABSA 피처 키 순서 문제

`prediction_service.py`에서 하드코딩된 ABSA 키 순서가 실제 학습 데이터와 다를 수 있음

### 4. DeepFM이 1.0만 예측하는 문제

- 모델 파일 손상 가능성
- 입력 데이터 전처리 불일치 가능성
- ABSA 피처 순서 불일치

## 해결 방안

### Step 1: 학습 데이터 구조 확인

ranking_train.csv의 실제 컬럼 순서 확인:

- User 기본 피처 6개
- Business 기본 피처 4개
- ABSA 피처 51개 (컬럼명 확인 필요)

### Step 2: ABSA 피처 키 순서 수정

`prediction_service.py`의 `_get_absa_keys()` 함수를:

- 실제 학습 데이터의 ABSA 컬럼 순서와 동일하게 수정
- 또는 학습 시 저장한 피처 순서 파일 사용

### Step 3: Multi-Tower 피처 분리 문제 해결

**옵션 A: Multi-Tower 재학습 (권장)**

- User 피처: 기본 6개 + ABSA 51개 = 57개
- Business 피처: 기본 4개 + ABSA 51개 = 55개
- 제대로 분리하여 재학습

**옵션 B: 예측 시 올바른 분리 적용**

- 현재 모델 파일 그대로 사용
- 예측 시 학습 때와 동일한 방식으로 피처 분리

### Step 4: prediction_service.py 수정

1. ABSA 키 순서 수정
2. Multi-Tower 예측 활성화
3. User/Business 피처 올바르게 분리
4. 디버그 로깅 추가

### Step 5: 테스트

실제 사용자 데이터로 예측 테스트:

- DeepFM 예측값이 1.0이 아닌지 확인
- Multi-Tower 예측값이 제대로 나오는지 확인
- 두 모델의 예측값이 합리적인 범위(1-5)인지 확인

## 중요 체크포인트

1. **ABSA 컬럼 순서**: `data/processed/ranking_train.csv` 확인
2. **Multi-Tower 입력 차원**: 학습 시 사용한 차원 확인
3. **Scaler 일관성**: user_scaler/business_scaler가 학습 시와 동일한지 확인

### To-dos

- [x] Two-Tower 모델 파일 삭제 (two_tower.py, .pth, index.faiss 등)
- [x] 
- [x] 
- [x] 
- [x] 
- [x] 
- [x] 
- [x] 