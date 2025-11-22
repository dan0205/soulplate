# 🎓 Phase 3: Google Colab 모델 학습 가이드

## 📌 개요

Phase 1-2가 완료되면 생성되는 학습 데이터를 Google Colab에 업로드하여 모델을 학습합니다.

---

## 🔧 Phase 1-2 완료 후 필요한 파일들

### 1. 학습 데이터 (Phase 2에서 생성)
```
data/training/
├── ranking_train_309d.csv          # 학습 데이터 (~70%)
├── ranking_valid_309d.csv          # 검증 데이터 (~15%)
├── ranking_test_309d.csv           # 테스트 데이터 (~15%)
├── scaler_params_309d.json         # 스케일링 파라미터
└── tfidf_vectorizer_309d.pkl       # TF-IDF 벡터라이저
```

### 2. 학습 스크립트
```
scripts/
├── colab_train_deepfm_309d.py      # DeepFM 학습
└── colab_train_multitower_309d.py  # Multi-Tower 학습
```

---

## 📋 Phase 3 단계별 가이드

### 🔹 Step 1: Google Drive 폴더 준비

1. **Google Drive 접속**
   - https://drive.google.com 접속

2. **폴더 생성**
   ```
   MyDrive/
   └── soulplate_training/          # 이름은 자유롭게 변경 가능
       ├── data/
       │   └── training/
       └── scripts/
   ```

3. **파일 업로드**
   
   **data/training/ 폴더에 업로드:**
   - `ranking_train_309d.csv` (가장 큰 파일, ~1-2GB)
   - `ranking_valid_309d.csv`
   - `ranking_test_309d.csv`
   - `scaler_params_309d.json`
   - `tfidf_vectorizer_309d.pkl`
   
   **scripts/ 폴더에 업로드:**
   - `colab_train_deepfm_309d.py`
   - `colab_train_multitower_309d.py`

---

### 🔹 Step 2: Google Colab 노트북 생성

1. **Colab 접속**
   - https://colab.research.google.com 접속

2. **새 노트북 생성**
   - "파일" → "새 노트북"

3. **GPU 런타임 설정** ⚡ (중요!)
   - "런타임" → "런타임 유형 변경"
   - 하드웨어 가속기: **GPU** 선택
   - GPU 유형: **T4** (무료) 또는 **V100/A100** (Colab Pro)
   - "저장" 클릭

---

### 🔹 Step 3: DeepFM 모델 학습

#### 3-1. Google Drive 마운트

첫 번째 셀에 입력:
```python
from google.colab import drive
drive.mount('/content/drive')
```

실행하면 Google 계정 인증 요구 → 허용

#### 3-2. 작업 디렉토리 설정

두 번째 셀에 입력 (폴더 경로를 본인의 경로로 수정):
```python
import os
os.chdir('/content/drive/MyDrive/soulplate_training')
print("현재 작업 디렉토리:", os.getcwd())

# 파일 존재 확인
print("\n데이터 파일:")
!ls -lh data/training/

print("\n스크립트 파일:")
!ls -lh scripts/
```

#### 3-3. DeepFM 학습 실행

세 번째 셀에 입력:
```python
!python scripts/colab_train_deepfm_309d.py
```

**실행 결과 예시:**
```
================================================================================
DeepFM 309차원 학습
================================================================================

디바이스: cuda

[1/5] 데이터 로딩 중...
  피처 shape: (309,)
  타겟 shape: (42225,)
  Train: 29,557개
  Valid: 6,334개
  Test:  6,334개

[2/5] DeepFM 모델 생성 중...
  입력 차원: 309
  FM 임베딩 차원: 16
  Deep 레이어: [256, 128, 64]
  총 파라미터: 157,889개

[3/5] 모델 학습 중...
  Epoch   1/100 | Train Loss: 1.2345 | Valid RMSE: 1.1234
    [BEST] 모델 저장! RMSE: 1.1234
  Epoch   2/100 | Train Loss: 1.1234 | Valid RMSE: 1.0987
    [BEST] 모델 저장! RMSE: 1.0987
  ...
  Early stopping at epoch 45

[4/5] Best 모델 로딩 중...

[5/5] Test 평가 중...
  Test RMSE: 0.9876

  학습 곡선 저장: models/deepfm_309d_training_curve.png

================================================================================
[SUCCESS] DeepFM 학습 완료!
Best Valid RMSE: 0.9876
Test RMSE: 0.9876

모델 저장: models/deepfm_ranking_309d.pth
================================================================================
```

**소요 시간:**
- GPU (T4): 약 30-45분
- GPU (V100/A100): 약 15-20분
- CPU: 약 2-3시간 (권장 안함)

#### 3-4. HuggingFace 업로드

학습 완료 후 프롬프트가 나타남:
```
HuggingFace에 업로드하시겠습니까? (y/n):
```

**'y' 입력 후:**
```
HuggingFace 토큰을 입력하세요:
```

**토큰 입력:**
- https://huggingface.co/settings/tokens 에서 토큰 복사
- Colab에 붙여넣기
- 엔터

**업로드 결과:**
```
[1/2] HuggingFace 로그인...
  ✓ 로그인 완료

[2/2] yidj/soulplate-models에 업로드 중...
  ✓ deepfm_ranking_309d.pth 업로드 완료
  ✓ scaler_params_309d.json 업로드 완료
  ✓ tfidf_vectorizer_309d.pkl 업로드 완료

================================================================================
✓ HuggingFace 업로드 완료!
  Repository: https://huggingface.co/yidj/soulplate-models
================================================================================
```

#### 3-5. 결과 파일 다운로드 (선택)

학습 곡선 이미지 다운로드:
```python
from google.colab import files
files.download('models/deepfm_309d_training_curve.png')
```

---

### 🔹 Step 4: Multi-Tower 모델 학습

**새 셀에서 실행:**
```python
!python scripts/colab_train_multitower_309d.py
```

**과정은 DeepFM과 동일:**
1. 데이터 로딩
2. 모델 생성 (User Tower: 154차원, Business Tower: 155차원)
3. 학습 (Early Stopping)
4. Test 평가
5. HuggingFace 업로드

**소요 시간:**
- GPU (T4): 약 30-45분
- GPU (V100/A100): 약 15-20분

---

## 📊 학습 완료 체크리스트

### DeepFM
- [ ] 학습 완료 (Test RMSE < 1.0 권장)
- [ ] `models/deepfm_ranking_309d.pth` 생성 확인
- [ ] HuggingFace 업로드 완료
- [ ] 학습 곡선 이미지 확인

### Multi-Tower
- [ ] 학습 완료 (Test RMSE < 1.0 권장)
- [ ] `models/multitower_ranking_309d.pth` 생성 확인
- [ ] HuggingFace 업로드 완료
- [ ] 학습 곡선 이미지 확인

---

## ⚠️ 주의사항 및 문제 해결

### 1. GPU 메모리 부족
```
RuntimeError: CUDA out of memory
```

**해결 방법:**
```python
# 배치 크기 줄이기
# colab_train_deepfm_309d.py 또는 colab_train_multitower_309d.py에서

# 원래
batch_size = 512

# 수정
batch_size = 256  # 또는 128
```

### 2. Google Drive 연결 끊김
```
OSError: [Errno 107] Transport endpoint is not connected
```

**해결 방법:**
```python
# Drive 재마운트
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 3. Colab 세션 타임아웃
- **무료 버전:** 12시간 또는 90분 유휴 시 종료
- **Pro 버전:** 24시간 또는 유휴 시간 증가

**예방 방법:**
- 학습 중 가끔 페이지 클릭
- 또는 Colab Pro 구독 ($9.99/월)

### 4. 파일 경로 오류
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/training/ranking_train_309d.csv'
```

**확인 사항:**
```python
# 현재 디렉토리 확인
!pwd

# 파일 존재 확인
!ls -lh data/training/
```

**해결 방법:**
```python
# 스크립트 내부의 PROJECT_ROOT 수정
# colab_train_deepfm_309d.py Line 20-30 근처

PROJECT_ROOT = "/content/drive/MyDrive/soulplate_training"  # 본인 경로로 수정
```

---

## 🎯 학습 파라미터 튜닝 (선택)

### 기본 설정
```python
# DeepFM
input_dim = 309
embed_dim = 16
hidden_dims = [256, 128, 64]
batch_size = 512
learning_rate = 0.001
epochs = 100
patience = 10

# Multi-Tower
user_input_dim = 154
business_input_dim = 155
tower_dims = [128, 64]
interaction_dims = [64, 32]
batch_size = 512
learning_rate = 0.001
```

### 성능 개선을 위한 튜닝
```python
# 더 깊은 네트워크 (과적합 주의)
hidden_dims = [512, 256, 128, 64]

# 더 큰 임베딩 차원
embed_dim = 32

# Learning Rate 조정
learning_rate = 0.0005  # 더 안정적
learning_rate = 0.002   # 더 빠른 수렴

# Dropout 조정 (과적합 방지)
# 모델 정의에서 Dropout(0.3) → Dropout(0.5)
```

---

## 📈 학습 결과 해석

### RMSE (Root Mean Square Error)
- **1.0 이하:** 우수
- **0.9 이하:** 매우 우수
- **0.8 이하:** 탁월

### 학습 곡선 확인
1. **Training Loss:** 지속적으로 감소해야 함
2. **Validation RMSE:** 감소하다가 상승하면 과적합 (Early Stopping이 작동)
3. **Test RMSE:** Valid RMSE와 비슷해야 함 (차이가 크면 과적합)

---

## ✅ Phase 3 완료 후

모든 학습이 완료되면:

1. **HuggingFace 저장소 확인**
   - https://huggingface.co/yidj/soulplate-models
   - `deepfm_ranking_309d.pth` 존재 확인
   - `multitower_ranking_309d.pth` 존재 확인
   - `scaler_params_309d.json` 존재 확인
   - `tfidf_vectorizer_309d.pkl` 존재 확인

2. **로컬에서 Phase 4 진행**
   ```bash
   python scripts/validate_309d_model.py
   ```

3. **서비스 통합 (Phase 5)**
   - `prediction_service_309d.py`로 교체

---

## 💡 팁

### 1. Colab Pro 추천 사항
- 무료: T4 GPU (~45분/모델)
- Pro ($9.99/월): V100/A100 GPU (~15분/모델)
- 2개 모델 학습 시 Pro가 시간 절약

### 2. 동시 학습
DeepFM과 Multi-Tower를 동시에 학습하려면:
- 2개의 Colab 노트북 열기
- 각각에서 다른 모델 학습
- 총 소요 시간: 단일 모델 시간

### 3. 백업
학습 완료 후:
- Google Drive에 `models/` 폴더 백업
- 로컬 PC에도 다운로드 권장

---

## 🎉 완료!

Phase 3가 완료되면:
- ✅ DeepFM 모델 학습 완료
- ✅ Multi-Tower 모델 학습 완료
- ✅ HuggingFace 업로드 완료
- ✅ Phase 4 (검증) 준비 완료

**예상 총 소요 시간:**
- 준비: 10-15분
- DeepFM 학습: 30-45분 (GPU)
- Multi-Tower 학습: 30-45분 (GPU)
- HuggingFace 업로드: 5-10분
- **총: 약 1.5-2시간**

---

**질문이나 문제가 있으면:**
- Colab 콘솔 출력 확인
- 로그 메시지 읽기
- `README_309d_RETRAINING.md` 참조

