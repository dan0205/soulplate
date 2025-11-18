# 로깅 과다 문제 수정 보고서

**작성일**: 2024
**문제**: Railway 로그 rate limit (500 logs/sec) 초과

---

## 🔍 문제 분석

### 발견된 문제

**증상**:
- Railway 로그 rate limit에 걸림: "Railway rate limit of 500 logs/sec reached"
- 4,301개 메시지가 드롭됨

**원인**:
- 매 예측 요청마다 4-5줄의 디버그 로그 출력
- `[DEBUG]` 로그가 너무 많이 출력됨:
  - Combined features shape
  - Combined features stats
  - Non-zero features
  - First 10 features
  - DeepFM 예측
  - Multi-Tower input shapes
  - Multi-Tower 예측
- 예측 요청 로그도 매번 출력

**좋은 소식**:
- ✅ DeepFM이 이제 정상적으로 3~4점대를 예측하고 있음!
- ✅ 수정이 성공적으로 적용됨

---

## 🔧 수정 내용

### 1. prediction_service.py 디버그 로그 제거

**파일**: `backend_model/prediction_service.py`

**변경 전**:
```python
print(f"[DEBUG] Combined features shape: {combined_features.shape}")
print(f"[DEBUG] Combined features stats: min=..., max=..., mean=...")
print(f"[DEBUG] Non-zero features: ...")
print(f"[DEBUG] First 10 features: ...")
print(f"[DEBUG] DeepFM 예측: {deepfm_pred:.2f}")
print(f"[DEBUG] Multi-Tower input shapes: ...")
print(f"[DEBUG] Multi-Tower 예측: {mt_pred:.2f}")
```

**변경 후**:
```python
# 디버그 로그는 필요시에만 출력 (환경변수로 제어)
if os.getenv("DEBUG_PREDICTION", "false").lower() == "true":
    logger.debug(f"[DEBUG] Combined features shape: {combined_features.shape}")
    logger.debug(f"[DEBUG] Combined features stats: ...")
    # ... (디버그 모드에서만 출력)

# 예측 결과 로그 제거
# print(f"[DEBUG] DeepFM 예측: ...")  ← 제거됨
# print(f"[DEBUG] Multi-Tower 예측: ...")  ← 제거됨
```

**개선사항**:
- `print()` → `logger.debug()`로 변경
- 환경변수 `DEBUG_PREDICTION=true`일 때만 출력
- 기본적으로 디버그 로그 출력 안 함

### 2. main.py 예측 로그 제거

**파일**: `backend_model/main.py`

**변경 전**:
```python
logger.info(f"Rating prediction request")
logger.info(f"Prediction: DeepFM={result['deepfm_rating']}, MT={result['multitower_rating']}, Ensemble={result['ensemble_rating']}")
```

**변경 후**:
```python
# 요청 로그는 디버그 모드에서만 출력
if os.getenv("DEBUG_PREDICTION", "false").lower() == "true":
    logger.debug(f"Rating prediction request")
    logger.debug(f"Prediction: DeepFM={result['deepfm_rating']}, MT={result['multitower_rating']}, Ensemble={result['ensemble_rating']}")
```

**개선사항**:
- `logger.info()` → `logger.debug()`로 변경
- 환경변수로 제어 가능

---

## 📊 예상 효과

### 수정 전
```
매 예측 요청마다:
- 1줄: "Rating prediction request"
- 4줄: Combined features 디버그
- 1줄: DeepFM 예측
- 1줄: Multi-Tower input shapes
- 1줄: Multi-Tower 예측
- 1줄: Prediction 결과
= 총 9줄/요청

초당 50개 요청 시: 450줄/초 → Rate limit 초과!
```

### 수정 후
```
매 예측 요청마다:
- 0줄 (디버그 모드 꺼짐)
= 총 0줄/요청

초당 50개 요청 시: 0줄/초 ✅
```

---

## 🎯 디버그 모드 활성화 방법

필요시 디버그 로그를 보고 싶다면:

### Railway 환경변수 설정
```bash
# Railway 대시보드에서 환경변수 추가
DEBUG_PREDICTION=true
```

### 로컬 테스트
```bash
export DEBUG_PREDICTION=true
python backend_model/main.py
```

---

## ✅ 변경 파일 목록

1. `backend_model/prediction_service.py`
   - 디버그 `print()` 문 제거 또는 환경변수 제어로 변경
   - 예측 결과 로그 제거

2. `backend_model/main.py`
   - `os` import 추가
   - 예측 요청/결과 로그를 디버그 모드로 변경

---

## 🔄 되돌리기 방법

```bash
# 백업 파일로 복원
git checkout backend_model/prediction_service.py backend_model/main.py

# 또는 특정 커밋으로 되돌리기
git revert <commit-hash>
```

---

## 📝 참고사항

### DeepFM 수정 성공 확인 ✅

로그에서 확인된 결과:
- DeepFM 예측: **3.26~4.35점** (정상 범위!)
- 이전: 1.02~1.11점
- **수정 성공!**

### Multi-Tower 점수

- Multi-Tower: 2.14~3.01점
- 실제 별점: 3.67~4.67점
- Multi-Tower가 약간 낮게 예측하는 경향 (별도 확인 필요할 수 있음)

---

**작성 완료일**: 2024
**상태**: 로컬 수정 완료, Railway 재배포 대기 중

