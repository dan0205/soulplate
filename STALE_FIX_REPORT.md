# Stale 예측 재계산 문제 수정 보고서

**작성일**: 2024
**문제**: stale 예측이 자동으로 재계산되지 않음

---

## 🔍 문제 분석

### 발견된 문제

**증상**: 
- `is_stale=True`로 표시된 예측이 재계산되지 않음
- 사용자가 로그인해도 stale 예측이 그대로 유지됨

**원인**:

1. **`check_predictions_exist()` 함수의 문제**:
   ```python
   # 수정 전
   def check_predictions_exist(user_id: int, db: Session) -> bool:
       count = db.query(models.UserBusinessPrediction).filter(
           models.UserBusinessPrediction.user_id == user_id
       ).count()
       return count > 0  # stale 여부 무시!
   ```
   - stale 예측도 "존재함"으로 처리
   - stale 여부를 확인하지 않음

2. **로그인 시 로직** (`main.py` line 502):
   ```python
   if not check_predictions_exist(user.id, db):
       # 예측이 없을 때만 재계산
   ```
   - stale 예측이 있어도 "예측이 있음"으로 판단
   - 재계산 트리거되지 않음

3. **API 호출 시 로직** (`main.py` line 884):
   - 동일한 문제 발생

---

## 🔧 수정 내용

### 1. `check_predictions_exist()` 함수 수정

**파일**: `backend_web/prediction_cache.py` (Line 334-353)

**변경 전**:
```python
def check_predictions_exist(user_id: int, db: Session) -> bool:
    count = db.query(models.UserBusinessPrediction).filter(
        models.UserBusinessPrediction.user_id == user_id
    ).count()
    return count > 0
```

**변경 후**:
```python
def check_predictions_exist(user_id: int, db: Session) -> bool:
    """
    사용자의 예측값이 존재하는지 확인
    
    Returns:
        bool: 예측값 존재 여부 (stale이 아닌 fresh 예측만 카운트)
    """
    # stale이 아닌 fresh 예측만 카운트
    count = db.query(models.UserBusinessPrediction).filter(
        and_(
            models.UserBusinessPrediction.user_id == user_id,
            models.UserBusinessPrediction.is_stale == False
        )
    ).count()
    
    return count > 0
```

**개선사항**:
- `is_stale == False` 조건 추가
- Fresh 예측만 "존재함"으로 처리
- Stale 예측은 "없음"으로 처리

### 2. `check_has_stale_predictions()` 함수 추가

**파일**: `backend_web/prediction_cache.py` (Line 356-374)

**새로 추가된 함수**:
```python
def check_has_stale_predictions(user_id: int, db: Session) -> bool:
    """
    사용자에게 stale 예측이 있는지 확인
    
    Args:
        user_id: 사용자 ID
        db: 데이터베이스 세션
    
    Returns:
        bool: stale 예측 존재 여부
    """
    count = db.query(models.UserBusinessPrediction).filter(
        and_(
            models.UserBusinessPrediction.user_id == user_id,
            models.UserBusinessPrediction.is_stale == True
        )
    ).count()
    
    return count > 0
```

**용도**: stale 예측 존재 여부를 명확히 확인

### 3. 로그인 시 로직 수정

**파일**: `backend_web/main.py` (Line 500-511)

**변경 전**:
```python
from prediction_cache import check_predictions_exist, calculate_and_store_predictions
if not check_predictions_exist(user.id, db):
    logger.info(f"사용자 {user.username}의 예측값이 없어 백그라운드 생성 시작")
    if background_tasks:
        background_tasks.add_task(calculate_and_store_predictions, user.id, db)
```

**변경 후**:
```python
from prediction_cache import check_predictions_exist, check_has_stale_predictions, calculate_and_store_predictions

# Fresh 예측이 없거나 stale 예측이 있으면 재계산
if not check_predictions_exist(user.id, db):
    logger.info(f"사용자 {user.username}의 예측값이 없어 백그라운드 생성 시작")
    if background_tasks:
        background_tasks.add_task(calculate_and_store_predictions, user.id, db)
elif check_has_stale_predictions(user.id, db):
    logger.info(f"사용자 {user.username}의 stale 예측이 있어 백그라운드 재계산 시작")
    if background_tasks:
        background_tasks.add_task(calculate_and_store_predictions, user.id, db)
```

**개선사항**:
- Fresh 예측이 없을 때: 재계산
- Stale 예측이 있을 때: 재계산
- 두 경우 모두 처리

### 4. API 호출 시 로직 수정

**파일**: `backend_web/main.py` (Line 887-893)

**변경 전**:
```python
if not check_predictions_exist(current_user.id, db):
    logger.info(f"사용자 {current_user.id}의 예측값이 없어 백그라운드 계산 시작")
    if background_tasks:
        background_tasks.add_task(calculate_and_store_predictions, current_user.id, db)
```

**변경 후**:
```python
from prediction_cache import check_has_stale_predictions
if not check_predictions_exist(current_user.id, db):
    logger.info(f"사용자 {current_user.id}의 예측값이 없어 백그라운드 계산 시작")
    if background_tasks:
        background_tasks.add_task(calculate_and_store_predictions, current_user.id, db)
elif check_has_stale_predictions(current_user.id, db):
    logger.info(f"사용자 {current_user.id}의 stale 예측이 있어 백그라운드 재계산 시작")
    if background_tasks:
        background_tasks.add_task(calculate_and_store_predictions, current_user.id, db)
```

**개선사항**: 로그인과 동일한 로직 적용

---

## 📊 수정 전후 비교

### 수정 전

```
사용자 로그인
  ↓
check_predictions_exist() 호출
  ↓
stale 예측도 "존재함"으로 판단
  ↓
재계산 안 됨 ❌
```

### 수정 후

```
사용자 로그인
  ↓
check_predictions_exist() 호출 (fresh만 확인)
  ↓
fresh 예측 없음 → check_has_stale_predictions() 호출
  ↓
stale 예측 있음 → 재계산 트리거 ✅
```

---

## ✅ 예상 결과

### 시나리오 1: Fresh 예측 없음
```
check_predictions_exist() → False
→ 재계산 트리거 ✅
```

### 시나리오 2: Stale 예측만 있음
```
check_predictions_exist() → False (fresh 없음)
check_has_stale_predictions() → True
→ 재계산 트리거 ✅
```

### 시나리오 3: Fresh 예측 있음
```
check_predictions_exist() → True
→ 재계산 안 함 (정상) ✅
```

---

## 🚀 테스트 방법

### 1. Stale 예측 생성
```bash
python scripts/fix_prediction_cache.py --action stale
```

### 2. 사용자 로그인
- 로그인 시 백그라운드 재계산 시작
- 로그 확인: "stale 예측이 있어 백그라운드 재계산 시작"

### 3. API 호출
- `/api/businesses` 호출 시 stale 예측 감지
- 자동 재계산 트리거

---

## 📝 변경 파일 목록

1. `backend_web/prediction_cache.py`
   - `check_predictions_exist()` 수정
   - `check_has_stale_predictions()` 추가

2. `backend_web/main.py`
   - 로그인 엔드포인트 수정
   - API 호출 엔드포인트 수정

---

## 🔄 되돌리기 방법

```bash
# Git을 통한 되돌리기
git checkout backend_web/prediction_cache.py backend_web/main.py

# 또는 특정 커밋으로 되돌리기
git revert <commit-hash>
```

---

## ✅ 완료 체크리스트

- [x] `check_predictions_exist()` 수정 (fresh만 확인)
- [x] `check_has_stale_predictions()` 함수 추가
- [x] 로그인 시 stale 체크 추가
- [x] API 호출 시 stale 체크 추가
- [x] 코드 linter 통과
- [x] 문서화 완료
- [ ] Railway 재배포 (사용자 실행 필요)
- [ ] 실제 테스트 (재배포 후)

---

**작성 완료일**: 2024
**상태**: 로컬 수정 완료, Railway 재배포 대기 중

