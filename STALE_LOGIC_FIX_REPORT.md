# Stale 예측 재계산 로직 수정 보고서

**작성일**: 2024
**문제**: Fresh와 Stale이 동시에 있을 때 Stale이 재계산되지 않음

---

## 🔍 문제 분석

### 현재 로직의 문제

**기존 코드**:
```python
if not check_predictions_exist(user.id, db):
    # Fresh 없음 → 재계산
elif check_has_stale_predictions(user.id, db):
    # Stale 있음 → 재계산
```

**문제 시나리오**:
```
사용자가 Fresh 100개 + Stale 500개를 가지고 있는 경우:
1. check_predictions_exist() → True (Fresh 있음)
2. 첫 번째 if 조건 False
3. elif 조건 확인 안 함
4. 재계산 안 됨 ❌
```

### DB 분석 결과

- **현재 상태**: Fresh와 Stale이 동시에 있는 사용자는 없음
- **미래 가능성**: Fresh와 Stale이 혼재할 수 있음
- **예방 조치**: Stale 우선 처리로 로직 수정

---

## 🔧 수정 내용

### 1. 로그인 로직 수정

**파일**: `backend_web/main.py` (Line 500-511)

**변경 전**:
```python
if not check_predictions_exist(user.id, db):
    # Fresh 없음 → 재계산
elif check_has_stale_predictions(user.id, db):
    # Stale 있음 → 재계산
```

**변경 후**:
```python
# Stale 우선 처리: Stale이 있으면 무조건 재계산 (Fresh 여부 무관)
if check_has_stale_predictions(user.id, db):
    logger.info(f"사용자 {user.username}의 stale 예측이 있어 백그라운드 재계산 시작")
    if background_tasks:
        background_tasks.add_task(calculate_and_store_predictions, user.id, db)
elif not check_predictions_exist(user.id, db):
    logger.info(f"사용자 {user.username}의 예측값이 없어 백그라운드 생성 시작")
    if background_tasks:
        background_tasks.add_task(calculate_and_store_predictions, user.id, db)
```

**개선사항**:
- Stale 체크를 먼저 수행
- Stale이 있으면 Fresh 여부와 관계없이 재계산
- Fresh와 Stale이 혼재해도 문제 없음

### 2. API 호출 로직 수정

**파일**: `backend_web/main.py` (Line 887-898)

동일한 로직으로 수정

---

## 📊 수정 전후 비교

### 시나리오 1: Stale만 있음
```
수정 전: check_predictions_exist() → False → 재계산 ✅
수정 후: check_has_stale_predictions() → True → 재계산 ✅
```

### 시나리오 2: Fresh만 있음
```
수정 전: check_predictions_exist() → True → 재계산 안 됨 ✅
수정 후: check_has_stale_predictions() → False → 재계산 안 됨 ✅
```

### 시나리오 3: Fresh와 Stale 혼재 (문제 케이스)
```
수정 전: check_predictions_exist() → True → 재계산 안 됨 ❌
수정 후: check_has_stale_predictions() → True → 재계산 ✅
```

---

## ✅ 예상 결과

### 수정 후 동작

1. **Stale이 있으면**: 무조건 재계산 (Fresh 여부 무관)
2. **Stale이 없고 Fresh도 없으면**: 재계산
3. **Fresh만 있으면**: 재계산 안 됨 (정상)

### 장점

- ✅ Fresh와 Stale이 혼재해도 Stale이 재계산됨
- ✅ 로직이 더 명확하고 예측 가능함
- ✅ Stale 예측이 항상 최신화됨

---

## 🔄 되돌리기 방법

```bash
git checkout backend_web/main.py
```

---

## 📝 변경 파일

1. `backend_web/main.py`
   - 로그인 엔드포인트 수정
   - API 호출 엔드포인트 수정

---

**작성 완료일**: 2024
**상태**: 로컬 수정 완료, Railway 재배포 대기 중

