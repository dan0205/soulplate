# Stale 예측 재계산 문제 분석 보고서

**작성일**: 2024
**문제**: stale=True인데도 어떤 예측은 재계산되고 어떤 것은 넘어감

---

## 📊 DB 데이터 분석 결과

### 전체 통계

```
총 예측 수: 79,798개
- Stale: 79,112개 (99.1%)
- Fresh: 686개 (0.9%)
```

### 사용자별 분석

**패턴 1: Stale만 있는 사용자 (대부분)**
- 예: admin, 테스트, yelp_* 사용자들
- Fresh: 0개, Stale: 676~686개
- **로직 결과**: `check_predictions_exist() → False`, `check_has_stale_predictions() → True`
- **예상 동작**: 재계산 트리거됨 ✅

**패턴 2: Fresh만 있는 사용자 (1명)**
- 사용자: `abc` (ID: 24288)
- Fresh: 686개 (100%), Stale: 0개
- 계산 시간: 2025-11-18 18:31:13 (최근!)
- **로직 결과**: `check_predictions_exist() → True`, `check_has_stale_predictions() → False`
- **예상 동작**: 재계산 안 됨 (정상) ✅

---

## 🔍 문제 원인 분석

### 발견된 문제

**사용자가 관찰한 현상**:
- "stale이 true인데도 어떤 건 예측을 진행했고, 어떤 건 넘어갔다"

**가능한 원인들**:

1. **Fresh와 Stale이 동시에 있는 경우** (현재는 없음)
   - Fresh가 있으면 `check_predictions_exist() → True`
   - 재계산 트리거 안 됨
   - Stale 예측이 재계산되지 않음

2. **재계산 진행 중**
   - 백그라운드 작업이 실행 중일 수 있음
   - 완료되기 전까지는 여전히 stale 상태

3. **재계산 실패**
   - API 호출 실패
   - 타임아웃
   - 에러 발생

4. **백그라운드 작업 미실행**
   - `background_tasks`가 None일 수 있음
   - 로그인 시 백그라운드 작업이 실행되지 않았을 수 있음

---

## 🧪 로직 시뮬레이션 결과

### 현재 코드 로직

```python
# 로그인 시 (main.py line 504-511)
if not check_predictions_exist(user.id, db):
    # Fresh 예측 없음 → 재계산 트리거
elif check_has_stale_predictions(user.id, db):
    # Stale 예측 있음 → 재계산 트리거
```

### 실제 데이터로 테스트

**Stale만 있는 사용자**:
```
check_predictions_exist() → False (Fresh 없음)
check_has_stale_predictions() → True (Stale 있음)
→ 재계산 트리거됨 ✅
```

**Fresh만 있는 사용자**:
```
check_predictions_exist() → True (Fresh 있음)
→ 재계산 안 됨 (정상) ✅
```

**Fresh와 Stale이 동시에 있는 경우** (현재는 없음):
```
check_predictions_exist() → True (Fresh 있음)
→ 재계산 안 됨 ⚠️
→ Stale 예측이 재계산되지 않음!
```

---

## 💡 해결 방안

### 문제: Fresh와 Stale이 동시에 있을 때

현재 로직의 문제:
- Fresh가 있으면 재계산 안 됨
- Stale 예측이 재계산되지 않음

### 해결책 1: Stale 우선 처리 (권장)

```python
# 수정 전
if not check_predictions_exist(user.id, db):
    # 재계산
elif check_has_stale_predictions(user.id, db):
    # 재계산

# 수정 후
if check_has_stale_predictions(user.id, db):
    # Stale이 있으면 무조건 재계산 (Fresh 여부 무관)
    logger.info(f"Stale 예측이 있어 재계산 시작")
    background_tasks.add_task(calculate_and_store_predictions, user.id, db)
elif not check_predictions_exist(user.id, db):
    # Fresh도 없고 Stale도 없으면 재계산
    logger.info(f"예측이 없어 재계산 시작")
    background_tasks.add_task(calculate_and_store_predictions, user.id, db)
```

**장점**:
- Stale 예측이 항상 재계산됨
- Fresh와 Stale이 혼재해도 문제 없음

### 해결책 2: 부분 재계산 (복잡함)

- Stale 예측만 선별적으로 재계산
- 구현 복잡도 증가

---

## 📝 권장 수정사항

### 1. 로그인 로직 수정

**파일**: `backend_web/main.py` (Line 504-511)

**변경 전**:
```python
if not check_predictions_exist(user.id, db):
    # Fresh 없음 → 재계산
elif check_has_stale_predictions(user.id, db):
    # Stale 있음 → 재계산
```

**변경 후**:
```python
# Stale 우선 처리: Stale이 있으면 무조건 재계산
if check_has_stale_predictions(user.id, db):
    logger.info(f"사용자 {user.username}의 stale 예측이 있어 백그라운드 재계산 시작")
    if background_tasks:
        background_tasks.add_task(calculate_and_store_predictions, user.id, db)
elif not check_predictions_exist(user.id, db):
    logger.info(f"사용자 {user.username}의 예측값이 없어 백그라운드 생성 시작")
    if background_tasks:
        background_tasks.add_task(calculate_and_store_predictions, user.id, db)
```

### 2. API 호출 로직도 동일하게 수정

**파일**: `backend_web/main.py` (Line 890-898)

동일한 로직 적용

---

## 🔄 재계산이 안 되는 다른 가능성

### 1. 백그라운드 작업 미실행

**확인 필요**:
- `background_tasks`가 None인지 확인
- 로그인 시 백그라운드 작업이 실제로 실행되는지 확인

### 2. 재계산 실패

**확인 필요**:
- Railway 로그에서 에러 확인
- API 호출 실패 여부 확인
- 타임아웃 발생 여부 확인

### 3. 재계산 진행 중

**확인 필요**:
- 재계산이 진행 중인지 확인
- 완료 시간 확인

---

## ✅ 결론

### 현재 상태

1. **대부분의 사용자**: Stale만 있음 → 재계산 트리거됨 ✅
2. **일부 사용자**: Fresh만 있음 → 재계산 안 됨 (정상) ✅
3. **문제 시나리오**: Fresh와 Stale이 동시에 있으면 → 재계산 안 됨 ⚠️

### 권장 조치

1. **즉시**: 로직을 Stale 우선 처리로 수정
2. **확인**: 실제 재계산이 실행되는지 Railway 로그 확인
3. **모니터링**: 재계산 성공/실패 로그 추가

---

**작성 완료일**: 2024
**상태**: 분석 완료, 수정 권장

