# ABSA 테이블 분리 마이그레이션 가이드

## 개요

User/Business 테이블의 ABSA 컬럼을 별도 테이블로 분리하여:
- **초기 마이그레이션 시간: 3시간 → 1-2분**
- ABSA 집계는 별도 스크립트로 처리 (5-10분)

## 변경 사항

### 1. 데이터베이스 스키마

**기존:**
```
users
  - absa_features (JSONB)

businesses
  - absa_features (JSONB)
```

**변경 후:**
```
users
  (absa_features 컬럼 제거)

businesses
  (absa_features 컬럼 제거)

user_absa_features (신규)
  - user_id (PK, FK)
  - absa_features (JSONB)
  - updated_at

business_absa_features (신규)
  - business_id (PK, FK)
  - absa_features (JSONB)
  - updated_at
```

### 2. 수정된 파일

- `backend_web/models.py` - 새 테이블 추가, relationship 설정
- `backend_web/main.py` - ABSA 접근 방식 변경 (relationship 사용)
- `scripts/fast_copy_migration.py` - ABSA 계산 로직 제거
- `scripts/aggregate_absa_features.py` - ABSA 집계 스크립트 (신규)

## 실행 순서

### Step 1: 스키마 준비

기존 데이터베이스가 있는 경우:

```bash
# ABSA 컬럼 제거
python scripts/drop_absa_columns.py

# 새 테이블 생성
python scripts/recreate_db_schema.py

# 스키마 확인
python scripts/test_absa_integration.py
```

새 데이터베이스인 경우:
```bash
# 자동으로 모든 테이블 생성됨
python scripts/recreate_db_schema.py
```

### Step 2: 빠른 마이그레이션 (1-2분)

```bash
python scripts/fast_copy_migration.py
```

**처리 내용:**
- User 데이터 삽입 (42k명) - ABSA 제외
- Business 데이터 삽입 (14k개) - ABSA 제외
- Review 데이터 삽입 (100k개) - ABSA 포함

### Step 3: ABSA 집계 (5-10분)

```bash
python scripts/aggregate_absa_features.py
```

**처리 내용:**
- Review 테이블에서 user_id별 ABSA 평균 계산
- UserABSAFeatures 테이블에 삽입
- Review 테이블에서 business_id별 ABSA 평균 계산
- BusinessABSAFeatures 테이블에 삽입

### Step 4: 검증

```bash
python scripts/test_absa_integration.py
```

**확인 사항:**
- 모든 테이블이 생성되었는지
- 데이터가 올바르게 삽입되었는지
- ABSA relationship이 작동하는지

## API 사용법

### Business 조회

```python
# 기존 코드 (동작하지 않음)
absa = business.absa_features

# 새 코드
absa = business.absa_features.absa_features if business.absa_features else None
```

### User 조회

```python
# 기존 코드 (동작하지 않음)
absa = user.absa_features

# 새 코드
absa = user.absa_features.absa_features if user.absa_features else None
```

## 장점

1. **속도**: 초기 마이그레이션 3시간 → 1-2분
2. **유연성**: ABSA 재계산 시 집계 스크립트만 다시 실행
3. **확장성**: 새 User/Business 추가 시 빠르게 삽입 가능
4. **데이터 무결성**: Review ABSA는 변하지 않고, User/Business ABSA만 집계로 관리

## 주의사항

1. **Relationship 사용**: `business.absa_features`는 이제 객체이므로, JSON 데이터는 `.absa_features` 속성으로 접근
2. **NULL 처리**: ABSA가 없는 경우 `None`을 반환하므로 체크 필요
3. **집계 재실행**: Review 데이터가 변경되면 `aggregate_absa_features.py` 재실행 필요

## 문제 해결

### 테이블이 생성되지 않음
```bash
python scripts/recreate_db_schema.py
```

### 기존 absa_features 컬럼이 남아있음
```bash
python scripts/drop_absa_columns.py
```

### ABSA 데이터가 없음
```bash
python scripts/aggregate_absa_features.py
```

## 성능 비교

| 작업 | 기존 방식 | 새 방식 |
|------|----------|---------|
| 초기 마이그레이션 | 3시간 | 1-2분 |
| ABSA 집계 | - | 5-10분 |
| 총 소요 시간 | 3시간 | 6-12분 |
| **개선율** | - | **95% 단축** |

