<!-- 2e3dea91-2012-4103-be89-9b7aa212673f 69bb4688-9856-4744-9b14-129052b4ce18 -->
# ABSA 테이블 분리 계획

## 문제점

- 현재 User/Business 테이블의 ABSA 평균값을 계산하여 삽입하는 데 3시간 소요
- 초기 마이그레이션이 너무 느림

## 해결 방안

ABSA 컬럼을 별도 테이블로 분리하여:

1. 초기 마이그레이션: User/Business/Review만 빠르게 삽입 (1-2분)
2. 이후 집계: 별도 스크립트로 Review ABSA를 집계하여 ABSA 테이블에 삽입

## Phase 1: DB 스키마 수정

### 새로운 테이블 추가 (models.py)

**UserABSAFeatures 테이블:**

```python
class UserABSAFeatures(Base):
    __tablename__ = "user_absa_features"
    
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    absa_features = Column(JSONB, nullable=False)
    updated_at = Column(DateTime, default=datetime.now(timezone.utc))
```

**BusinessABSAFeatures 테이블:**

```python
class BusinessABSAFeatures(Base):
    __tablename__ = "business_absa_features"
    
    business_id = Column(Integer, ForeignKey("businesses.id"), primary_key=True)
    absa_features = Column(JSONB, nullable=False)
    updated_at = Column(DateTime, default=datetime.now(timezone.utc))
```

### 기존 테이블 수정

- User 테이블: `absa_features` 컬럼 제거
- Business 테이블: `absa_features` 컬럼 제거
- Review 테이블: `absa_features` 유지 (이미 계산됨)

### 파일

- `backend_web/models.py`

## Phase 2: 마이그레이션 스크립트 수정

### fast_copy_migration.py 수정

**User CSV 준비 (prepare_users_csv):**

- ABSA 계산 로직 제거
- absa_features 컬럼 제외

**Business CSV 준비 (prepare_businesses_csv):**

- ABSA 계산 로직 제거
- absa_features 컬럼 제외

**Review CSV 준비 (prepare_reviews_csv):**

- 변경 없음 (이미 계산된 ABSA 그대로 사용)

### 예상 속도

- 기존: 3시간 (ABSA 계산 포함)
- 개선: 1-2분 (ABSA 제외)

### 파일

- `scripts/fast_copy_migration.py`

## Phase 3: ABSA 집계 스크립트 생성

### 새 스크립트: aggregate_absa_features.py

**기능:**

1. Review 테이블에서 user_id별로 ABSA 평균 계산
2. UserABSAFeatures 테이블에 삽입
3. Review 테이블에서 business_id별로 ABSA 평균 계산
4. BusinessABSAFeatures 테이블에 삽입

**처리 방식:**

- PostgreSQL의 집계 쿼리 활용 (GROUP BY + AVG)
- JSON 필드 처리는 Python에서 수행
- Batch 삽입으로 성능 최적화

**예상 시간:**

- 100k 리뷰 집계: 5-10분

### 파일

- `scripts/aggregate_absa_features.py` (신규)

## Phase 4: API 수정

### backend_web/main.py 수정

**GET /businesses 엔드포인트:**

- Business와 BusinessABSAFeatures LEFT JOIN
- 상위 5개 ABSA 특징 추출

**GET /businesses/{id} 엔드포인트:**

- Business와 BusinessABSAFeatures LEFT JOIN
- 전체 ABSA JSON 반환

### backend_web/schemas.py

- 변경 없음 (ABSA는 Optional이므로 null 허용)

### 파일

- `backend_web/main.py`

## Phase 5: 테스트

### 테스트 시나리오

1. DB 스키마 재생성 확인
2. 빠른 마이그레이션 실행 (1-2분)
3. ABSA 집계 스크립트 실행 (5-10분)
4. API 조회 테스트

   - ABSA가 있는 Business 조회
   - ABSA가 없는 Business 조회 (null 처리)

## 주요 파일 목록

### Backend

- `backend_web/models.py` - 스키마 수정
- `backend_web/main.py` - API JOIN 쿼리 수정

### Scripts

- `scripts/fast_copy_migration.py` - ABSA 제거
- `scripts/aggregate_absa_features.py` - ABSA 집계 (신규)

## 장점

1. **속도 개선**: 3시간 → 1-2분 (초기 마이그레이션)
2. **유연성**: ABSA 재계산이 필요할 때 집계 스크립트만 다시 실행
3. **확장성**: User/Business가 추가되어도 기본 데이터는 빠르게 삽입
4. **일관성**: Review ABSA는 변하지 않고, User/Business ABSA만 집계로 관리

### To-dos

- [ ] models.py에 UserABSAFeatures, BusinessABSAFeatures 테이블 추가, User/Business에서 absa_features 제거
- [ ] fast_copy_migration.py에서 ABSA 계산 로직 제거
- [ ] aggregate_absa_features.py 스크립트 생성 (Review 집계)
- [ ] backend_web/main.py에서 ABSA 테이블 JOIN 쿼리 추가
- [ ] 마이그레이션 → 집계 → API 조회 전체 플로우 테스트