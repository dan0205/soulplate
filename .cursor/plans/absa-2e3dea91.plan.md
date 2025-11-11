<!-- 2e3dea91-2012-4103-be89-9b7aa212673f 745ece05-9b52-4143-8a54-30d3906dac33 -->
# 필터링된 데이터 마이그레이션 계획

## 목표

- ABSA 테이블 분리 롤백 (원래 구조로 복구)
- Business 20개 이상 + User 3개 이상 필터링 (676개/1,812명/13,780개)
- tqdm 사용하여 안정적이고 가시적인 마이그레이션

## Phase 1: ABSA 구조 롤백 (원래대로 복구)

### 1-1. models.py 복구

**변경 사항:**

- User 테이블: `absa_features = Column(JSONB, nullable=True)` 복구
- Business 테이블: `absa_features = Column(JSONB, nullable=True)` 복구
- UserABSAFeatures 클래스 제거
- BusinessABSAFeatures 클래스 제거
- relationship 제거

### 1-2. backend_web/main.py 복구

**변경 사항:**

```python
# 기존 (복잡)
user_absa = user.absa_features.absa_features if user.absa_features else {}
business_absa = business.absa_features.absa_features if business.absa_features else {}

# 복구 (간단)
user_absa = user.absa_features or {}
business_absa = business.absa_features or {}
```

### 1-3. DB 스키마 재구성

**작업:**

1. 기존 데이터 백업 (필요시)
2. user_absa_features, business_absa_features 테이블 삭제
3. users, businesses 테이블에 absa_features 컬럼 추가 (JSONB)

**파일:**

- `backend_web/models.py`
- `backend_web/main.py`
- `scripts/restore_absa_columns.py` (신규)

## Phase 2: 데이터 필터링

### 2-1. 필터링 기준

**Business 필터:**

- 리뷰 20개 이상 받은 Business만 선택
- 결과: 679개

**User 필터:**

- 위 Business에 리뷰를 3개 이상 작성한 User만 선택
- 결과: 1,812명

**Review 필터:**

- 위 Business와 User에 해당하는 Review만 선택
- 결과: 13,780개

### 2-2. 필터링된 CSV 생성

**생성할 파일:**

- `data/filtered/user_filtered_20_3.csv` - 1,812명
- `data/filtered/business_filtered_20_3.csv` - 676개
- `data/filtered/review_filtered_20_3.csv` - 13,780개

**파일:**

- `scripts/create_filtered_dataset.py` (신규)

## Phase 3: 마이그레이션 스크립트 작성

### 3-1. 안정적인 마이그레이션 방식

**핵심 요소:**

```python
from tqdm import tqdm

# 1. 데이터 로딩 및 준비
# 2. tqdm으로 진행률 표시
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Users"):
    obj = models.User(...)
    db.add(obj)
    
    if (idx + 1) % 500 == 0:
        db.commit()

# 3. 최종 커밋
db.commit()
```

**장점:**

- 실시간 진행 상황 확인 가능
- 단순한 session.add() 방식으로 안정적
- 500개마다 커밋으로 메모리 관리

### 3-2. 마이그레이션 순서

1. 기존 데이터 삭제 (truncate)
2. Users 삽입 (1,812명, ABSA 포함) - ~1분
3. Businesses 삽입 (676개, ABSA 포함) - ~10초
4. Reviews 삽입 (13,780개, ABSA 포함) - ~1분
5. 검증

**파일:**

- `scripts/migrate_filtered_data.py` (신규)

## Phase 4: 검증 및 테스트

### 4-1. 데이터 검증

- User 수: 1,812명
- Business 수: 676개
- Review 수: 13,780개
- ABSA 필드 존재 확인
- 샘플 데이터 출력

### 4-2. API 테스트

- GET /businesses - ABSA 포함 확인
- GET /businesses/{id} - 상세 ABSA 확인
- ABSA가 없는 경우 null 처리 확인

**파일:**

- `scripts/verify_filtered_migration.py` (신규)

## 파일 체크리스트

### 수정

- [ ] `backend_web/models.py` - ABSA 컬럼 복구, 별도 테이블 제거
- [ ] `backend_web/main.py` - relationship 제거, 직접 접근으로 변경

### 신규 생성

- [ ] `scripts/restore_absa_columns.py` - DB 스키마 복구
- [ ] `scripts/create_filtered_dataset.py` - 필터링된 CSV 생성
- [ ] `scripts/migrate_filtered_data.py` - 메인 마이그레이션
- [ ] `scripts/verify_filtered_migration.py` - 검증

## 예상 소요 시간

- Phase 1 (롤백): 10분
- Phase 2 (필터링): 5분
- Phase 3 (마이그레이션): 3분
- Phase 4 (검증): 2분
- **총: 20분**

## 장점

1. **단순성**: 원래의 간단한 구조로 복귀
2. **안정성**: tqdm + session.add()로 검증된 방식
3. **규모**: 1,812명으로 관리 가능한 데모 데이터
4. **품질**: User당 평균 7.6개 리뷰로 높은 활동성
5. **가시성**: 진행 상황을 실시간으로 확인 가능

### To-dos

- [ ] models.py에서 User/Business absa_features 컬럼 복구, ABSA 테이블 클래스 제거
- [ ] main.py에서 relationship 접근을 직접 접근으로 변경
- [ ] DB에 absa_features 컬럼 추가, ABSA 테이블 삭제 스크립트 작성 및 실행
- [ ] Business 20+/User 3+ 필터링된 CSV 생성 (1,812명/676개/13,780개)
- [ ] tqdm 사용한 안정적인 마이그레이션 스크립트 작성
- [ ] 필터링된 데이터로 마이그레이션 실행
- [ ] 데이터 검증 및 API 테스트