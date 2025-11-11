<!-- d86bc682-c3f1-497c-ba02-8ef2308c550f 58930eee-1e1d-45d6-8e0e-752d53b08459 -->
# ABSA 모델 서비스 통합 및 DB 재구축

## 목표

1. DeepFM과 Multi-Tower 모델을 사용하여 사용자별 맞춤 별점 예측 제공
2. PostgreSQL DB를 재구축하고 Yelp 100k 데이터 전체 삽입
3. 메인 홈페이지와 BusinessDetail 페이지에 AI 예측 및 ABSA 특징 표시

## Phase 1: PostgreSQL DB 설정 및 스키마 정의

### 새로운 DB 스키마

**User 테이블:**

- id (auto increment, PK)
- yelp_user_id (nullable, Yelp 데이터 매칭용, unique)
- username, email, hashed_password (인증용)
- review_count, useful, compliment, fans, average_stars, yelping_since_days
- absa_features (JSON) - 51개 ABSA 평균값

**Business 테이블:**

- id (auto increment, PK)
- business_id (Yelp ID, unique)
- name, categories, stars, review_count
- latitude, longitude, address, city, state
- absa_features (JSON) - 51개 ABSA 평균값

**Review 테이블:**

- id (auto increment, PK)
- user_id (FK)
- business_id (FK)
- stars, text, date
- absa_features (JSON) - 51개 ABSA 값

### 파일

- `backend_web/models.py` (업데이트)
- `scripts/setup_postgresql.py` (PostgreSQL 설정 스크립트)

## Phase 2: 데이터 마이그레이션

### Step 1: 기존 DB 백업 및 삭제

- `app.db` 백업
- PostgreSQL 데이터베이스 생성

### Step 2: Yelp 데이터 삽입

**User 데이터:**

- `data/processed/user_preprocessed.csv` (42,223명)
- yelp_user_id로 저장, 가상 회원으로 처리
- username: yelp_{user_id}, 임시 비밀번호

**Business 데이터:**

- `data/processed/business_preprocessed.csv` (14,519개)
- business_id, name, categories, 위치 정보
- ABSA 피처를 JSON으로 변환

**Review 데이터:**

- `data/processed/review_absa_features.csv` (100,000개)
- user_id, business_id 매칭
- ABSA 피처를 JSON으로 변환

### 파일

- `scripts/migrate_data_to_postgresql.py`
- `scripts/convert_absa_to_json.py`

## Phase 3: 예측 API 추가

### backend_model API에 엔드포인트 추가

**POST /predict_rating**

- 입력: `{user_id: str, business_id: str}`
- 출력: 
```json
{
  "deepfm_rating": 4.2,
  "multitower_rating": 4.5,
  "ensemble_rating": 4.35,
  "confidence": 0.85
}
```


**처리 로직:**

1. user_id, business_id로 DB에서 피처 조회
2. 신규 사용자인 경우 평균값 사용
3. DeepFM, Multi-Tower 모델 로딩 및 예측
4. 앙상블 결과 반환

### 파일

- `backend_model/main.py` (업데이트)
- `backend_model/prediction_service.py` (새로 생성)

## Phase 4: 프론트엔드 UI 업데이트

### 메인 홈페이지 (옵션 4: 간결)

**비즈니스 리스트 카드에 추가:**

```
🏪 [가게 이름]
⭐ 4.2 | AI 예상: 4.5 (DeepFM) / 4.3 (Multi-Tower)
맛(96%) 서비스(88%) 분위기(75%)
```

**구현:**

- `/api/businesses` 응답에 상위 3-5개 ABSA 특징 추가
- 로그인 사용자면 `/predict_rating` 호출하여 예측 표시

### BusinessDetail 페이지 (옵션 2: 상세)

**AI 예측 섹션 추가:**

```
🤖 AI 예상 별점
⭐ DeepFM: 4.2
⭐ Multi-Tower: 4.5
⭐ 앙상블: 4.35
```

**ABSA 특징 섹션 추가:**

```
📍 이 가게의 특징 (리뷰 분석)

🍽️ 음식 관련
  맛        ████████░░ 85% 긍정
  품질/신선도 ███████░░░ 78% 긍정
  양        ██████░░░░ 65% 긍정

🙋 서비스
  서비스     ████████░░ 82% 긍정
  대기      ███░░░░░░░ 35% 부정

💰 가격/가치
  가격      ████░░░░░░ 45% 부정

🏠 분위기/시설
  분위기     ███████░░░ 75% 긍정
  청결도     ████████░░ 80% 긍정
  소음      ████░░░░░░ 42% 중립
```

**구현:**

- ABSA JSON 데이터를 파싱하여 카테고리별 그룹화
- 프로그레스 바 컴포넌트 생성
- 긍정/부정/중립 비율 계산 및 표시

### 파일

- `frontend/src/pages/HomePage.js` (업데이트)
- `frontend/src/pages/BusinessDetailPage.js` (업데이트)
- `frontend/src/components/AIPrediction.js` (새로 생성)
- `frontend/src/components/ABSAFeatures.js` (새로 생성)
- `frontend/src/components/ProgressBar.js` (새로 생성)

## Phase 5: backend_web API 업데이트

### 엔드포인트 수정

**GET /businesses**

- ABSA 상위 특징 추가 (상위 5개)

**GET /businesses/{id}**

- 전체 ABSA JSON 반환
- AI 예측 포함 (로그인 사용자)

### 파일

- `backend_web/main.py` (업데이트)
- `backend_web/schemas.py` (업데이트)

## Phase 6: 통합 테스트

### 테스트 항목

1. PostgreSQL 연결 및 데이터 확인
2. API 엔드포인트 테스트

   - `/predict_rating` (DeepFM, Multi-Tower)
   - `/businesses` (ABSA 특징 포함)

3. 프론트엔드 UI 확인

   - 홈페이지 예측 표시
   - BusinessDetail ABSA 특징 표시

4. 신규 사용자 처리 테스트
5. Yelp 가상 사용자 로그인 테스트

### 파일

- `scripts/test_integration.py`

## 주요 파일 목록

### Backend

- `backend_web/models.py` (업데이트)
- `backend_web/main.py` (업데이트)
- `backend_web/schemas.py` (업데이트)
- `backend_model/main.py` (업데이트)
- `backend_model/prediction_service.py` (신규)

### Frontend

- `frontend/src/pages/HomePage.js` (업데이트)
- `frontend/src/pages/BusinessDetailPage.js` (업데이트)
- `frontend/src/components/AIPrediction.js` (신규)
- `frontend/src/components/ABSAFeatures.js` (신규)
- `frontend/src/components/ProgressBar.js` (신규)

### Scripts

- `scripts/setup_postgresql.py` (신규)
- `scripts/migrate_data_to_postgresql.py` (신규)
- `scripts/convert_absa_to_json.py` (신규)
- `scripts/test_integration.py` (신규)

## 예상 소요 시간

- PostgreSQL 설정: 30분
- 데이터 마이그레이션: 1시간
- 예측 API 구현: 1시간
- 프론트엔드 UI: 2-3시간
- 테스트 및 디버깅: 1시간
- **총: 5-6시간**

## 주요 기술 결정

1. **DB**: PostgreSQL (JSON 컬럼 지원)
2. **ABSA 저장**: JSON 컬럼 (51개 값)
3. **사용자 처리**: Yelp 42k 가상 회원 + 신규 회원 평균값
4. **UI 표시**: 

   - 홈: 간결 (상위 3-5개)
   - Detail: 상세 (카테고리별 그룹)

5. **예측**: DeepFM + Multi-Tower 앙상블

### To-dos

- [ ] PostgreSQL 설정 및 스키마 정의
- [ ] 기존 app.db 백업
- [ ] User 데이터 마이그레이션 (42k Yelp 사용자)
- [ ] Business 데이터 마이그레이션 (14k 가게)
- [ ] Review 데이터 마이그레이션 (100k 리뷰 + ABSA)
- [ ] 예측 API 추가 (backend_model)
- [ ] backend_web API 업데이트 (ABSA 포함)
- [ ] 홈페이지 UI 업데이트 (AI 예측 + 간결 ABSA)
- [ ] BusinessDetail UI 업데이트 (상세 ABSA)
- [ ] 통합 테스트 및 디버깅