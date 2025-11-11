<!-- dc299d1c-36b6-4f9d-a8d7-d3628c5145da 23ae83a8-c6dc-44bb-8771-31a9d8ea7c95 -->
# 홈페이지 페이지네이션 및 AI 예측 표시 개선

## 현재 문제점

1. 홈페이지에서 상위 20개 음식점만 표시되고 페이지네이션 없음
2. AI 예측 별점이 하나만 표시됨 (DeepFM, Multi-Tower 구분 없음)
3. BusinessDetail 페이지에 AI 예측 및 ABSA 특징이 표시되지 않음

## 수정 계획

### 1. Backend API 수정 (backend_web/main.py)

`GET /api/businesses` 엔드포인트에 총 비즈니스 개수 반환 추가

- 응답에 `total` 필드 추가하여 전체 비즈니스 개수 반환
- 프론트엔드에서 총 페이지 수 계산에 사용

### 2. Frontend 홈페이지 수정 (frontend/src/pages/HomePage.js)

**상태 관리 추가:**

- `currentPage` (현재 페이지, 기본값 1)
- `totalPages` (총 페이지 수)
- `itemsPerPage` (페이지당 항목 수, 20)

**API 호출 수정:**

- `skip = (currentPage - 1) * itemsPerPage` 계산
- 응답에서 `total` 받아서 `totalPages` 계산

**AI 예측 표시 개선:**

- 카드에 `ai_prediction` 있으면 표시
- 형식: "⭐ 4.2 | AI 예상: 4.5 (DeepFM) / 4.3 (Multi-Tower)"
- `business.top_features` 표시 (맛, 서비스, 분위기 등)

**페이지네이션 UI 추가:**

- 이전/다음 버튼
- 페이지 번호 버튼 (현재 페이지 ±2 범위)
- 첫 페이지/마지막 페이지 버튼

### 3. BusinessDetail 페이지 수정 (frontend/src/pages/BusinessDetailPage.js)

**컴포넌트 import:**

- `AIPrediction` (이미 생성됨)
- `ABSAFeaturesDetailed` (이미 생성됨)

**AI 예측 섹션 추가:**

- business-header 아래에 AIPrediction 컴포넌트 추가
- `business.ai_prediction` 전달

**ABSA 특징 섹션 추가:**

- review-section 위에 ABSAFeaturesDetailed 컴포넌트 추가
- `business.absa_features` 전달

### 4. 스타일 추가 (frontend/src/pages/Home.css)

**페이지네이션 스타일:**

- `.pagination-container`: 페이지네이션 컨테이너
- `.pagination-button`: 페이지 버튼
- `.pagination-button.active`: 현재 페이지 버튼

**AI 예측 인라인 표시:**

- `.ai-prediction-inline`: 카드 내 AI 예측 표시 스타일

## 주요 파일 변경

- `backend_web/main.py`: total 개수 반환 추가
- `frontend/src/pages/HomePage.js`: 페이지네이션 및 AI 예측 표시 개선
- `frontend/src/pages/BusinessDetailPage.js`: AI 예측 및 ABSA 특징 컴포넌트 추가
- `frontend/src/pages/Home.css`: 페이지네이션 스타일 추가

## 예상 결과

**홈페이지:**

```
🏪 Restaurant Name
⭐ 4.2 | AI 예상: 4.5 (DeepFM) / 4.3 (Multi-Tower)
맛(96%) 서비스(88%) 분위기(75%)

[페이지네이션]
[처음] [이전] [1] [2] [3] [4] [5] [다음] [마지막]
```

**BusinessDetail 페이지:**

```
[가게 정보]

🤖 AI 예상 별점
⭐ DeepFM: 4.2
⭐ Multi-Tower: 4.5
⭐ 앙상블: 4.35

📍 이 가게의 특징 (리뷰 분석)
🍽️ 음식 관련
  맛        ████████░░ 85% 긍정
  품질/신선도 ███████░░░ 78% 긍정

[리뷰 작성 폼]
```

### To-dos

- [ ] backend_web/main.py - GET /api/businesses에 총 개수 반환 추가
- [ ] HomePage.js - 페이지네이션 상태 관리 및 UI 추가
- [ ] HomePage.js - AI 예측 표시 개선 (DeepFM/Multi-Tower 구분)
- [ ] Home.css - 페이지네이션 및 AI 예측 인라인 스타일 추가
- [ ] BusinessDetailPage.js - AIPrediction 및 ABSAFeaturesDetailed 컴포넌트 추가
- [ ] 전체 기능 테스트 (페이지네이션, AI 예측, ABSA 특징)