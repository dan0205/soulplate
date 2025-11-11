# ABSA 모델 통합 가이드

## 📋 완료된 작업

### ✅ Backend

1. **PostgreSQL 스키마 정의** (`backend_web/models.py`)
   - User, Business, Review 테이블에 ABSA JSON 컬럼 추가
   - Yelp 데이터 매칭을 위한 필드 추가

2. **데이터 마이그레이션 스크립트**
   - `scripts/setup_postgresql.py`: PostgreSQL 설정 및 테이블 생성
   - `scripts/migrate_data_to_postgresql.py`: 100k 데이터 마이그레이션

3. **예측 API** (`backend_model/`)
   - `prediction_service.py`: DeepFM + Multi-Tower 예측 서비스
   - `main.py`: POST /predict_rating 엔드포인트 추가

4. **Web API 업데이트** (`backend_web/`)
   - ABSA 헬퍼 함수: `get_top_absa_features()`, `get_ai_prediction()`
   - GET /api/businesses: 상위 ABSA 특징 + AI 예측 포함
   - GET /api/businesses/{id}: 전체 ABSA + AI 예측 포함

### ✅ Frontend Components

1. **AIPrediction.js**: AI 예측 별점 표시
2. **ProgressBar.js**: 진행률 바 (긍정/부정/중립)
3. **ABSAFeatures.js**: 
   - `ABSAFeaturesCompact`: 홈페이지용 간결 버전
   - `ABSAFeaturesDetailed`: 디테일 페이지용 상세 버전

---

## 🚀 실행 순서

### 1. PostgreSQL 설정 및 데이터 마이그레이션

```bash
# PostgreSQL 시작 확인
# Windows: PostgreSQL 서비스 확인
# Mac: brew services start postgresql

# 1. PostgreSQL 설정 및 테이블 생성
python scripts/setup_postgresql.py

# 2. Yelp 데이터 마이그레이션 (42k users + 14k businesses + 100k reviews)
python scripts/migrate_data_to_postgresql.py
```

### 2. Backend 서버 시작

```bash
# Terminal 1: Model API (포트 8001)
cd backend_model
python main.py

# Terminal 2: Web API (포트 8000)
cd backend_web
python main.py
```

### 3. Frontend 통합

#### HomePage.js 수정 예시

```javascript
import React from 'react';
import { ABSAFeaturesCompact } from '../components/ABSAFeatures';
import AIPrediction from '../components/AIPrediction';

// 비즈니스 카드 컴포넌트
const BusinessCard = ({ business }) => {
  return (
    <div className="business-card">
      <h3>{business.name}</h3>
      <p>⭐ {business.stars} | {business.review_count} reviews</p>
      
      {/* AI 예측 (로그인 사용자만) */}
      {business.ai_prediction && (
        <AIPrediction prediction={business.ai_prediction} />
      )}
      
      {/* 상위 ABSA 특징 */}
      <ABSAFeaturesCompact topFeatures={business.top_features} />
      
      <p>{business.categories}</p>
    </div>
  );
};
```

#### BusinessDetailPage.js 수정 예시

```javascript
import React, { useEffect, useState } from 'react';
import { ABSAFeaturesDetailed } from '../components/ABSAFeatures';
import AIPrediction from '../components/AIPrediction';
import api from '../services/api';

const BusinessDetailPage = ({ businessId }) => {
  const [business, setBusiness] = useState(null);

  useEffect(() => {
    const fetchBusiness = async () => {
      try {
        const response = await api.get(`/api/businesses/${businessId}`);
        setBusiness(response.data);
      } catch (error) {
        console.error('Failed to fetch business:', error);
      }
    };
    fetchBusiness();
  }, [businessId]);

  if (!business) return <div>Loading...</div>;

  return (
    <div className="business-detail">
      <h1>{business.name}</h1>
      <p>⭐ {business.stars} | {business.review_count} reviews</p>
      
      {/* AI 예측 (로그인 사용자만) */}
      {business.ai_prediction && (
        <AIPrediction prediction={business.ai_prediction} />
      )}
      
      {/* 상세 ABSA 특징 */}
      <ABSAFeaturesDetailed 
        absaFeatures={business.absa_features}
        topFeatures={business.top_features}
      />
      
      {/* 기존 리뷰 섹션... */}
    </div>
  );
};
```

---

## 📊 API 응답 예시

### GET /api/businesses (리스트)

```json
[
  {
    "id": 1,
    "business_id": "abc123",
    "name": "맛있는 식당",
    "stars": 4.2,
    "review_count": 523,
    "top_features": [
      {"aspect": "맛", "sentiment": "긍정", "score": 0.96},
      {"aspect": "서비스", "sentiment": "긍정", "score": 0.88},
      {"aspect": "가격", "sentiment": "부정", "score": 0.45}
    ],
    "ai_prediction": {
      "deepfm_rating": 4.2,
      "multitower_rating": 4.5,
      "ensemble_rating": 4.35
    }
  }
]
```

### GET /api/businesses/{id} (상세)

```json
{
  "id": 1,
  "business_id": "abc123",
  "name": "맛있는 식당",
  "stars": 4.2,
  "absa_features": {
    "맛_긍정": 0.96,
    "맛_부정": 0.02,
    "맛_중립": 0.02,
    "서비스_긍정": 0.88,
    "서비스_부정": 0.08,
    ...
  },
  "top_features": [...],
  "ai_prediction": {...}
}
```

---

## 🔧 트러블슈팅

### PostgreSQL 연결 실패
```bash
# DATABASE_URL 확인
echo $DATABASE_URL

# PostgreSQL 실행 확인
# Windows: services.msc에서 PostgreSQL 서비스 확인
# Mac: pg_isready
```

### 모델 로딩 실패
```bash
# 모델 파일 확인
ls -l models/deepfm_ranking.pth
ls -l models/multitower_ranking.pth
ls -l models/user_scaler.pkl
ls -l models/business_scaler.pkl
```

### CORS 오류
```python
# backend_web/main.py에서 CORS 설정 확인
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 프론트엔드 주소
    ...
)
```

---

## 📈 다음 단계

1. **프론트엔드 완전 통합**
   - HomePage.js, BusinessDetailPage.js 전체 수정
   - 스타일링 개선

2. **성능 최적화**
   - AI 예측 캐싱
   - ABSA 피처 인덱싱

3. **추가 기능**
   - 사용자별 추천 리스트 (AI 예측 높은 순)
   - ABSA 필터링 (맛 좋은 곳만 보기)
   - 비교 기능 (여러 가게의 ABSA 비교)

---

## 📝 주요 파일 목록

### Backend
- `backend_web/models.py`: DB 스키마
- `backend_web/main.py`: Web API
- `backend_web/schemas.py`: API 스키마
- `backend_web/auth.py`: 인증 (optional 추가)
- `backend_model/main.py`: Model API
- `backend_model/prediction_service.py`: 예측 서비스

### Scripts
- `scripts/setup_postgresql.py`
- `scripts/migrate_data_to_postgresql.py`

### Frontend
- `frontend/src/components/AIPrediction.js`
- `frontend/src/components/ABSAFeatures.js`
- `frontend/src/components/ProgressBar.js`

---

**작성일**: 2025-01-10
**버전**: 1.0.0

