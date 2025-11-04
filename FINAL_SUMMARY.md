# 🎉 Two-Tower 추천 시스템 완성 보고서

## 프로젝트 개요

Yelp 데이터셋 기반의 실시간 개인화 추천 시스템을 Phase 0부터 Phase 6까지 완전히 구현했습니다.

## ✅ 완료된 Phase

### Phase 0: 프로젝트 설정 및 환경 구성 ✅
- Git 저장소 초기화
- 프로젝트 디렉터리 구조 생성
- Requirements.txt 작성
- React 앱 생성
- 문서화 (README, DECISIONS.md)

### Phase 1: 모델 학습 및 FAISS 인덱스 구축 ✅
- 샘플 Yelp 데이터 생성 (1000 users, 500 businesses, 5000 reviews)
- 데이터 전처리 스크립트
- Two-Tower 모델 구현 (PyTorch)
  - UserTower: 사용자 임베딩 (128차원)
  - ItemTower: 비즈니스 임베딩 (128차원)
- 모델 학습 (Validation AUC: 0.5338, Test AUC: 0.5467)
- FAISS 인덱스 구축 (500개 비즈니스 벡터)

### Phase 2: Tier 3 - 모델 API 서버 ✅
- FastAPI 기반 모델 서빙 API
- 모델 로더 (UserTower + FAISS 인덱스)
- `/recommend` 엔드포인트
- `/health`, `/model/info` 엔드포인트
- 모든 테스트 통과 (3/3)

### Phase 3: Tier 2 - 웹 백엔드 서버 ✅
- FastAPI 기반 웹 백엔드
- SQLAlchemy ORM (User, Business, Review 모델)
- JWT 기반 인증 시스템
- 데이터베이스 초기화 스크립트
- API 엔드포인트:
  - 인증: `/api/auth/register`, `/api/auth/login`, `/api/auth/me`
  - 비즈니스: `/api/businesses`, `/api/businesses/{id}`
  - 리뷰: `/api/businesses/{id}/reviews` (GET, POST)
  - 추천: `/api/recommendations`

### Phase 4: React 프론트엔드 ✅
- React SPA 구조
- API 클라이언트 (Axios with interceptors)
- AuthContext (로그인 상태 관리)
- 페이지 컴포넌트:
  - LoginPage: 로그인
  - RegisterPage: 회원가입
  - HomePage: 개인화 추천 목록 표시
  - BusinessDetailPage: 비즈니스 상세 정보 + 리뷰 작성
- PrivateRoute: 인증 보호
- 반응형 디자인 및 스타일링

### Phase 5: 배포 가이드 문서 ✅
- 완전한 배포 가이드 작성 (`docs/DEPLOYMENT_GUIDE.md`)
- Systemd 서비스 설정 예제
- Nginx 리버스 프록시 설정
- SSL 인증서 설정 (Let's Encrypt)
- 모니터링 및 트러블슈팅 가이드

### Phase 6: 자동화 파이프라인 ✅
- GitHub Actions 워크플로 (`.github/workflows/daily_retrain.yml`)
- 일일 자동 재학습 스케줄 (매일 UTC 3시)
- S3 업로드 스크립트 (`scripts/upload_to_s3.py`)
- 로컬 테스트 스크립트 (`scripts/run_local_test.sh`)

## 📊 프로젝트 통계

### 코드 통계
- **Python 파일**: 15개 이상
- **JavaScript/React 파일**: 12개 이상
- **총 코드 라인**: ~5,000+ 라인
- **문서 파일**: 5개

### 모델 성능
- **학습 데이터**: 7,995 샘플 (80%)
- **검증 데이터**: 1,000 샘플 (10%)
- **테스트 데이터**: 1,000 샘플 (10%)
- **Test AUC**: 0.5467
- **임베딩 차원**: 128
- **FAISS 인덱스**: 500 벡터

### 데이터베이스
- **사용자**: 3명 (테스트 계정)
- **비즈니스**: 500개
- **리뷰**: 최대 5,000개

## 🏗️ 아키텍처

```
┌─────────────────────┐
│  Frontend (React)   │  Port 3000
│  - Login/Register   │
│  - Recommendations  │
│  - Business Details │
└──────────┬──────────┘
           │ HTTP/REST
           ↓
┌─────────────────────┐
│  Web API (FastAPI)  │  Port 8000
│  - Authentication   │
│  - Business/Reviews │
│  - Gateway          │
└──────────┬──────────┘
           │ HTTP
           ↓
┌─────────────────────┐
│ Model API (FastAPI) │  Port 8001
│  - User Tower       │
│  - FAISS Search     │
│  - Recommendations  │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  FAISS + Models     │
│  - index.faiss      │
│  - user_tower.pth   │
│  - item_tower.pth   │
└─────────────────────┘
```

## 🚀 로컬 실행 방법

### 빠른 시작

```bash
# 1. Model API 실행
python -m uvicorn backend_model.main:app --host 0.0.0.0 --port 8001

# 2. Web API 실행 (새 터미널)
python -m uvicorn backend_web.main:app --host 0.0.0.0 --port 8000

# 3. Frontend 실행 (새 터미널)
cd frontend
npm start
```

### 통합 테스트 스크립트 (Linux/Mac)

```bash
chmod +x scripts/run_local_test.sh
./scripts/run_local_test.sh
```

### 접속

- **Frontend**: http://localhost:3000
- **Web API Docs**: http://localhost:8000/docs
- **Model API Docs**: http://localhost:8001/docs

### 데모 계정

- `testuser` / `test123`
- `alice` / `alice123`
- `bob` / `bob123`

## 📁 프로젝트 구조

```
demo/
├── frontend/              # React 앱
│   ├── src/
│   │   ├── components/   # PrivateRoute
│   │   ├── context/      # AuthContext
│   │   ├── pages/        # Login, Register, Home, BusinessDetail
│   │   └── services/     # API 클라이언트
│   └── package.json
├── backend_web/          # Tier 2: 웹 백엔드
│   ├── main.py           # FastAPI 앱
│   ├── models.py         # DB 모델
│   ├── schemas.py        # Pydantic 스키마
│   ├── auth.py           # JWT 인증
│   └── database.py       # DB 설정
├── backend_model/        # Tier 3: 모델 API
│   ├── main.py           # FastAPI 앱
│   ├── models/
│   │   └── two_tower.py  # Two-Tower 모델
│   ├── model_loader.py   # 모델 로더
│   └── schemas.py        # Pydantic 스키마
├── scripts/              # 유틸리티 스크립트
│   ├── generate_sample_data.py
│   ├── preprocess_yelp.py
│   ├── create_train_data.py
│   ├── train_two_tower.py
│   ├── build_faiss_index.py
│   ├── init_db.py
│   ├── upload_to_s3.py
│   └── run_local_test.sh
├── data/                 # 데이터
│   ├── raw/             # 원본 데이터
│   └── processed/       # 전처리 데이터
├── models/              # 학습된 모델
│   ├── user_tower.pth
│   ├── item_tower.pth
│   └── index.faiss
├── docs/                # 문서
│   ├── DECISIONS.md
│   ├── DEPLOYMENT_GUIDE.md
│   └── EXECUTION_PLAN.md
├── .github/workflows/   # CI/CD
│   └── daily_retrain.yml
├── app.db               # SQLite 데이터베이스
└── README.md            # 프로젝트 README
```

## 🎯 주요 기능

### 1. 개인화 추천
- Two-Tower 아키텍처 기반
- 실시간 유사도 검색 (FAISS)
- Top-K 추천 (기본 10개)

### 2. 사용자 인증
- JWT 기반 토큰 인증
- 회원가입 / 로그인
- 보안 비밀번호 해싱 (bcrypt)

### 3. 리뷰 시스템
- 별점 (1-5)
- 텍스트 리뷰
- 리뷰 목록 조회

### 4. 실시간 업데이트
- 리뷰 작성 후 즉시 반영
- 추천 목록 새로고침 기능

## 🛠️ 기술 스택

### Backend
- **Python 3.10+**
- **FastAPI** - 웹 프레임워크
- **PyTorch** - 딥러닝 프레임워크
- **FAISS** - 벡터 유사도 검색
- **SQLAlchemy** - ORM
- **SQLite** - 데이터베이스 (개발)
- **JWT** - 인증
- **Uvicorn** - ASGI 서버

### Frontend
- **React 18**
- **React Router v6** - 라우팅
- **Axios** - HTTP 클라이언트
- **Context API** - 상태 관리

### DevOps
- **GitHub Actions** - CI/CD
- **AWS S3** - 모델 스토리지 (계획)
- **Nginx** - 리버스 프록시
- **Systemd** - 프로세스 관리

## 📈 향후 개선 사항

### 모델
- [ ] Transformer 기반 더 복잡한 모델
- [ ] Hard negative mining
- [ ] Multi-task learning

### 시스템
- [ ] Redis 캐싱
- [ ] PostgreSQL 전환
- [ ] Docker 컨테이너화
- [ ] Kubernetes 오케스트레이션

### 기능
- [ ] 실시간 A/B 테스팅
- [ ] 사용자 프로필 페이지
- [ ] 고급 필터링
- [ ] 소셜 기능 (팔로우, 좋아요)

## 🎓 학습 내용

이 프로젝트를 통해 다음을 학습하고 구현했습니다:

1. **Two-Tower 아키텍처** - 추천 시스템 설계
2. **FAISS** - 대규모 벡터 검색
3. **FastAPI** - 고성능 API 서버
4. **React** - 모던 프론트엔드
5. **JWT 인증** - 보안 시스템
6. **CI/CD** - 자동화 파이프라인
7. **Full-Stack 개발** - 엔드투엔드 시스템

## 📝 참고 자료

- [Two-Tower Neural Networks](https://research.google/pubs/pub48840/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Yelp Open Dataset](https://www.yelp.com/dataset)

## 📧 Contact

프로젝트에 대한 질문이나 피드백은 환영합니다!

---

**프로젝트 완성일**: 2025-11-04
**개발 기간**: 1일 (집중 개발)
**총 Commits**: 6개 (Phase별 1개)

🎉 **Two-Tower 추천 시스템 프로젝트 완성!** 🎉

