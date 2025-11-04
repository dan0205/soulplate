# 🚀 실시간 Two-Tower 추천 시스템

Yelp 데이터셋 기반의 실시간 개인화 추천 시스템입니다. Two-Tower 아키텍처를 사용하여 사용자와 비즈니스를 각각 임베딩하고, FAISS를 통해 빠른 유사도 검색을 수행합니다.

## 🏗️ 아키텍처

```
┌─────────────────────┐
│  Frontend (React)   │
│   Port: 3000/80     │
└──────────┬──────────┘
           │ (Nginx Proxy)
           ↓
┌─────────────────────┐
│  Backend Web        │
│  (FastAPI Tier 2)   │
│   Port: 8000        │
│  - 인증 & 권한      │
│  - DB 관리          │
│  - 게이트웨이       │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Backend Model      │
│  (FastAPI Tier 3)   │
│   Port: 8001        │
│  - 모델 추론        │
│  - FAISS 검색       │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Vector Database    │
│  (FAISS + Models)   │
│  - User Tower       │
│  - Item Tower       │
│  - Index            │
└─────────────────────┘
```

## 📋 주요 기술 스택

### Backend
- **Framework**: FastAPI (Python)
- **ML Framework**: PyTorch
- **Vector DB**: FAISS
- **Database**: SQLite (개발), PostgreSQL (운영)
- **Authentication**: JWT

### Frontend
- **Framework**: React
- **HTTP Client**: Axios
- **Routing**: React Router v6
- **State Management**: Context API

### Model
- **Architecture**: Two-Tower (User Tower + Item Tower)
- **Text Encoder**: DistilBERT
- **Embedding Dimension**: 128
- **Similarity**: Cosine Similarity (Inner Product)

### Deployment
- **Server**: Monolithic (1 VM)
- **Web Server**: Nginx
- **Process Manager**: Gunicorn + Uvicorn Workers
- **CI/CD**: GitHub Actions
- **Storage**: AWS S3 (Model Assets)

## 📁 프로젝트 구조

```
demo/
├── frontend/              # React 프론트엔드
│   ├── src/
│   │   ├── components/   # 재사용 컴포넌트
│   │   ├── pages/        # 페이지 컴포넌트
│   │   ├── services/     # API 클라이언트
│   │   └── context/      # 전역 상태 관리
│   └── package.json
├── backend_web/          # Tier 2: 웹 백엔드
│   ├── main.py           # FastAPI 앱
│   ├── models.py         # DB 모델
│   ├── schemas.py        # Pydantic 스키마
│   ├── auth.py           # 인증 로직
│   ├── database.py       # DB 설정
│   └── requirements.txt
├── backend_model/        # Tier 3: 모델 API
│   ├── main.py           # FastAPI 앱
│   ├── models/
│   │   └── two_tower.py  # Two-Tower 모델 정의
│   ├── model_loader.py   # 모델 로딩
│   ├── schemas.py        # Pydantic 스키마
│   └── requirements.txt
├── data/                 # 데이터
│   ├── raw/             # 원본 Yelp 데이터
│   └── processed/       # 전처리된 데이터
├── models/              # 학습된 모델
│   ├── user_tower.pth
│   ├── item_tower.pth
│   ├── index.faiss
│   └── idx_to_business_id.json
├── scripts/             # 유틸리티 스크립트
│   ├── preprocess_yelp.py
│   ├── create_train_data.py
│   ├── train_two_tower.py
│   ├── build_faiss_index.py
│   ├── init_db.py
│   └── run_offline_pipeline.py
├── .github/
│   └── workflows/
│       └── daily_retrain.yml
└── docs/                # 문서
    ├── DECISIONS.md
    └── EXECUTION_PLAN.md
```

## 🚀 빠른 시작

### 1. 데이터 준비
```bash
# Yelp 데이터셋 다운로드 (https://www.yelp.com/dataset)
# data/raw/ 폴더에 JSON 파일 배치

# 데이터 전처리
python scripts/preprocess_yelp.py
```

### 2. 모델 학습
```bash
# 학습 데이터셋 생성
python scripts/create_train_data.py

# Two-Tower 모델 학습
python scripts/train_two_tower.py

# FAISS 인덱스 구축
python scripts/build_faiss_index.py
```

### 3. Backend 실행

#### Tier 3: Model API
```bash
cd backend_model
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

#### Tier 2: Web Backend
```bash
cd backend_web
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# DB 초기화
python scripts/init_db.py

# 서버 실행
uvicorn main:app --reload --port 8000
```

### 4. Frontend 실행
```bash
cd frontend
npm install
npm start
```

브라우저에서 `http://localhost:3000` 접속

## 🔧 개발 단계

프로젝트는 7개의 Phase로 구성되어 있습니다:

- **Phase 0**: 프로젝트 설정 및 환경 구성 ✅
- **Phase 1**: 모델 학습 및 벡터 인덱스 구축
- **Phase 2**: Tier 3 - 모델 API 서버 개발
- **Phase 3**: Tier 2 - 웹 백엔드 서버 개발
- **Phase 4**: Tier 1 - React 프론트엔드 개발
- **Phase 5**: 통합 배포 (모놀리식 서버)
- **Phase 6**: 오프라인 파이프라인 자동화

자세한 내용은 `docs/EXECUTION_PLAN.md` 참조

## 📊 주요 기능

- **개인화 추천**: 사용자의 행동 이력 기반 실시간 추천
- **Two-Tower 아키텍처**: 효율적인 검색을 위한 분리된 임베딩
- **빠른 검색**: FAISS를 활용한 밀리초 단위 유사도 검색
- **실시간 업데이트**: 리뷰 작성 후 즉시 추천 목록 갱신
- **자동화 파이프라인**: GitHub Actions를 통한 일일 재학습
- **Hot-Reload**: 서비스 중단 없이 모델 업데이트

## 🔐 환경 변수

각 서비스별 `.env` 파일 설정:

### backend_web/.env
```
DATABASE_URL=sqlite:///./app.db
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
MODEL_API_URL=http://localhost:8001
```

### backend_model/.env
```
MODEL_PATH=../models
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
S3_BUCKET=two-tower-model-assets
```

## 📈 성능 메트릭

- **추론 지연시간**: ~50ms (FAISS 검색 포함)
- **처리량**: ~100 requests/sec (단일 서버)
- **모델 크기**: User Tower (~10MB), Item Tower (~50MB)
- **인덱스 크기**: ~500MB (100만 개 아이템 기준)

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 👥 개발팀

- **데이터 출처**: [Yelp Open Dataset](https://www.yelp.com/dataset)
- **개발 기간**: 2025년

## 📚 참고 자료

- [Two-Tower Neural Network Paper](https://research.google/pubs/pub48840/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

---

**Note**: 이 프로젝트는 교육 및 데모 목적으로 개발되었습니다.

