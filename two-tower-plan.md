# 실시간 Two-Tower 추천 시스템 실행 계획

## 📋 프로젝트 의사결정 요약

- **데이터**: Yelp Open Dataset (전처리 후 사용)
- **프론트엔드**: React
- **범위**: Phase 0-6 전체 완성
- **모델 복잡도**: 중간 수준 (Transformer + MLP)
- **개발 방식**: 각 단계 완벽 완성 + 테스트 후 진행

## 🏗️ 아키텍처 개요

```
[Frontend: React] 
    ↓ (Nginx)
[Tier 2: FastAPI Web Backend - 인증/DB/게이트웨이]
    ↓
[Tier 3: FastAPI Model API - 추론 전용]
    ↓
[FAISS Vector DB + Two-Tower Models]
```

---

## Phase 0: 프로젝트 설정 및 환경 구성

### 0.1 프로젝트 구조 생성

- [ ] Git 저장소 초기화 (Monorepo 방식)
- [ ] 루트 디렉터리 구조 생성:
  ```
  demo/
  ├── frontend/           # React 앱
  ├── backend_web/        # Tier 2: 웹 백엔드
  ├── backend_model/      # Tier 3: 모델 API
  ├── data/              # 원본 및 전처리 데이터
  ├── models/            # 학습된 모델 파일
  ├── scripts/           # 유틸리티 스크립트
  ├── .github/workflows/ # CI/CD
  └── docs/              # 문서
  ```

- [ ] `.gitignore` 생성 (Python, Node, 데이터 파일 제외)

### 0.2 문서화

- [ ] `README.md` 생성: 아키텍처 다이어그램, 실행 방법 작성
- [ ] `docs/DECISIONS.md` 생성: 8가지 주요 의사결정 기록
- [ ] 이 실행 계획을 `docs/EXECUTION_PLAN.md`로 저장

### 0.3 Backend 환경 설정

- [ ] `backend_web/` Python 가상환경 생성 및 활성화
- [ ] `backend_web/requirements.txt` 생성:
  ```
  fastapi==0.104.1
  uvicorn[standard]==0.24.0
  sqlalchemy==2.0.23
  httpx==0.25.2
  python-jose[cryptography]==3.3.0
  passlib[bcrypt]==1.7.4
  python-multipart==0.0.6
  pydantic==2.5.0
  pydantic-settings==2.1.0
  ```

- [ ] `backend_model/` Python 가상환경 생성 및 활성화
- [ ] `backend_model/requirements.txt` 생성:
  ```
  fastapi==0.104.1
  uvicorn[standard]==0.24.0
  torch==2.1.1
  transformers==4.36.0
  faiss-cpu==1.7.4
  numpy==1.24.3
  pydantic==2.5.0
  ```


### 0.4 Frontend 환경 설정

- [ ] `npx create-react-app frontend` 실행
- [ ] 필수 패키지 설치:
  ```bash
  cd frontend
  npm install axios react-router-dom@6
  ```

- [ ] 프로젝트 구조 확인 및 개발 서버 테스트 실행

### ✅ Phase 0 완료 기준

- [ ] 모든 디렉터리와 가상환경이 생성됨
- [ ] 각 백엔드의 requirements.txt로 패키지 설치 완료
- [ ] React 개발 서버가 정상 실행됨 (`npm start`)
- [ ] Git 커밋 완료

---

## Phase 1: 모델 학습 및 벡터 인덱스 구축

### 1.1 Yelp 데이터셋 다운로드 및 이해

- [ ] Yelp Open Dataset 다운로드 (yelp_academic_dataset_*.json)
- [ ] 필요 파일 확인:
                                                                - `yelp_academic_dataset_user.json`
                                                                - `yelp_academic_dataset_business.json`
                                                                - `yelp_academic_dataset_review.json`
- [ ] 데이터 스키마 분석 및 문서화

### 1.2 데이터 전처리

- [ ] `scripts/preprocess_yelp.py` 작성
- [ ] User 데이터 전처리:
                                                                - user_id, review_count, useful, funny, cool 추출
                                                                - 더미 age/gender 생성 (데모용)
- [ ] Business 데이터 전처리:
                                                                - business_id, name, categories, stars, review_count, attributes 추출
                                                                - 텍스트 정규화 (카테고리, 속성)
- [ ] Review 데이터 전처리:
                                                                - user_id, business_id, stars, text, date 추출
                                                                - 최근 N개월 데이터만 사용 (성능 최적화)
- [ ] 전처리된 데이터를 `data/processed/` 에 저장 (CSV 또는 Parquet)

### 1.3 학습 데이터셋 생성

- [ ] `scripts/create_train_data.py` 작성
- [ ] Positive 샘플 생성:
                                                                - (user, business, 1) - 실제 리뷰가 있고 평점 ≥ 4
- [ ] Negative 샘플 생성:
                                                                - In-batch negatives 또는 랜덤 샘플링
                                                                - Positive : Negative = 1:4 비율
- [ ] Train/Valid/Test 스플릿 (80/10/10)
- [ ] PyTorch Dataset 클래스 구현

### 1.4 Two-Tower 모델 정의

- [ ] `backend_model/models/two_tower.py` 작성
- [ ] **UserTower** 클래스:
  ```python
  Input: user_id (임베딩), user_features (review_count 등)
  Architecture: Embedding + MLP (256→128)
  Output: 128-dim vector
  ```

- [ ] **ItemTower** 클래스:
  ```python
  Input: business_id (임베딩), categories (텍스트), attributes
  Architecture: Embedding + Text Encoder (DistilBERT) + MLP (256→128)
  Output: 128-dim vector
  ```

- [ ] **CombinedModel** 클래스:
                                                                - Dot product similarity
                                                                - Binary Cross Entropy Loss

### 1.5 모델 학습

- [ ] `scripts/train_two_tower.py` 작성
- [ ] 학습 설정:
                                                                - Optimizer: AdamW (lr=1e-4)
                                                                - Batch size: 256
                                                                - Epochs: 10-20
                                                                - Loss: BCE with Logits
- [ ] Training loop 구현:
                                                                - Progress bar (tqdm)
                                                                - Validation 평가 (AUC, Recall@K)
                                                                - Early stopping
                                                                - 체크포인트 저장
- [ ] 학습 실행 및 로그 기록
- [ ] 최종 모델 저장:
                                                                - `models/user_tower.pth`
                                                                - `models/item_tower.pth`

### 1.6 FAISS 인덱스 구축

- [ ] `scripts/build_faiss_index.py` 작성
- [ ] ItemTower 로드 및 추론 모드 전환
- [ ] 모든 Business 아이템을 ItemTower에 통과:
                                                                - 배치 처리 (batch_size=512)
                                                                - item_vectors: shape (num_businesses, 128)
- [ ] FAISS 인덱스 생성:
  ```python
  import faiss
  index = faiss.IndexFlatIP(128)  # Inner Product (Cosine similarity)
  faiss.normalize_L2(item_vectors)
  index.add(item_vectors)
  ```

- [ ] 인덱스 저장: `models/index.faiss`
- [ ] ID 매핑 파일 생성: `models/idx_to_business_id.json`
  ```json
  {"0": "business_abc123", "1": "business_def456", ...}
  ```


### ✅ Phase 1 완료 기준

- [ ] 학습 완료, Validation AUC ≥ 0.75
- [ ] `models/` 에 3개 파일 존재: user_tower.pth, item_tower.pth, index.faiss
- [ ] `models/idx_to_business_id.json` 파일 생성
- [ ] 간단한 추론 테스트 스크립트 실행 성공

---

## Phase 2: Tier 3 - 모델 API 서버 개발

### 2.1 FastAPI 앱 초기화

- [ ] `backend_model/main.py` 생성
- [ ] FastAPI 앱 인스턴스 생성
- [ ] CORS 설정 (개발 환경용)
- [ ] Health check 엔드포인트: `GET /health`

### 2.2 모델 로더 구현

- [ ] `backend_model/model_loader.py` 작성
- [ ] `load_user_tower()` 함수:
                                                                - PyTorch 모델 로드
                                                                - GPU/CPU 자동 선택
                                                                - 추론 모드 전환
- [ ] `load_faiss_index()` 함수:
                                                                - FAISS 인덱스 로드
                                                                - ID 매핑 JSON 로드
- [ ] Startup 이벤트에서 모델 로드:
  ```python
  @app.on_event("startup")
  async def startup_event():
      app.state.user_tower = load_user_tower()
      app.state.faiss_index, app.state.id_map = load_faiss_index()
  ```


### 2.3 Pydantic 스키마 정의

- [ ] `backend_model/schemas.py` 작성
- [ ] `RecommendRequest`:
  ```python
  user_id: str
  recent_business_ids: List[str] = []
  context: Optional[dict] = None
  top_k: int = 10
  ```

- [ ] `RecommendResponse`:
  ```python
  recommendations: List[str]  # business_ids
  scores: List[float]
  ```


### 2.4 추천 엔드포인트 구현

- [ ] `POST /recommend` 엔드포인트 구현
- [ ] 로직:

                                                                1. Request 검증
                                                                2. User features 준비
                                                                3. UserTower로 user_vector 생성
                                                                4. FAISS search: `distances, indices = index.search(user_vector, top_k)`
                                                                5. indices → business_ids 변환
                                                                6. Response 반환

- [ ] 에러 핸들링 추가

### 2.5 테스트

- [ ] `uvicorn backend_model.main:app --reload --port 8001` 실행
- [ ] `curl` 또는 Postman으로 테스트:
  ```bash
  curl -X POST http://localhost:8001/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id": "test_user", "top_k": 5}'
  ```

- [ ] 정상 응답 확인 (business_ids 리스트 반환)

### ✅ Phase 2 완료 기준

- [ ] 모델 API 서버가 8001 포트에서 실행됨
- [ ] `/recommend` 엔드포인트가 정상 동작
- [ ] 추천 결과가 10개 이내로 반환됨
- [ ] 로그에 에러가 없음

---

## Phase 3: Tier 2 - 웹 백엔드 서버 개발

### 3.1 데이터베이스 설정

- [ ] `backend_web/database.py` 작성
- [ ] SQLAlchemy 엔진 및 세션 설정 (SQLite: `sqlite:///./app.db`)
- [ ] Base 클래스 정의

### 3.2 데이터베이스 모델 정의

- [ ] `backend_web/models.py` 작성
- [ ] **User** 모델:
  ```python
  id, username (unique), email, hashed_password
  age, gender, created_at
  ```

- [ ] **Business** 모델:
  ```python
  business_id (PK), name, categories, stars
  review_count, address, city, state
  ```

- [ ] **Review** 모델:
  ```python
  id, user_id (FK), business_id (FK)
  stars, text, created_at
  ```

- [ ] Relationship 설정

### 3.3 데이터베이스 초기화 및 시드

- [ ] `scripts/init_db.py` 작성
- [ ] 테이블 생성
- [ ] Yelp 전처리 데이터를 DB에 삽입:
                                                                - Business 테이블 (전체 또는 샘플)
                                                                - 테스트용 User 2-3명 생성
- [ ] DB 초기화 실행 및 확인

### 3.4 인증 시스템 구현

- [ ] `backend_web/auth.py` 작성
- [ ] 비밀번호 해시 함수 (bcrypt)
- [ ] JWT 토큰 생성/검증 함수
- [ ] `get_current_user` dependency 함수

### 3.5 Pydantic 스키마

- [ ] `backend_web/schemas.py` 작성
- [ ] UserCreate, UserLogin, Token, UserResponse
- [ ] BusinessResponse, ReviewCreate, ReviewResponse
- [ ] RecommendationResponse

### 3.6 FastAPI 앱 및 엔드포인트 구현

- [ ] `backend_web/main.py` 생성
- [ ] **인증 엔드포인트**:
                                                                - `POST /api/auth/register`: 회원가입
                                                                - `POST /api/auth/login`: 로그인 (JWT 발급)
                                                                - `GET /api/auth/me`: 현재 유저 정보
- [ ] **비즈니스 엔드포인트**:
                                                                - `GET /api/businesses`: 비즈니스 목록 (페이징)
                                                                - `GET /api/businesses/{business_id}`: 상세 정보
- [ ] **리뷰 엔드포인트**:
                                                                - `POST /api/businesses/{business_id}/reviews`: 리뷰 작성 (인증 필요)
                                                                - `GET /api/businesses/{business_id}/reviews`: 리뷰 목록
- [ ] **추천 엔드포인트**:
                                                                - `GET /api/recommendations`: 개인화 추천 (인증 필요)
                                                                - 로직:

                                                                                                                                1. 현재 유저 정보 조회 (age, gender 등)
                                                                                                                                2. 유저의 최근 리뷰한 business_ids 조회
                                                                                                                                3. Tier 3 API 호출 (httpx.AsyncClient)
                                                                                                                                4. 추천 결과를 Business 상세 정보와 함께 반환

### 3.7 Tier 3 연동

- [ ] `backend_web/services/model_service.py` 작성
- [ ] `get_recommendations()` 함수:
  ```python
  async def get_recommendations(user_id, recent_ids, top_k=10):
      async with httpx.AsyncClient() as client:
          response = await client.post(
              "http://localhost:8001/recommend",
              json={"user_id": user_id, "recent_business_ids": recent_ids, "top_k": top_k}
          )
          return response.json()
  ```


### 3.8 테스트

- [ ] `uvicorn backend_web.main:app --reload --port 8000` 실행
- [ ] 회원가입 테스트
- [ ] 로그인 후 JWT 토큰 받기
- [ ] 토큰으로 `/api/recommendations` 호출 테스트
- [ ] 리뷰 작성 테스트

### ✅ Phase 3 완료 기준

- [ ] 웹 백엔드가 8000 포트에서 실행됨
- [ ] 모든 API 엔드포인트가 정상 동작
- [ ] 인증 시스템 동작 확인
- [ ] Tier 3와 통신하여 추천 결과 반환 성공

---

## Phase 4: Tier 1 - React 프론트엔드 개발

### 4.1 프로젝트 구조 정리

- [ ] `src/` 디렉터리 구조:
  ```
  src/
  ├── components/   # 재사용 컴포넌트
  ├── pages/        # 페이지 컴포넌트
  ├── services/     # API 서비스
  ├── context/      # Context API
  ├── App.js
  └── index.js
  ```


### 4.2 API 클라이언트 설정

- [ ] `src/services/api.js` 작성
- [ ] axios 인스턴스 생성:
  ```javascript
  const api = axios.create({
    baseURL: 'http://localhost:8000/api'
  });
  ```

- [ ] Request 인터셉터: LocalStorage에서 토큰 읽어서 헤더에 추가
- [ ] Response 인터셉터: 401 에러 시 로그아웃 처리

### 4.3 인증 Context 구현

- [ ] `src/context/AuthContext.js` 작성
- [ ] AuthProvider 컴포넌트:
                                                                - 로그인/로그아웃 함수
                                                                - 현재 유저 상태 관리
                                                                - 토큰 LocalStorage 저장/삭제
- [ ] useAuth 훅 제공

### 4.4 라우팅 설정

- [ ] `src/App.js` 수정
- [ ] React Router 설정:
  ```javascript
  <BrowserRouter>
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/" element={<PrivateRoute><HomePage /></PrivateRoute>} />
      <Route path="/business/:id" element={<PrivateRoute><BusinessDetailPage /></PrivateRoute>} />
    </Routes>
  </BrowserRouter>
  ```

- [ ] PrivateRoute 컴포넌트 구현

### 4.5 페이지 컴포넌트 구현

#### 4.5.1 LoginPage

- [ ] `src/pages/LoginPage.js` 작성
- [ ] 이메일/비밀번호 폼
- [ ] 로그인 버튼 클릭 시 `/api/auth/login` 호출
- [ ] 성공 시 토큰 저장 및 홈으로 이동

#### 4.5.2 RegisterPage

- [ ] `src/pages/RegisterPage.js` 작성
- [ ] 회원가입 폼 (username, email, password, age, gender)
- [ ] `/api/auth/register` 호출 후 로그인 페이지로 이동

#### 4.5.3 HomePage

- [ ] `src/pages/HomePage.js` 작성
- [ ] useEffect에서 `/api/recommendations` 호출
- [ ] 로딩 상태 표시
- [ ] 추천 비즈니스 목록을 카드 형태로 렌더링
- [ ] 각 카드 클릭 시 상세 페이지로 이동

#### 4.5.4 BusinessDetailPage

- [ ] `src/pages/BusinessDetailPage.js` 작성
- [ ] useParams로 business_id 획득
- [ ] `/api/businesses/{id}` 호출하여 상세 정보 표시
- [ ] 리뷰 목록 표시
- [ ] ReviewForm 컴포넌트 포함

### 4.6 주요 컴포넌트 구현

- [ ] `src/components/BusinessCard.js`: 비즈니스 카드
- [ ] `src/components/ReviewForm.js`: 리뷰 작성 폼
                                                                - 별점, 텍스트 입력
                                                                - 제출 시 `/api/businesses/{id}/reviews` POST
                                                                - 성공 시 이벤트 발생 (Context 또는 callback)
- [ ] `src/components/ReviewList.js`: 리뷰 목록

### 4.7 실시간 업데이트 구현

- [ ] 리뷰 작성 후 HomePage 추천 목록 자동 갱신 로직
- [ ] 방법 1: Context API로 전역 상태 관리
- [ ] 방법 2: ReviewForm에서 작성 완료 시 부모에게 이벤트 전달
- [ ] HomePage에서 갱신 트리거 받으면 `/api/recommendations` 재호출

### 4.8 UI/UX 개선

- [ ] 기본 CSS 또는 Tailwind CSS 추가
- [ ] 로딩 스피너 컴포넌트
- [ ] 에러 메시지 표시
- [ ] 반응형 디자인 적용

### 4.9 테스트

- [ ] `npm start`로 개발 서버 실행
- [ ] 전체 플로우 테스트:

                                                                1. 회원가입
                                                                2. 로그인
                                                                3. 추천 목록 확인
                                                                4. 비즈니스 상세 페이지 이동
                                                                5. 리뷰 작성
                                                                6. 홈으로 돌아가서 추천 목록 변경 확인

### ✅ Phase 4 완료 기준

- [ ] React 앱이 정상 실행됨
- [ ] 모든 페이지가 정상 렌더링됨
- [ ] 백엔드와 통신이 정상적으로 이루어짐
- [ ] 리뷰 작성 후 추천 목록 갱신 동작 확인
- [ ] UI가 깔끔하고 사용하기 편함

---

## Phase 5: 통합 배포 (모놀리식 서버)

### 5.1 서버 준비

- [ ] 클라우드 VM 생성 (AWS EC2, GCP Compute Engine 등)
- [ ] OS: Ubuntu 22.04 LTS
- [ ] 방화벽: 22 (SSH), 80 (HTTP), 443 (HTTPS) 포트 오픈
- [ ] SSH 접속 확인

### 5.2 서버 환경 설정

- [ ] 시스템 업데이트:
  ```bash
  sudo apt update && sudo apt upgrade -y
  ```

- [ ] 필수 패키지 설치:
  ```bash
  sudo apt install -y nginx python3-pip python3-venv nodejs npm git
  ```

- [ ] Git 리포지토리 클론

### 5.3 Frontend 빌드 및 배포

- [ ] `cd frontend && npm install` 실행
- [ ] `.env.production` 파일 생성:
  ```
  REACT_APP_API_URL=/api
  ```

- [ ] `npm run build` 실행
- [ ] `build/` 디렉터리 생성 확인

### 5.4 Backend 배포 준비

- [ ] 모델 파일 복사:
  ```bash
  scp models/* user@server:/path/to/backend_model/models/
  ```

- [ ] 또는 서버에서 학습 실행 (시간 소요)

### 5.5 Backend_web 실행

- [ ] 가상환경 생성 및 패키지 설치:
  ```bash
  cd backend_web
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

- [ ] DB 초기화: `python scripts/init_db.py`
- [ ] Gunicorn으로 실행:
  ```bash
  gunicorn -k uvicorn.workers.UvicornWorker backend_web.main:app --bind 0.0.0.0:8000 --daemon
  ```

- [ ] 프로세스 확인: `ps aux | grep gunicorn`

### 5.6 Backend_model 실행

- [ ] 가상환경 생성 및 패키지 설치
- [ ] Gunicorn으로 실행:
  ```bash
  gunicorn -k uvicorn.workers.UvicornWorker backend_model.main:app --bind 0.0.0.0:8001 --daemon
  ```


### 5.7 Nginx 설정

- [ ] `/etc/nginx/sites-available/default` 수정:
  ```nginx
  server {
      listen 80;
      server_name your_domain_or_ip;
  
      # Frontend
      location / {
          root /path/to/frontend/build;
          try_files $uri /index.html;
      }
  
      # API Proxy
      location /api/ {
          proxy_pass http://localhost:8000/api/;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
      }
  }
  ```

- [ ] Nginx 설정 테스트: `sudo nginx -t`
- [ ] Nginx 재시작: `sudo systemctl restart nginx`

### 5.8 통합 테스트

- [ ] 브라우저에서 `http://your_server_ip` 접속
- [ ] 회원가입, 로그인, 추천 받기, 리뷰 작성 전체 플로우 테스트
- [ ] 네트워크 탭에서 API 호출 확인
- [ ] 에러 로그 확인:
  ```bash
  tail -f /var/log/nginx/error.log
  ```


### 5.9 프로세스 관리 (Optional but Recommended)

- [ ] Systemd 서비스 파일 생성:
                                                                - `/etc/systemd/system/backend_web.service`
                                                                - `/etc/systemd/system/backend_model.service`
- [ ] 서비스 시작 및 자동 시작 설정:
  ```bash
  sudo systemctl enable backend_web
  sudo systemctl start backend_web
  ```


### ✅ Phase 5 완료 기준

- [ ] 서버의 공인 IP로 웹사이트 접속 가능
- [ ] 모든 기능이 정상 동작
- [ ] Backend 프로세스가 안정적으로 실행 중
- [ ] Nginx가 정상적으로 프록시 역할 수행

---

## Phase 6: 오프라인 파이프라인 자동화

### 6.1 AWS S3 설정

- [ ] AWS 계정 생성 (또는 GCS)
- [ ] S3 버킷 생성: `two-tower-model-assets`
- [ ] IAM 사용자 생성 및 S3 접근 권한 부여
- [ ] Access Key, Secret Key 저장

### 6.2 오프라인 파이프라인 스크립트

- [ ] `scripts/run_offline_pipeline.py` 작성
- [ ] 로직:

                                                                1. `train_two_tower.py` 실행
                                                                2. `build_faiss_index.py` 실행
                                                                3. 생성된 파일들을 S3에 업로드:

                                                                                                                                                                - `models/user_tower.pth`
                                                                                                                                                                - `models/item_tower.pth`
                                                                                                                                                                - `models/index.faiss`
                                                                                                                                                                - `models/idx_to_business_id.json`
- [ ] boto3 사용:
  ```python
  import boto3
  s3 = boto3.client('s3')
  s3.upload_file('models/index.faiss', 'bucket-name', 'index.faiss')
  ```

- [ ] 로컬에서 테스트 실행

### 6.3 GitHub Actions 워크플로

- [ ] `.github/workflows/daily_retrain.yml` 생성
- [ ] 트리거 설정:
  ```yaml
  on:
    schedule:
   - cron: '0 3 * * *'  # 매일 UTC 3시 (한국 시간 정오)
    workflow_dispatch:  # 수동 실행도 가능
  ```

- [ ] Job 정의:
  ```yaml
  jobs:
    retrain:
      runs-on: ubuntu-latest
      steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
          with:
            python-version: '3.10'
    - name: Install dependencies
          run: pip install -r requirements.txt
    - name: Run pipeline
          env:
            AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
            AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          run: python scripts/run_offline_pipeline.py
  ```

- [ ] GitHub Secrets에 AWS 자격 증명 등록

### 6.4 Tier 3 모델 Hot-Reload 구현

#### 방법 A: 단순 재시작 방식

- [ ] `backend_model/model_loader.py` 수정
- [ ] S3에서 파일 다운로드 함수 추가:
  ```python
  def download_from_s3():
      s3 = boto3.client('s3')
      s3.download_file('bucket', 'index.faiss', 'models/index.faiss')
      # ... 다른 파일들도
  ```

- [ ] Startup 이벤트에서 S3 다운로드 후 로드
- [ ] 서버 재시작 스크립트 작성

#### 방법 B: Hot-Swap 방식 (고급)

- [ ] `POST /model/reload` 엔드포인트 추가
- [ ] Secret Key 인증 (환경변수)
- [ ] 엔드포인트 로직:

                                                                1. S3에서 새 모델 다운로드
                                                                2. 메모리에 새 모델 로드
                                                                3. `app.state`의 모델/인덱스 교체
                                                                4. 이전 모델 메모리 해제

- [ ] GitHub Actions 마지막 단계에서 이 엔드포인트 호출:
  ```bash
  curl -X POST http://your_server_ip:8001/model/reload \
    -H "Authorization: Bearer $SECRET_KEY"
  ```


### 6.5 모니터링 및 로깅

- [ ] 파이프라인 실행 로그를 S3에 저장
- [ ] 학습 메트릭 (AUC, Loss) 기록
- [ ] 실패 시 이메일 알림 설정 (Optional)

### 6.6 테스트

- [ ] GitHub Actions에서 Workflow 수동 실행
- [ ] 파이프라인이 정상 완료되는지 확인
- [ ] S3에 파일이 업로드되었는지 확인
- [ ] 서버에서 새 모델이 로드되는지 확인
- [ ] 추천 결과가 변경되는지 확인

### ✅ Phase 6 완료 기준

- [ ] GitHub Actions 워크플로가 성공적으로 실행됨
- [ ] S3에 최신 모델 파일들이 저장됨
- [ ] 서버가 새 모델을 자동으로 로드함
- [ ] 전체 시스템이 안정적으로 동작함

---

## 🎯 최종 검증 체크리스트

- [ ] 사용자가 회원가입하고 로그인할 수 있다
- [ ] 개인화된 추천 목록이 표시된다
- [ ] 비즈니스 상세 페이지에서 정보를 볼 수 있다
- [ ] 리뷰를 작성할 수 있다
- [ ] 리뷰 작성 후 추천 목록이 업데이트된다
- [ ] 서버에 배포되어 외부에서 접근 가능하다
- [ ] 자동화 파이프라인이 정기적으로 실행된다
- [ ] 모델이 자동으로 업데이트된다

---

## 📝 개발 팁

1. **각 Phase 완료 후 Git Commit**: 롤백 가능하도록
2. **로그 활용**: 문제 발생 시 로그를 먼저 확인
3. **환경변수 관리**: `.env` 파일 사용, `.gitignore`에 추가
4. **단위 테스트**: 중요한 함수는 pytest로 테스트
5. **문서화**: 코드에 주석 및 README 업데이트

## 🚨 주의사항

- Yelp 데이터셋 크기가 크므로 샘플링 고려 (예: 특정 도시만)
- GPU 없이 CPU로 학습 시 시간이 오래 걸림 (Colab 활용 가능)
- S3 비용 발생 가능 (Free Tier 확인)
- 보안: JWT Secret, AWS Key를 절대 Git에 커밋하지 말 것