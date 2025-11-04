# **🚀 개발 계획 문서: 실시간 Two-Tower 추천 시스템**

이 문서는 8단계의 아키텍처 설계를 기반으로 실제 개발을 위한 단계별 실행 계획과 체크리스트를 정의합니다.

**\[최종 아키텍처 요약\]**

* **모델:** Two-Tower (User-Tower, Item-Tower)  
* **Tier 1 (Frontend):** 모던 SPA (React/Vue)  
* **Tier 2 (Web Backend):** FastAPI (Python) \- 웹 로직/게이트웨이  
* **Tier 3 (Model API):** FastAPI (Python) \- 모델 추론 전용  
* **Vector DB:** FAISS (파일로 메모리에 로드)  
* **Deployment:** 모놀리식 (1개 서버) \+ Nginx  
* **Pipeline:** 오프라인 자동화 (GitHub Actions / Cronjob)

## **Phase 0: 프로젝트 설정 및 환경 구성**

모든 개발의 기초가 되는 환경을 설정합니다.

* \[ \] **\[공통\]** Git 저장소 생성 (Monorepo 또는 개별)  
* \[ \] **\[공통\]** README.md에 최종 아키텍처 다이어그램 및 8가지 '선택 기록(Decision Log)' 요약 추가  
* \[ \] **\[Tier 2/3\]** backend\_web/ 및 backend\_model/ 디렉터리 생성  
* \[ \] **\[Tier 2/3\]** 각 백엔드 디렉터리에 Python 가상 환경 설정 (venv)  
* \[ \] **\[Tier 2/3\]** 공통 라이브러리 설치 (pip install fastapi uvicorn\[standard\] python-multipart pydantic)  
* \[ \] **\[Tier 1\]** frontend/ 디렉터리 생성  
* \[ \] **\[Tier 1\]** React (npx create-react-app) 또는 Vue (npm init vue@latest) 프로젝트 생성  
* \[ \] **\[Tier 1\]** axios (API 통신용), react-router-dom (라우팅용) 설치

## **Phase 1: \[Offline\] 모델 학습 및 벡터 인덱스 구축 (The Heart ❤️)**

가장 중요합니다. 이 단계의 산출물(user\_tower.pth, item\_tower.pth, index.faiss)이 없으면 Tier 3가 동작할 수 없습니다.

* \[ \] **\[Data\]** 원본 데이터(User, Item, Review, ABSA) 로드 및 전처리 스크립트 작성  
* \[ \] **\[Train\]** train\_two\_tower.py 스크립트 작성  
  * \[ \] Pytorch/Tensorflow로 UserTower 모델 클래스 정의 (Input: user\_id, age, gender, recent\_items, context 등)  
  * \[ \] ItemTower 모델 클래스 정의 (Input: business\_id, review\_summary\_text, avg\_absa\_vector, content\_features 등) \[문서 참조\]  
  * \[ \] 두 타워를 결합한 CombinedTwoTowerModel 정의 (Dot product \+ Sigmoid Loss)  
  * \[ \] **(핵심)** 네거티브 샘플링(Negative Sampling) 로직 구현 (In-batch negatives 또는 Hard negatives)  
  * \[ \] (user, item\_positive, 1\) / (user, item\_negative, 0\) 쌍으로 학습 데이터셋 구축  
  * \[ \] 모델 학습(Training) 루프 실행  
* \[ \] **\[Export\]** 학습 완료 후, CombinedTwoTowerModel이 아닌 \*\*user\_tower.pth\*\*와 **item\_tower.pth** 모델 가중치를 개별 파일로 저장  
* \[ \] **\[Build Index\]** build\_faiss\_index.py 스크립트 작성  
  * \[ \] 저장된 item\_tower.pth 모델 로드  
  * \[ \] **모든** 아이템(가게) 데이터를 ItemTower에 통과시켜 item\_vector (e.g., \[1,000,000 x 128\]) 2D 배열 생성  
  * \[ \] faiss.IndexFlatIP (Dot Product용) 또는 IndexFlatL2 (L2 거리용) 인덱스 생성  
  * \[ \] index.add(item\_vectors)로 모든 벡터 주입  
  * \[ \] faiss.write\_index(index, "index.faiss")로 인덱스 파일 저장  
  * \[ \] **(중요)** FAISS 인덱스 ID(0, 1, 2...)를 실제 business\_id("store\_A", "store\_B"...)와 매핑하는 idx\_to\_business\_id.json 맵(Map) 파일 생성 및 저장

## **Phase 2: \[Tier 3\] 모델 API 서버 개발 (The Brain 🧠)**

Phase 1에서 생성된 자산(user\_tower.pth, index.faiss)을 서빙하는 API입니다.

* \[ \] **\[설정\]** backend\_model/ 디렉터리에서 작업  
* \[ \] **\[설정\]** requirements.txt에 faiss-cpu (또는 faiss-gpu), numpy, torch/tensorflow, transformers 추가  
* \[ \] **\[App\]** main.py 파일 생성 (FastAPI 앱 초기화)  
* \[ \] **\[Loader\]** model\_loader.py 작성  
  * \[ \] load\_user\_tower(): user\_tower.pth를 메모리에 로드하는 함수  
  * \[ \] load\_faiss\_index(): index.faiss와 idx\_to\_business\_id.json을 메모리에 로드하는 함수  
  * \[ \] FastAPI의 startup 이벤트를 사용해 앱 시작 시 위 함수들을 호출하고, 전역 변수(또는 app.state)에 모델/인덱스 저장  
* \[ \] **\[Schema\]** Pydantic을 사용한 입력/출력 스키마 정의  
  * \[ \] RecommendRequest: user\_id, age, gender, recent\_item\_ids, context, top\_k 등 (User Tower 입력과 일치)  
  * \[ \] RecommendResponse: recommendations: List\[str\] (가게 ID 리스트)  
* \[ \] **\[Endpoint\]** POST /recommend/ 엔드포인트 구현  
  * \[ \] RecommendRequest 바디 수신  
  * \[ \] 로드된 user\_tower 모델에 입력값을 전달하여 user\_vector (1x128) 생성  
  * \[ \] 로드된 faiss\_index.search(user\_vector, top\_k) 실행  
  * \[ \] 반환된 FAISS 인덱스 ID (\[5, 42, 101\])를 idx\_to\_business\_id.json 맵을 사용해 실제 business\_id 리스트 (\["store\_A", "store\_C", "store\_B"\])로 변환  
  * \[ \] RecommendResponse 형태로 JSON 응답 반환  
* \[ \] **\[Test\]** uvicorn으로 서버 (e.g., 8001 포트) 실행 후 curl 이나 'Postman'으로 POST /recommend/ 기능 정상 동작 확인

## **Phase 3: \[Tier 2\] 웹 백엔드 서버 개발 (The Body 💃)**

사용자 인증, DB 관리, 그리고 Tier 3로의 요청을 중개하는 게이트웨이입니다.

* \[ \] **\[설정\]** backend\_web/ 디렉터리에서 작업  
* \[ \] **\[설정\]** requirements.txt에 httpx (비동기 HTTP 클라이언트), sqlalchemy (ORM), passlib\[bcrypt\] (비밀번호 해시), python-jose\[cryptography\] (JWT) 추가  
* \[L\] **\[DB\]** database.py: DB 세션 설정 (데모용: sqlite, 운영용: postgresql)  
* \[ \] **\[DB\]** models.py: User (hashed\_password, age, gender 포함), Business (가게 정보), Review (유저가 작성한 리뷰) SQLAlchemy 모델 정의  
* \[ \] **\[Auth\]** auth.py: JWT 토큰 생성/검증, 비밀번호 해시/검증 유틸리티 함수 작성  
* \[ \] **\[Schema\]** Pydantic 스키마 정의 (UserCreate, UserLogin, Token, ReviewCreate 등)  
* \[ \] **\[Endpoint\]** POST /api/auth/register: 회원가입 (User 생성)  
* \[ \] **\[Endpoint\]** POST /api/auth/login: 로그인 (JWT 토큰 발급)  
* \[ \] **\[Endpoint\]** GET /api/items/{business\_id}: 가게 상세 정보 조회 (DB에서)  
* \[ \] **\[Endpoint\]** POST /api/items/{business\_id}/review: 리뷰 작성 (DB에 저장)  
* \[ \] **\[Endpoint\]** GET /api/recommendations/ (로그인 필요 Depends(get\_current\_user))  
  * \[ \] **(핵심)** 인증된 유저 정보(user)를 DB에서 조회 (age, gender 등)  
  * \[ \] 유저의 최근 활동(recent\_item\_ids, context)을 DB에서 조회  
  * \[ \] Tier 3 API(POST /recommend/)에 보낼 RecommendRequest 데이터 구성  
  * \[ \] httpx.AsyncClient를 사용해 Tier 3 (http://localhost:8001/recommend/)로 비동기 요청 전송  
  * \[ \] Tier 3로부터 받은 추천 결과(RecommendResponse)를 그대로 프론트엔드(Tier 1)에 전달

## **Phase 4: \[Tier 1\] 프론트엔드 UI 개발 (The Face 😎)**

사용자가 실제로 보는 화면입니다.

* \[ \] **\[설정\]** frontend/ 디렉터리에서 작업  
* \[ \] **\[API\]** src/api/apiClient.js (또는 services/api.js) 생성  
  * \[ \] axios 인스턴스 생성 (baseURL: /api \- Nginx 프록시 예정)  
  * \[ \] 요청/응답 인터셉터 설정 (LocalStorage에서 JWT 토큰을 읽어 Authorization 헤더에 자동 추가)  
* \[ \] **\[Routing\]** App.js에 react-router-dom 설정  
  * \[ \] /login, /register, / (Home), /item/:id 경로 정의  
  * \[ \] PrivateRoute 컴포넌트 구현 (로그인 안 했으면 /login으로 리디렉션)  
* \[ \] **\[Pages\]** LoginPage.js, RegisterPage.js 구현 (폼, /api/auth/ 호출)  
* \[ \] **\[Pages\]** HomePage.js 구현 (Private)  
  * \[ \] useEffect 훅에서 /api/recommendations/ 호출  
  * \[ \] 로딩 스피너 표시  
  * \[ \] 받아온 추천 목록(가게 ID)을 기반으로 ItemCard 컴포넌트 렌더링  
* \[ \] **\[Pages\]** ItemDetailPage.js 구현  
  * \[ \] useParams로 id 획득, /api/items/:id 호출  
  * \[ \] 가게 상세 정보 표시  
  * \[ \] 리뷰 작성 폼 (\<ReviewForm /\>) 포함  
* \[ \] **\[Component\]** ReviewForm.js 구현  
  * \[ \] 폼 제출 시 /api/items/:id/review 호출  
  * \[ \] **(핵심 UX)** 제출 성공 시, HomePage의 추천 목록을 **자동으로 다시 불러오도록** 상태 관리(Context API 또는 Recoil/Zustand) 트리거 (→ 실시간성 체감)

## **Phase 5: \[Deployment\] 통합 배포 (모놀리식 서버)**

3개의 애플리케이션을 1대의 서버에 올리고 연결합니다.

* \[ \] **\[서버\]** 클라우드 VM(EC2, GCP 등) 1대 준비 (Ubuntu 22.04 LTS 추천)  
* \[ \] **\[서버\]** nginx, python3-venv, npm 설치  
* \[ \] **\[서버\]** Git 리포지토리 클론  
* \[ \] **\[Tier 1\]** frontend/에서 npm install 및 npm run build 실행 (build 디렉터리 생성 확인)  
* \[ \] **\[Tier 2/3\]** index.faiss 등 Phase 1의 자산 파일들을 backend\_model/로 복사  
* \[ \] **\[Tier 2\]** gunicorn \+ uvicorn 워커로 backend\_web 앱 실행 (e.g., 8000 포트)  
  * \[ \] (참고: gunicorn \-k uvicorn.workers.UvicornWorker backend\_web.main:app \--bind 0.0.0.0:8000)  
* \[ \] **\[Tier 3\]** gunicorn \+ uvicorn 워커로 backend\_model 앱 실행 (e.g., 8001 포트)  
* \[ \] **\[Nginx\]** /etc/nginx/sites-available/default 설정  
  * \[ \] location / { ... }: frontend/build 디렉터리 정적 서빙  
  * \[ \] location /api/ { ... }: http://localhost:8000/ (Tier 2)로 리버스 프록시(Proxy Pass) 설정  
* \[ \] **\[Test\]** sudo systemctl restart nginx 실행 후, VM의 공인 IP(또는 도메인)로 접속하여 회원가입, 로그인, 추천 기능이 모두 동작하는지 최종 확인

## **Phase 6: \[Automation\] 오프라인 파이프라인 자동화 (The Lungs 🫁)**

데모가 '살아있음'을 보여주는 자동 업데이트 파이프라인입니다.

* \[ \] **\[Storage\]** AWS S3 (또는 GCS) 버킷 생성 (model-assets 등)  
* \[ \] **\[Script\]** scripts/run\_offline\_pipeline.py 마스터 스크립트 작성  
  * \[ \] Phase 1의 train\_two\_tower.py와 build\_faiss\_index.py 실행을 차례로 호출  
  * \[ \] boto3 라이브러리를 사용해 최종 산출물(index.faiss, idx\_to\_business\_id.json, user\_tower.pth)을 S3 버킷에 업로드  
* \[ \] **\[Automation\]** GitHub Actions 워크플로 파일(.github/workflows/daily\_retrain.yml) 작성  
  * \[ \] on: schedule: \- cron: '0 3 \* \* \*' (매일 새벽 3시 UTC)  
  * \[ \] AWS 자격 증명 (Access Key, Secret Key)을 GitHub Secrets에 등록  
  * \[ \] actions/checkout, actions/setup-python 설정  
  * \[ \] pip install \-r requirements.txt 실행  
  * \[ \] python scripts/run\_offline\_pipeline.py 실행  
* \[ \] **\[Tier 3 수정\]** Tier 3(모델 API)가 S3에서 최신 모델/인덱스를 읽어오도록 수정  
  * \[ \] **(방법 A \- 간단)** startup 이벤트에서 로컬 파일이 아닌 S3에서 파일을 다운로드하도록 model\_loader.py 수정. 서버 재시작 시 최신 파일 로드.  
  * \[ \] **(방법 B \- 고급)** /model/reload/ (Secret Key 필요) 엔드포인트 생성. 이 엔드포인트가 S3에서 새 파일을 다운로드하고 메모리의 모델/인덱스를 교체(Hot-swap)하도록 구현. GitHub Actions 마지막 단계에서 이 엔드포인트를 curl로 호출.