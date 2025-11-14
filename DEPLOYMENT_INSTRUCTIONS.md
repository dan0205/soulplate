# 배포 가이드 (Railway + Vercel)

GitHub 저장소가 준비되었습니다! 이제 Railway와 Vercel에 배포하세요.

## 완료된 작업 ✅

1. ✅ 프론트엔드 API URL이 환경 변수를 사용하도록 수정됨 (`process.env.REACT_APP_API_URL`)
2. ✅ 대용량 모델 파일이 Git에서 제거됨
3. ✅ 모든 변경사항이 GitHub에 푸시됨

## 1단계: Railway에 백엔드 배포

Railway 한 곳에서 백엔드 2개 + PostgreSQL을 모두 관리합니다!

### A. Railway 프로젝트 생성

1. https://railway.app 접속 후 GitHub 계정으로 로그인
2. "New Project" 클릭
3. "Deploy from GitHub repo" 선택
4. 저장소 선택: `dan0205/soulplate`

### B. PostgreSQL 추가

1. 프로젝트 대시보드에서 "+ New" 클릭
2. "Database" → "PostgreSQL" 선택
3. 자동으로 생성됨 (1-2분 소요)
4. PostgreSQL 서비스 클릭 → "Variables" 탭
5. **DATABASE_URL 복사** (나중에 사용)
   - 형식: `postgresql://user:password@host:port/dbname`

### C. Model Backend 서비스 추가

1. 프로젝트 대시보드에서 "+ New" 클릭
2. "GitHub Repo" → 같은 저장소 선택
3. 서비스 설정:
   - **Root Directory**: `backend_model`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   
4. "Deploy" 클릭
5. 배포 완료 후 "Settings" 탭
6. "Networking" → "Public Networking" 활성화
7. **배포 URL 복사** (예: `https://backend-model-production-xxxx.up.railway.app`)

### D. Web Backend 서비스 추가

1. 프로젝트 대시보드에서 "+ New" 클릭
2. "GitHub Repo" → 같은 저장소 선택
3. 서비스 설정:
   - **Root Directory**: `backend_web`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **환경 변수 설정** (Variables 탭):
   ```
   DATABASE_URL=<B에서 복사한 PostgreSQL URL>
   SECRET_KEY=<강력한 랜덤 문자열 생성>
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   MODEL_API_URL=<C에서 복사한 Model Backend URL>
   ```
   
   SECRET_KEY 생성:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

5. "Deploy" 클릭
6. "Settings" → "Networking" → "Public Networking" 활성화
7. **배포 URL 복사** (예: `https://backend-web-production-xxxx.up.railway.app`)

## 2단계: Vercel에 프론트엔드 배포

1. https://vercel.com 접속 후 GitHub 계정으로 로그인
2. "Add New" → "Project" 클릭
3. GitHub에서 `dan0205/soulplate` 저장소 Import
4. 프로젝트 설정:
   ```
   Framework Preset: Create React App
   Root Directory: frontend
   Build Command: npm run build (자동 감지됨)
   Output Directory: build (자동 감지됨)
   Install Command: npm install (자동 감지됨)
   ```

5. **환경 변수 설정 (중요!)**:
   ```
   Name: REACT_APP_API_URL
   Value: <1-D에서 복사한 Web Backend URL>/api
   예: https://backend-web-production-xxxx.up.railway.app/api
   ```

6. "Deploy" 클릭
7. **배포 URL 복사** (예: `https://soulplate.vercel.app`)

## 3단계: CORS 설정 업데이트

배포가 완료되면, 백엔드의 CORS 설정을 실제 도메인으로 업데이트해야 합니다.

### backend_web/main.py 수정:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # 로컬 개발
        "https://soulplate.vercel.app",  # 실제 Vercel URL로 변경
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### backend_model/main.py 수정:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  # 로컬 Web Backend
        "https://backend-web-production-xxxx.up.railway.app",  # 실제 Railway URL로 변경
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

변경 후:
```bash
git add backend_web/main.py backend_model/main.py
git commit -m "Update CORS with production URLs"
git push origin master
```

Railway와 Vercel이 자동으로 재배포합니다 (2-3분 소요).

## 4단계: 데이터베이스 마이그레이션

로컬에서 Railway PostgreSQL로 데이터를 마이그레이션합니다:

```bash
# 환경 변수 설정 (Railway에서 복사한 DATABASE_URL 사용)
export DATABASE_URL="postgresql://user:password@host:5432/dbname"

# 마이그레이션 실행
python scripts/migrate_filtered_data.py
```

또는 Railway Web Backend가 시작되면 SQLAlchemy가 자동으로 테이블을 생성합니다.

## 5단계: 테스트

1. Vercel URL로 접속
2. 회원가입 테스트
3. 비즈니스 목록 조회
4. AI 추천 정렬 테스트 (DeepFM, Multi-Tower)
5. 리뷰 작성 테스트
6. 취향 테스트 완료

## 6단계: QR 코드 생성

배포 완료 후, QR 코드를 생성하려면:

```bash
python scripts/generate_qr.py https://soulplate.vercel.app
```

QR 코드가 `restaurant_qr_code.png`로 저장됩니다.

## 배포 URL 요약

배포 완료 후 여기에 실제 URL을 기록하세요:

- **프론트엔드**: https://soulplate.vercel.app (예시)
- **Web Backend**: https://backend-web-production-xxxx.up.railway.app (예시)
- **Model Backend**: https://backend-model-production-xxxx.up.railway.app (예시)
- **Database**: Railway PostgreSQL

## 자동 배포

이제 `git push`만 하면 자동으로 배포됩니다:
- Vercel: 프론트엔드 자동 재배포 (2-3분)
- Railway: 백엔드 자동 재배포 (2-3분)

## 주의사항

1. **모델 파일**: `models/absa/` 디렉토리의 파일들은 Git에 포함되지 않습니다. Railway 배포 후 Hugging Face Hub에서 다운로드하도록 설정하거나, Persistent Storage를 사용하세요.

2. **환경 변수**: `.env` 파일은 Git에 포함되지 않으므로, 각 플랫폼에서 직접 설정해야 합니다.

3. **CORS**: 배포 후 실제 URL로 CORS 설정을 업데이트하는 것을 잊지 마세요.

4. **무료 플랜 제한**:
   - Railway: 월 $5 크레딧 (약 500시간 사용 가능)
   - Vercel: 월 100GB 대역폭
   - 충분히 월 10,000+ 방문자 처리 가능

## 문제 해결

### 1. 백엔드 502 에러
- Railway 대시보드 → 서비스 → "Deployments" 탭에서 로그 확인
- 환경 변수가 올바르게 설정되었는지 확인
- DATABASE_URL 형식 확인 (`postgresql://` vs `postgres://`)
- "Settings" → "Networking"에서 Public Networking이 활성화되었는지 확인

### 2. 프론트엔드 API 연결 실패
- Vercel 환경 변수 확인 (`REACT_APP_API_URL`)
- 백엔드 CORS 설정 확인
- 브라우저 콘솔에서 에러 확인
- Railway URL이 올바른지 확인

### 3. 모델 로딩 실패
- Railway Model Backend 로그 확인
- 모델 파일 경로 확인
- 메모리 부족 시 Railway 플랜 업그레이드 고려

### 4. Railway 크레딧 소진
- Railway 대시보드에서 사용량 확인
- 불필요한 서비스 비활성화
- 또는 유료 플랜으로 업그레이드 ($5/월부터)

