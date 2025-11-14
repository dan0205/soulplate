# 🎉 배포 준비 완료! (Railway + Vercel)

클라우드 배포를 위한 모든 코드 준비가 완료되었습니다.

## ✅ 완료된 작업

### 1. 코드 수정 및 커밋
- ✅ 프론트엔드 API URL이 환경 변수를 사용하도록 수정
  - `frontend/src/services/api.js`: `process.env.REACT_APP_API_URL` 사용
- ✅ Git history에서 대용량 모델 파일 제거 (451MB)
  - `models/absa/` 디렉토리가 `.gitignore`에 추가됨
- ✅ 모든 변경사항이 GitHub에 푸시됨
  - 저장소: `https://github.com/dan0205/soulplate`

### 2. 배포 문서 작성
- ✅ **DEPLOYMENT_INSTRUCTIONS.md**: 단계별 배포 가이드 (6.6KB)
- ✅ **DEPLOYMENT_CHECKLIST.md**: 체크하면서 배포하기 (6.4KB)
- ✅ **README.md**: 배포 정보 추가

### 3. 배포 자동화 스크립트
- ✅ **scripts/generate_qr.py**: QR 코드 생성 스크립트 (2.3KB)
- ✅ **scripts/update_cors.py**: CORS 설정 자동 업데이트 (3.2KB)
- ✅ **scripts/check_deployment.py**: 배포 상태 확인 (4.1KB)

## 📋 다음 단계 (수동 작업 필요)

배포는 웹 UI에서 수동으로 진행해야 합니다. 다음 순서대로 진행하세요:

### 1단계: Railway 배포 (백엔드 + PostgreSQL)

Railway 한 곳에서 모든 백엔드를 관리합니다!

#### A. Railway 프로젝트 생성
1. https://railway.app 접속 (GitHub 로그인)
2. "New Project" 클릭
3. "Deploy from GitHub repo" 선택
4. 저장소: `dan0205/soulplate`

#### B. PostgreSQL 추가
1. "+ New" → "Database" → "PostgreSQL"
2. 자동 생성 (1-2분)
3. DATABASE_URL 복사

#### C. Model Backend 서비스 추가
1. "+ New" → "GitHub Repo" → 같은 저장소
2. Root Directory: `backend_model`
3. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy → Public Networking 활성화
5. 배포 URL 복사

#### D. Web Backend 서비스 추가
1. "+ New" → "GitHub Repo" → 같은 저장소
2. Root Directory: `backend_web`
3. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. 환경 변수 설정:
   - `DATABASE_URL`: (B에서 복사)
   - `SECRET_KEY`: 랜덤 생성
   - `MODEL_API_URL`: (C에서 복사)
   - `ALGORITHM`: HS256
   - `ACCESS_TOKEN_EXPIRE_MINUTES`: 30
5. Deploy → Public Networking 활성화
6. 배포 URL 복사

### 2단계: Vercel 배포 (프론트엔드)

1. https://vercel.com 접속 (GitHub 로그인)
2. "Add New" → "Project"
3. Import `dan0205/soulplate`
4. Root directory: `frontend`
5. 환경 변수:
   - `REACT_APP_API_URL`: `<Railway Web Backend URL>/api`
6. Deploy 클릭

### 3단계: CORS 업데이트

배포 완료 후:

```bash
# CORS 설정 업데이트
python scripts/update_cors.py \
    https://your-app.vercel.app \
    https://backend-web-production-xxxx.up.railway.app

# 커밋 및 푸시
git add backend_web/main.py backend_model/main.py
git commit -m "Update CORS with production URLs"
git push origin master
```

Railway와 Vercel이 자동으로 재배포합니다 (2-3분).

### 4단계: 배포 확인

```bash
# 배포 상태 확인
export FRONTEND_URL="https://your-app.vercel.app"
export WEB_BACKEND_URL="https://your-backend.koyeb.app"
export MODEL_BACKEND_URL="https://your-model.koyeb.app"
python scripts/check_deployment.py
```

### 5단계: QR 코드 생성

```bash
# QR 코드 생성
python scripts/generate_qr.py https://your-app.vercel.app
```

## 📚 참고 문서

- **[DEPLOYMENT_INSTRUCTIONS.md](DEPLOYMENT_INSTRUCTIONS.md)**: 상세한 배포 가이드
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)**: 단계별 체크리스트
- **[README.md](README.md)**: 프로젝트 개요 및 배포 정보

## 🔧 배포 도구

| 스크립트 | 용도 | 사용 시점 |
|---------|------|----------|
| `scripts/generate_qr.py` | QR 코드 생성 | 배포 완료 후 |
| `scripts/update_cors.py` | CORS 설정 업데이트 | 배포 완료 후 |
| `scripts/check_deployment.py` | 배포 상태 확인 | 배포 후 테스트 시 |

## 💡 주요 포인트

1. **환경 변수**: 각 플랫폼(Koyeb, Vercel)에서 환경 변수를 직접 설정해야 합니다.
2. **모델 파일**: Git에 포함되지 않으므로, Hugging Face Hub나 Persistent Storage를 사용하세요.
3. **CORS**: 배포 후 실제 URL로 CORS 설정을 업데이트해야 합니다.
4. **자동 배포**: 이후 `git push`만 하면 자동으로 재배포됩니다.

## 🚀 예상 결과

배포 완료 시:
- ✅ 전 세계 어디서든 접속 가능
- ✅ QR 코드로 즉시 접근
- ✅ Git push만으로 자동 업데이트
- ✅ 월 10,000+ 방문자 처리 가능
- ✅ 완전 무료 (Railway $5 크레딧 + Vercel 무료)
- ✅ HTTPS 자동 적용
- ✅ 한 대시보드에서 모든 백엔드 관리

## ⏱️ 예상 소요 시간

- Railway 설정 및 배포: 15분 (한 곳에서 모두 관리!)
- Vercel 설정 및 배포: 10분
- CORS 업데이트 및 테스트: 5분
- **총 30분** (Koyeb보다 15분 빠름!)

## 🆘 문제 해결

문제가 발생하면:
1. **DEPLOYMENT_INSTRUCTIONS.md**의 "문제 해결" 섹션 참고
2. Koyeb/Vercel 로그 확인
3. 환경 변수 설정 재확인
4. CORS 설정 확인

---

**준비 완료!** DEPLOYMENT_CHECKLIST.md를 열고 단계별로 체크하면서 배포를 시작하세요! 🚀

