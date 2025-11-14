# 배포 체크리스트 ✅

단계별로 체크하면서 배포를 진행하세요.

## 사전 준비 ✅

- [x] GitHub 저장소에 코드 푸시 완료
- [x] 대용량 모델 파일 제거 완료
- [x] 프론트엔드 API URL 환경 변수 적용 완료
- [x] 배포 가이드 문서 작성 완료

## 1단계: Railway에 백엔드 배포

Railway 한 곳에서 백엔드 2개 + PostgreSQL을 모두 관리합니다!

### A. Railway 프로젝트 생성

- [ ] Railway 계정 생성/로그인 (https://railway.app)
- [ ] GitHub 계정으로 로그인
- [ ] "New Project" 클릭
- [ ] "Deploy from GitHub repo" 선택
- [ ] 저장소 선택: `dan0205/soulplate`

### B. PostgreSQL 데이터베이스 추가

- [ ] 프로젝트 대시보드에서 "+ New" 클릭
- [ ] "Database" → "PostgreSQL" 선택
- [ ] 자동 생성 대기 (1-2분)
- [ ] PostgreSQL 서비스 클릭
- [ ] "Variables" 탭 선택
- [ ] DATABASE_URL 복사

**DATABASE_URL**: _______________________________________________

### C. Model Backend 서비스 추가

- [ ] 프로젝트 대시보드에서 "+ New" 클릭
- [ ] "GitHub Repo" 선택 → 같은 저장소
- [ ] "Settings" → "Service Settings" 이동
- [ ] **Root Directory** 설정: `backend_model`
- [ ] **Start Command** 설정: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] "Deploy" 클릭
- [ ] 배포 완료 대기 (2-3분)
- [ ] "Settings" → "Networking" 이동
- [ ] "Public Networking" 활성화
- [ ] 배포 URL 복사

**Model Backend URL**: _______________________________________________

### D. Web Backend 서비스 추가

- [ ] 프로젝트 대시보드에서 "+ New" 클릭
- [ ] "GitHub Repo" 선택 → 같은 저장소
- [ ] "Settings" → "Service Settings" 이동
- [ ] **Root Directory** 설정: `backend_web`
- [ ] **Start Command** 설정: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] "Variables" 탭에서 환경 변수 설정:
  - [ ] `DATABASE_URL`: (B에서 복사한 값)
  - [ ] `SECRET_KEY`: (아래 명령어로 생성)
  - [ ] `ALGORITHM`: `HS256`
  - [ ] `ACCESS_TOKEN_EXPIRE_MINUTES`: `30`
  - [ ] `MODEL_API_URL`: (C에서 복사한 Model Backend URL)
- [ ] "Deploy" 클릭
- [ ] 배포 완료 대기 (2-3분)
- [ ] "Settings" → "Networking" 이동
- [ ] "Public Networking" 활성화
- [ ] 배포 URL 복사

**Web Backend URL**: _______________________________________________

**SECRET_KEY 생성**:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### E. Railway 배포 확인

- [ ] Railway 대시보드에서 3개 서비스 모두 "Active" 상태 확인
  - [ ] PostgreSQL: Active
  - [ ] backend_model: Deployed  
  - [ ] backend_web: Deployed
- [ ] 각 서비스 로그에서 에러 없는지 확인

## 2단계: Vercel에 프론트엔드 배포

- [ ] Vercel 계정 생성/로그인 (https://vercel.com)
- [ ] GitHub 계정으로 로그인
- [ ] "Add New" → "Project" 클릭
- [ ] GitHub에서 `dan0205/soulplate` 저장소 Import
- [ ] Framework: Create React App (자동 감지)
- [ ] **Root Directory** 설정: `frontend`
- [ ] Build/Output directory 확인 (자동 설정됨)
- [ ] 환경 변수 추가:
  - [ ] Name: `REACT_APP_API_URL`
  - [ ] Value: `<Web Backend URL>/api` (예: `https://backend-web-production-xxxx.up.railway.app/api`)
- [ ] "Deploy" 클릭
- [ ] 배포 완료 대기 (2-3분)
- [ ] 배포 URL 확인

**Frontend URL**: _______________________________________________

## 3단계: CORS 설정 업데이트

- [ ] CORS 업데이트 스크립트 실행:
```bash
python scripts/update_cors.py <Frontend URL> <Web Backend URL>
```

- [ ] 변경사항 확인
- [ ] Git 커밋 및 푸시:
```bash
git add backend_web/main.py backend_model/main.py
git commit -m "Update CORS with production URLs"
git push origin master
```

- [ ] Railway와 Vercel 자동 재배포 대기 (2-3분)
- [ ] Railway 대시보드에서 재배포 완료 확인

## 4단계: 데이터베이스 마이그레이션

옵션 1: 자동 (권장)
- [ ] Railway Web Backend가 시작되면 SQLAlchemy가 자동으로 테이블 생성
- [ ] Railway Web Backend 로그에서 테이블 생성 확인

옵션 2: 수동 마이그레이션
- [ ] 로컬에서 환경 변수 설정:
```bash
export DATABASE_URL="<Railway PostgreSQL URL>"
```

- [ ] 마이그레이션 실행:
```bash
python scripts/migrate_filtered_data.py
```

## 5단계: 배포 상태 확인

- [ ] 배포 상태 체크 스크립트 실행:
```bash
export FRONTEND_URL="<Frontend URL>"
export WEB_BACKEND_URL="<Web Backend URL>"
export MODEL_BACKEND_URL="<Model Backend URL>"
python scripts/check_deployment.py
```

- [ ] 모든 서비스 정상 작동 확인

## 6단계: 기능 테스트

- [ ] 프론트엔드 URL로 접속
- [ ] 페이지 로딩 확인
- [ ] 회원가입 테스트
  - [ ] 회원가입 성공
  - [ ] 자동 로그인 확인
- [ ] 비즈니스 목록 조회
  - [ ] 목록 표시 확인
  - [ ] 페이지네이션 동작 확인
- [ ] AI 추천 정렬
  - [ ] DeepFM 정렬 테스트
  - [ ] Multi-Tower 정렬 테스트
- [ ] 비즈니스 상세 페이지
  - [ ] 상세 정보 표시 확인
  - [ ] AI 예측 별점 표시 확인
- [ ] 리뷰 작성
  - [ ] 리뷰 작성 성공
  - [ ] 백그라운드 ABSA 분석 확인
- [ ] 취향 테스트
  - [ ] Quick Test (8문항) 완료
  - [ ] MBTI 타입 표시 확인
  - [ ] 추천 업데이트 확인

## 7단계: QR 코드 생성

- [ ] QR 코드 생성:
```bash
python scripts/generate_qr.py <Frontend URL>
```

- [ ] `restaurant_qr_code.png` 파일 확인
- [ ] 스마트폰으로 QR 코드 스캔 테스트

## 8단계: 성능 및 보안 체크

### 성능
- [ ] 페이지 로딩 속도 확인 (< 3초)
- [ ] API 응답 시간 확인 (< 1초)
- [ ] 이미지 로딩 확인

### 보안
- [ ] HTTPS 적용 확인 (자동)
- [ ] CORS 설정 확인 (특정 도메인만 허용)
- [ ] SECRET_KEY 안전하게 설정됨
- [ ] DATABASE_URL 노출되지 않음
- [ ] .env 파일이 Git에 포함되지 않음

### 브라우저 호환성
- [ ] Chrome 테스트
- [ ] Firefox 테스트
- [ ] Safari 테스트 (가능한 경우)
- [ ] 모바일 브라우저 테스트

## 9단계: 모니터링 설정

- [ ] Railway 대시보드에서 리소스 사용량 확인
- [ ] Railway 크레딧 잔액 확인
- [ ] Vercel 대시보드에서 트래픽 확인
- [ ] 에러 로그 확인

## 10단계: 문서화

- [ ] 실제 배포 URL을 README.md에 추가
- [ ] 배포 일시 기록
- [ ] 알려진 이슈 문서화 (있는 경우)

## 배포 완료! 🎉

축하합니다! 앱이 성공적으로 배포되었습니다.

### 배포 정보 요약

| 항목 | URL |
|------|-----|
| 프론트엔드 | _______________ |
| Web Backend | _______________ |
| Model Backend | _______________ |
| Database | Railway PostgreSQL |
| GitHub | https://github.com/dan0205/soulplate |

### 다음 단계

1. **팀원들과 공유**: QR 코드나 URL 공유
2. **피드백 수집**: 사용자 경험 개선
3. **기능 추가**: 새로운 기능 개발
4. **성능 모니터링**: 정기적으로 성능 확인

### 자동 배포

이제 코드를 수정하고 `git push`하면 자동으로 재배포됩니다:
- Vercel: 2-3분
- Railway: 2-3분

### Railway 크레딧 관리

- Railway 대시보드에서 사용량 모니터링
- 무료 $5 크레딧 소진 시:
  - 불필요한 서비스 비활성화
  - 또는 유료 플랜 업그레이드 ($5/월부터)

### 지원

문제가 발생하면 `DEPLOYMENT_INSTRUCTIONS.md`의 "문제 해결" 섹션을 참고하세요.
