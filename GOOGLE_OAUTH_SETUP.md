# Google OAuth 설정 가이드

## 🔐 1단계: Google Cloud Console 설정

### 1.1 프로젝트 생성/선택
1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. 기존 프로젝트 선택 또는 새 프로젝트 생성

### 1.2 OAuth 동의 화면 구성
1. **좌측 메뉴**: `API 및 서비스` > `OAuth 동의 화면`
2. **User Type**: `외부` 선택
3. **앱 정보** 입력:
   - 앱 이름: `Soulplate`
   - 사용자 지원 이메일: `your-email@example.com`
   - 앱 로고: (선택사항)
   - 앱 도메인:
     - 홈페이지: `https://soulplate.vercel.app`
     - 개인정보처리방침: `https://soulplate.vercel.app/privacy` (있다면)
     - 서비스 약관: `https://soulplate.vercel.app/terms` (있다면)
   - 개발자 연락처: `your-email@example.com`
4. **범위** 추가:
   - `/.../auth/userinfo.email`
   - `/.../auth/userinfo.profile`
   - `openid`
5. **테스트 사용자** (선택사항): 테스트 계정 추가
6. **저장 후 계속**

### 1.3 OAuth 클라이언트 ID 생성
1. **좌측 메뉴**: `API 및 서비스` > `사용자 인증 정보`
2. **+ 사용자 인증 정보 만들기** > `OAuth 클라이언트 ID`
3. **애플리케이션 유형**: `웹 애플리케이션`
4. **이름**: `Soulplate Web Client`

5. **승인된 자바스크립트 원본** 추가:
   ```
   https://soulplate.vercel.app
   ```

6. **승인된 리디렉션 URI** 추가:
   ```
   https://backendweb-production-14de.up.railway.app/api/auth/google/callback
   ```

7. **만들기** 클릭

8. **클라이언트 ID**와 **클라이언트 보안 비밀** 복사 저장
   ```
   클라이언트 ID: 1234567890-abcdefghijk.apps.googleusercontent.com
   클라이언트 보안 비밀: GOCSPX-xxxxxxxxxxxxxxxxxxxxx
   ```

---

## 🚀 2단계: Railway 환경변수 설정

### Backend Web (backendweb-production-14de)
Railway 프로젝트 > Variables 탭에서 추가:

```bash
DATABASE_URL=postgresql://postgres:CIirVNgnCPOhbljkayXakvLFttNSodnu@interchange.proxy.rlwy.net:52092/railway

SECRET_KEY=your-secure-random-secret-key-min-32-characters-long
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

GOOGLE_CLIENT_ID=1234567890-abcdefghijk.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxxxxxxxxxx

FRONTEND_URL=https://soulplate.vercel.app
MODEL_API_URL=https://backendmodel-production-4594.up.railway.app

PREDICTION_CONCURRENCY=3
PREDICTION_CHUNK_SIZE=50
PREDICTION_TIMEOUT=360
```

**⚠️ SECRET_KEY 생성 방법:**
```python
# Python에서 실행
import secrets
print(secrets.token_urlsafe(32))
```

---

## 🎨 3단계: Vercel 환경변수 설정

### Frontend (soulplate.vercel.app)
Vercel 프로젝트 > Settings > Environment Variables에서 추가:

```bash
REACT_APP_API_URL=https://backendweb-production-14de.up.railway.app/api
REACT_APP_KAKAO_MAP_KEY=your_kakao_map_key
```

**Production, Preview, Development** 모두 체크

---

## 🧪 4단계: 프로덕션 테스트

1. Railway에 배포 (자동 배포 설정되어 있으면 자동)
2. Vercel에 배포 (자동 배포 설정되어 있으면 자동)
3. `https://soulplate.vercel.app`에서 구글 로그인 테스트
4. 신규 사용자 생성 확인
5. 로그인/로그아웃 테스트

---

## ✅ 체크리스트

### Google Cloud Console
- [ ] OAuth 동의 화면 구성 완료
- [ ] OAuth 클라이언트 ID 생성
- [ ] 승인된 자바스크립트 원본 추가:
  - [ ] `https://soulplate.vercel.app`
- [ ] 승인된 리디렉션 URI 추가:
  - [ ] `https://backendweb-production-14de.up.railway.app/api/auth/google/callback`
- [ ] 클라이언트 ID와 시크릿 복사

### Railway (Backend Web)
- [ ] `GOOGLE_CLIENT_ID` 설정
- [ ] `GOOGLE_CLIENT_SECRET` 설정
- [ ] `FRONTEND_URL=https://soulplate.vercel.app` 설정
- [ ] `MODEL_API_URL` 확인 (이미 있을 수 있음)
- [ ] `SECRET_KEY` 생성 및 설정
- [ ] 배포 확인

### Vercel (Frontend)
- [ ] `REACT_APP_API_URL=https://backendweb-production-14de.up.railway.app/api` 설정
- [ ] 배포 확인

### 기능 테스트
- [ ] 프로덕션 환경에서 구글 로그인 성공
- [ ] 새 계정 생성 확인
- [ ] 기존 계정 로그인 확인
- [ ] JWT 토큰 정상 작동 확인
- [ ] 로그아웃 정상 작동 확인

---

## 🔍 문제 해결

### "redirect_uri_mismatch" 에러
- Google Cloud Console에서 리디렉션 URI가 정확히 일치하는지 확인
- `http://` vs `https://` 확인
- 뒤에 슬래시(`/`) 있는지 확인

### CORS 에러
- Backend의 CORS 설정에 프론트엔드 URL 포함되어 있는지 확인
- Railway 환경변수 재배포 확인

### 환경변수 적용 안 됨
- Railway/Vercel에서 환경변수 저장 후 **재배포** 필요
- Railway: Settings > Deploy > Manual Deploy
- Vercel: Deployments > Redeploy

---

## 📞 지원

문제가 발생하면:
1. Railway 로그 확인: `View Logs`
2. Vercel 로그 확인: `Deployments > 해당 배포 > Functions`
3. 브라우저 개발자 도구 콘솔 확인

---

## 🎉 완료!

모든 설정이 완료되면 사용자들은 구글 계정으로 간편하게 로그인할 수 있습니다!

