# ABSA 재분석 가이드

## 개요

admin 계정의 16개 리뷰에 대해 ABSA 분석을 재실행하는 가이드입니다.

## 문제 상황

- admin 계정으로 작성한 리뷰 16개가 ABSA 분석을 거치지 않음
- Railway 백엔드에서 백그라운드 작업이 실패했거나 MODEL API 연결 문제

## 준비 사항

### 필수 환경변수

```bash
export RAILWAY_DATABASE_URL="postgresql://..."
export MODEL_API_URL="https://backendmodel-production-xxxx.up.railway.app"
```

**MODEL_API_URL 찾는 방법:**
1. Railway 대시보드 접속
2. `backend_model` 서비스 선택
3. Settings → Public Networking
4. URL 복사

## 실행 단계

### 1단계: MODEL API 상태 확인 ✅

먼저 MODEL API가 정상 작동하는지 확인합니다:

```bash
python scripts/check_model_api.py
```

**예상 출력:**
```
🔍 MODEL API Health Check
🤖 MODEL API: https://backendmodel-production-xxxx.up.railway.app

1️⃣ Root 엔드포인트 확인 (GET /)
   상태 코드: 200
   ✅ 응답: {'message': 'DeepFM & Multi-Tower Rating Prediction API', ...}

2️⃣ Health Check (GET /health)
   상태 코드: 200
   상태: healthy
   DeepFM 로딩: ✅
   MultiTower 로딩: ✅
   ABSA 로딩: ✅
   ✅ 모든 모델이 정상 로딩됨

3️⃣ ABSA 분석 테스트 (POST /analyze_review)
   테스트 리뷰: 음식이 정말 맛있고 서비스도 친절했습니다. 분위기도 좋았어요!
   상태 코드: 200
   ✅ ABSA 분석 성공!
   ABSA 특성 수: 51개
   텍스트 임베딩 차원: 100차원

✅ MODEL API Health Check 성공!
```

### ❌ 만약 실패한다면?

**에러: "연결 실패" 또는 "타임아웃"**
- Railway에서 backend_model 서비스가 실행 중인지 확인
- MODEL_API_URL이 올바른지 확인
- Railway 로그에서 에러 확인

**에러: "ABSA 로딩: ❌"**
- Railway에서 backend_model 서비스 재시작
- 로그에서 모델 로딩 에러 확인

### 2단계: admin 리뷰 ABSA 재분석 🔄

MODEL API가 정상이면 admin 리뷰를 재분석합니다:

```bash
python scripts/reanalyze_reviews_absa.py --username admin
```

**예상 출력:**
```
🚀 리뷰 ABSA 재분석 스크립트
🗄️  DB: interchange.proxy.rlwy.net:52092
🤖 MODEL API: https://backendmodel-production-xxxx.up.railway.app

🔄 사용자 'admin'의 ABSA 없는 리뷰 재분석 시작...
사용자: admin (ID: 24290)
ABSA가 없는 리뷰: 16개

[1/16] admin → 우만동족발집 아주대점
  리뷰: 우만동족발 즐겨먹은지는 거의 10년은 되가는거 같아요...
  HTTP 상태: 200
  ✅ ABSA 분석 완료 (51개 특성)

[2/16] admin → 우만동족발집 아주대점
  리뷰: 비오는날에도 역시 우만동족발!!! 반반은 진리입니다...
  HTTP 상태: 200
  ✅ ABSA 분석 완료 (51개 특성)

...

🔄 1개 비즈니스의 ABSA 재계산 중...
  ✅ 우만동족발집 아주대점 업데이트 완료

✅ 완료: 16개 성공, 0개 실패
```

### 3단계: 결과 확인 ✅

재분석이 완료되면 DB 상태를 다시 확인합니다:

```bash
python scripts/check_cloud_db.py
```

**확인 사항:**
- 우만동족발집의 ABSA 특성 값이 0.00이 아닌 실제 값으로 표시되는지
- 리뷰의 ABSA가 있는지 (ABSA가 있는 리뷰: 21/21개)

## Windows 사용자

### PowerShell
```powershell
$env:RAILWAY_DATABASE_URL="postgresql://..."
$env:MODEL_API_URL="https://backendmodel-production-xxxx.up.railway.app"
python scripts/check_model_api.py
python scripts/reanalyze_reviews_absa.py --username admin
```

### CMD
```cmd
set RAILWAY_DATABASE_URL=postgresql://...
set MODEL_API_URL=https://backendmodel-production-xxxx.up.railway.app
python scripts/check_model_api.py
python scripts/reanalyze_reviews_absa.py --username admin
```

## 트러블슈팅

### 문제: 모든 리뷰 분석 실패

**증상:**
```
ERROR:__main__:  연결 실패: ...
ERROR:__main__:  ❌ ABSA 분석 실패
```

**해결:**
1. MODEL API URL 확인:
   ```bash
   echo $MODEL_API_URL
   ```
2. Railway 대시보드에서 backend_model 서비스 상태 확인
3. Health check 실행:
   ```bash
   curl https://backendmodel-production-xxxx.up.railway.app/health
   ```

### 문제: 일부 리뷰만 분석 실패

**증상:**
```
✅ 완료: 12개 성공, 4개 실패
```

**해결:**
- 실패한 리뷰의 에러 메시지 확인
- 리뷰 텍스트가 너무 길거나 특수문자가 많은 경우 발생 가능
- 스크립트를 다시 실행 (이미 성공한 리뷰는 스킵됨)

### 문제: 타임아웃

**증상:**
```
ERROR:__main__:  타임아웃: 60초 이내 응답 없음
```

**해결:**
- Railway의 backend_model 서비스가 느리거나 리소스 부족
- Railway 대시보드에서 로그 확인
- 필요시 서비스 재시작

## 다른 옵션

### 수원시 음식점의 모든 리뷰 재분석
```bash
python scripts/reanalyze_reviews_absa.py --suwon
```

### 전체 DB의 모든 ABSA 없는 리뷰 재분석
```bash
python scripts/reanalyze_reviews_absa.py --all
```

## 주의사항

1. **MODEL API가 필수**: Railway의 backend_model 서비스가 실행 중이어야 함
2. **시간 소요**: 리뷰 1개당 약 1-5초 소요 (16개 = 약 1-2분)
3. **중복 실행 안전**: 이미 ABSA가 있는 리뷰는 자동으로 스킵됨

## 성공 후 확인 사항

✅ admin 계정으로 로그인하여 프론트엔드에서 확인:
- 우만동족발집 상세 페이지의 음식/서비스/분위기 점수가 표시되는지
- AI 예측 점수가 정상적으로 계산되는지

✅ 다른 사용자 계정으로도 확인:
- 수원시 음식점들이 제대로 추천되는지
- ABSA 특성이 올바르게 표시되는지

## 문의

문제가 계속되면 다음을 확인하세요:
- Railway 대시보드의 backend_model 로그
- Railway 대시보드의 backend_web 로그
- 환경변수 설정이 올바른지

