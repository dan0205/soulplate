# 테스트 참고사항

## ⚠️ 중요 안내

현재 수정한 코드는 **로컬 파일**이므로, **Railway 클라우드 API에는 아직 적용되지 않았습니다**.

## 테스트 시나리오

### 1. 현재 상태에서 테스트 (클라우드 API)
- **예상 결과**: 여전히 DeepFM이 1점대 예측 (수정 사항 미적용)
- **목적**: 수정 전 상태 재확인

### 2. Railway 재배포 후 테스트
- **예상 결과**: DeepFM이 3.5~4.5점으로 정상 예측
- **목적**: 수정 사항 검증

## 테스트 실행 방법

```bash
# 환경변수 설정
export RAILWAY_DATABASE_URL="postgresql://postgres:CIirVNgnCPOhbljkayXakvLFttNSodnu@interchange.proxy.rlwy.net:52092/railway"

# 테스트 실행
python scripts/test_new_model_api.py
```

## Railway 재배포 필요

```bash
# Git을 통한 재배포
git add backend_model/prediction_service.py DEEPFM_FIX_REPORT.md
git commit -m "fix: DeepFM input scaling issue"
git push origin main
```

재배포 후 다시 테스트를 실행하면 수정 사항이 적용된 결과를 확인할 수 있습니다.

