# Business ABSA 업데이트 가이드

## 개요

이 문서는 수원시 음식점의 ABSA features를 재집계하는 방법을 안내합니다.

## 문제 상황

기존에는 리뷰 작성 시 `Business.absa_features`가 자동으로 업데이트되지 않았습니다. 이로 인해:
- 수원시 음식점에 리뷰를 작성해도 음식점의 ABSA가 업데이트되지 않음
- 다른 사용자가 해당 음식점을 볼 때 음식/서비스/분위기 점수가 0으로 표시됨
- AI 예측이 부정확해짐

## 해결 방법

### 1. 코드 수정 완료 ✅

다음 수정사항이 이미 적용되었습니다:

**`backend_web/main.py`**:
- `update_business_profile` 함수 추가: 비즈니스의 모든 리뷰로부터 ABSA 평균 계산
- `process_review_features` 함수 수정: 리뷰 작성 시 Business 프로필도 자동 업데이트

**결과**: 이제부터 새로 작성되는 리뷰는 자동으로 Business ABSA를 업데이트합니다.

### 2. 기존 데이터 재집계 필요 🔄

기존에 작성된 리뷰들에 대해서는 수동으로 재집계가 필요합니다.

## 스크립트 실행 방법

### A. DB 상태 확인 (선택 사항)

먼저 현재 DB 상태를 확인하려면:

```bash
# 환경변수 설정
export RAILWAY_DATABASE_URL="postgresql://..."

# DB 확인 스크립트 실행
python scripts/check_cloud_db.py
```

**출력 예시**:
```
📍 수원시 음식점 조회 (위도 37.2-37.3, 경도 126.9-127.1)
✅ 수원시 음식점 총 10개 발견

[1] 우만동족발집 아주대점
    ABSA features: ❌ 없음
    실제 리뷰: 3개
      - ABSA가 있는 리뷰: 3개
      - ABSA가 없는 리뷰: 0개
```

### B. ABSA 재집계 실행 (필수)

#### 옵션 1: 수원시 음식점만 업데이트 (권장)

```bash
# 환경변수 설정
export RAILWAY_DATABASE_URL="postgresql://..."

# 수원시 음식점만 재집계
python scripts/update_business_absa.py --suwon
```

#### 옵션 2: 모든 음식점 업데이트

```bash
python scripts/update_business_absa.py --all
```

#### 옵션 3: 특정 음식점만 업데이트

```bash
python scripts/update_business_absa.py --business-id "BUSINESS_ID_HERE"
```

### C. 업데이트 확인

```bash
# 다시 DB 확인
python scripts/check_cloud_db.py
```

**출력 예시**:
```
[1] 우만동족발집 아주대점
    ABSA features: ✅ 있음
      - 음식 긍정: 0.82
      - 서비스 긍정: 0.65
      - 분위기 긍정: 0.71
```

## Windows 환경변수 설정 (Windows 사용자)

PowerShell:
```powershell
$env:RAILWAY_DATABASE_URL="postgresql://..."
python scripts/update_business_absa.py --suwon
```

CMD:
```cmd
set RAILWAY_DATABASE_URL=postgresql://...
python scripts/update_business_absa.py --suwon
```

## 주의사항

1. **환경변수 필수**: `RAILWAY_DATABASE_URL` 환경변수가 설정되어 있어야 합니다.
2. **백업 권장**: 중요한 작업이므로 가능하면 DB 백업 후 실행을 권장합니다.
3. **프로덕션 적용**: Railway에 배포 시 새로운 코드가 자동으로 적용됩니다.

## 스크립트 상세 설명

### `check_cloud_db.py`

현재 DB 상태를 진단합니다:
- 수원시 음식점 목록
- 각 음식점의 리뷰 수와 ABSA 존재 여부
- 리뷰의 ABSA 존재 여부
- AI 예측 캐시 상태

### `update_business_absa.py`

Business ABSA를 재집계합니다:
- 각 음식점의 모든 리뷰로부터 ABSA 평균 계산
- `Business.absa_features` 업데이트
- 별점 평균 및 리뷰 수도 함께 업데이트

## 자주 묻는 질문

### Q: 스크립트를 실행하면 기존 데이터가 손상되나요?
A: 아니요. 스크립트는 리뷰 데이터를 읽기만 하고 Business 테이블의 집계 데이터만 업데이트합니다.

### Q: 스크립트 실행 시간은 얼마나 걸리나요?
A: 수원시 음식점(10개 내외)만 업데이트하는 경우 몇 초 이내에 완료됩니다.

### Q: 앞으로는 수동 실행이 필요한가요?
A: 아니요. 이번 코드 수정으로 앞으로는 리뷰 작성 시 자동으로 Business ABSA가 업데이트됩니다.

## 완료 후 확인 사항

✅ 수원시 음식점의 ABSA features가 설정되었는지 확인
✅ 프론트엔드에서 음식점 상세 페이지의 음식/서비스/분위기 점수가 제대로 표시되는지 확인
✅ AI 예측 점수가 개선되었는지 확인

## 문의

문제가 발생하면 로그를 확인하거나 개발팀에 문의하세요.

