# DeepFM 스케일링 수정 보고서

**작성일**: 2024
**수정자**: AI Assistant
**상태**: 완료

---

## 📋 요약

DeepFM 모델이 1점대 점수를 예측하는 문제를 해결하기 위해 `prediction_service.py`의 입력 피처 스케일링 로직을 수정했습니다.

**주요 변경사항**:
- 스케일러 로딩 로그 강화
- 스케일링 적용 전후 값 로깅 추가
- `scaler_params` 없을 때 명확한 에러 처리

---

## 🔍 문제 분석

### 발견된 문제

**증상**:
- DeepFM 모델이 모든 음식점에 대해 1.02~1.11점으로 예측
- Multi-Tower 모델은 3.5~4.5점으로 정상 예측
- 실제 별점 평균: 4.22점

**원인 분석**:

1. **입력 피처 분석 결과**:
   ```
   useful: 2,123 ~ 41,594 (원본 값, 스케일링 안 됨)
   compliment_log: 5.11 ~ 9.15 (로그 변환 적용됨)
   fans_log: 3.58 ~ 6.91 (로그 변환 적용됨)
   ```

2. **학습 시 데이터**:
   - `scaler_params.json`에 따르면 `useful` mean=3003.31, std=13183.66
   - 학습 시에는 이미 스케일링된 데이터 사용
   - 예상 스케일링 값: (2123 - 3003) / 13183 ≈ -0.07

3. **추론 시 문제**:
   - `scaler_params`가 로딩되지 않거나
   - `if self.scaler_params:` 조건이 False로 평가되어
   - 원본 값이 그대로 사용됨

4. **결과**:
   - 극단적으로 큰 `useful` 값(수만)이 입력됨
   - 학습 시 본 적 없는 분포로 인해 모델이 극단적으로 낮은 점수 출력
   - `sigmoid(매우 작은 음수) * 4 + 1 ≈ 1.0`

### 기타 발견사항

- **텍스트 임베딩 손실**: 100개 텍스트 임베딩이 모두 0 (별도 수정 필요)
- **Multi-Tower 정상**: 독립적인 타워 구조로 스케일링 불일치에 더 robust함

---

## 🔧 수정 내용

### 1. 스케일러 로딩 로그 강화

**파일**: `backend_model/prediction_service.py`
**메서드**: `_load_scaler_params()` (Line 43-82)

**변경 전**:
```python
def _load_scaler_params(self):
    scaler_path = 'models/scaler_params.json'
    hf_path = ensure_model_file("models/scaler_params.json", scaler_path)
    
    if hf_path and os.path.exists(hf_path):
        with open(hf_path, 'r') as f:
            self.scaler_params = json.load(f)
    elif os.path.exists(scaler_path):
        with open(scaler_path, 'r') as f:
            self.scaler_params = json.load(f)
    else:
        print(f"  [WARNING] Scaler params 파일 없음: {scaler_path}")
        self.scaler_params = None
```

**변경 후**:
```python
def _load_scaler_params(self):
    scaler_path = 'models/scaler_params.json'
    
    logger.info(f"[Scaler] 파라미터 로딩 시도: {scaler_path}")
    
    hf_path = ensure_model_file("models/scaler_params.json", scaler_path)
    
    loaded_path = None
    if hf_path and os.path.exists(hf_path):
        loaded_path = hf_path
        logger.info(f"[Scaler] HuggingFace에서 로딩: {hf_path}")
        with open(hf_path, 'r') as f:
            self.scaler_params = json.load(f)
    elif os.path.exists(scaler_path):
        loaded_path = scaler_path
        logger.info(f"[Scaler] 로컬에서 로딩: {scaler_path}")
        with open(scaler_path, 'r') as f:
            self.scaler_params = json.load(f)
    else:
        logger.error(f"[Scaler] ❌ 파일을 찾을 수 없음: {scaler_path}")
        self.scaler_params = None
        return
    
    # 로딩 성공 시 내용 확인
    if self.scaler_params:
        logger.info(f"[Scaler] ✅ 로딩 성공")
        logger.info(f"[Scaler] User params keys: {list(self.scaler_params.get('user', {}).keys())}")
        logger.info(f"[Scaler] Business params keys: {list(self.scaler_params.get('business', {}).keys())}")
        
        # useful 파라미터 확인 (핵심 피처)
        if 'user' in self.scaler_params and 'useful' in self.scaler_params['user']:
            useful_params = self.scaler_params['user']['useful']
            logger.info(f"[Scaler] useful mean={useful_params['mean']:.2f}, std={useful_params['std']:.2f}")
        else:
            logger.warning(f"[Scaler] ⚠️ useful 파라미터가 없음!")
```

**개선사항**:
- 파일 로딩 시도 로그 추가
- HuggingFace vs 로컬 로딩 구분
- 로딩 성공 시 파라미터 키 확인
- `useful` 파라미터 값 출력 (핵심 피처)

### 2. User 피처 스케일링 로직 수정

**파일**: `backend_model/prediction_service.py`
**메서드**: `prepare_combined_features()` (Line 244-267)

**변경 전**:
```python
# Standard Scaling
if self.scaler_params:
    user_params = self.scaler_params['user']
    useful_scaled = (useful - user_params['useful']['mean']) / user_params['useful']['std']
    compliment_scaled = (compliment_log - user_params['compliment']['mean']) / user_params['compliment']['std']
    fans_scaled = (fans_log - user_params['fans']['mean']) / user_params['fans']['std']
    average_stars_scaled = (average_stars - user_params['average_stars']['mean']) / user_params['average_stars']['std']
    yelping_since_days_scaled = (yelping_since_days - user_params['yelping_since_days']['mean']) / user_params['yelping_since_days']['std']
else:
    # Scaler 없으면 원본 값 사용 (비권장)
    useful_scaled = useful
    compliment_scaled = compliment_log
    fans_scaled = fans_log
    average_stars_scaled = average_stars
    yelping_since_days_scaled = yelping_since_days
```

**변경 후**:
```python
# Standard Scaling
if self.scaler_params:
    user_params = self.scaler_params['user']
    
    # 스케일링 전 로그 (디버깅용)
    logger.debug(f"[Scaling] User 원본 값 - useful: {useful:.2f}, compliment_log: {compliment_log:.2f}, fans_log: {fans_log:.2f}")
    
    useful_scaled = (useful - user_params['useful']['mean']) / user_params['useful']['std']
    compliment_scaled = (compliment_log - user_params['compliment']['mean']) / user_params['compliment']['std']
    fans_scaled = (fans_log - user_params['fans']['mean']) / user_params['fans']['std']
    average_stars_scaled = (average_stars - user_params['average_stars']['mean']) / user_params['average_stars']['std']
    yelping_since_days_scaled = (yelping_since_days - user_params['yelping_since_days']['mean']) / user_params['yelping_since_days']['std']
    
    # 스케일링 후 로그 (디버깅용)
    logger.debug(f"[Scaling] User 스케일링 후 - useful: {useful_scaled:.2f}, compliment: {compliment_scaled:.2f}, fans: {fans_scaled:.2f}")
else:
    # Scaler 없으면 에러 발생
    logger.error(f"[Scaling] ❌ scaler_params가 None입니다! 스케일링 불가능")
    logger.error(f"[Scaling] 원본 값: useful={useful}, compliment={compliment}, fans={fans}")
    raise ValueError("scaler_params is required for prediction. Please ensure scaler_params.json is loaded correctly.")
```

**개선사항**:
- 스케일링 전후 값 로깅 추가
- `scaler_params`가 None일 때 ValueError 발생 (원본 값 사용 방지)
- 명확한 에러 메시지

### 3. Business 피처 스케일링 로직 수정

**파일**: `backend_model/prediction_service.py`
**메서드**: `prepare_combined_features()` (Line 269-291)

**변경 전**:
```python
# Standard Scaling
if self.scaler_params:
    business_params = self.scaler_params['business']
    stars_scaled = (stars - business_params['stars']['mean']) / business_params['stars']['std']
    latitude_scaled = (latitude - business_params['latitude']['mean']) / business_params['latitude']['std']
    longitude_scaled = (longitude - business_params['longitude']['mean']) / business_params['longitude']['std']
else:
    # Scaler 없으면 원본 값 사용 (비권장)
    stars_scaled = stars
    latitude_scaled = latitude
    longitude_scaled = longitude
```

**변경 후**:
```python
# Standard Scaling
if self.scaler_params:
    business_params = self.scaler_params['business']
    
    # 스케일링 전 로그 (디버깅용)
    logger.debug(f"[Scaling] Business 원본 값 - stars: {stars:.2f}, lat: {latitude:.4f}, lng: {longitude:.4f}")
    
    stars_scaled = (stars - business_params['stars']['mean']) / business_params['stars']['std']
    latitude_scaled = (latitude - business_params['latitude']['mean']) / business_params['latitude']['std']
    longitude_scaled = (longitude - business_params['longitude']['mean']) / business_params['longitude']['std']
    
    # 스케일링 후 로그 (디버깅용)
    logger.debug(f"[Scaling] Business 스케일링 후 - stars: {stars_scaled:.2f}, lat: {latitude_scaled:.2f}, lng: {longitude_scaled:.2f}")
else:
    # Scaler 없으면 에러 발생
    logger.error(f"[Scaling] ❌ scaler_params가 None입니다! 스케일링 불가능")
    logger.error(f"[Scaling] 원본 값: stars={stars}, latitude={latitude}, longitude={longitude}")
    raise ValueError("scaler_params is required for prediction. Please ensure scaler_params.json is loaded correctly.")
```

**개선사항**:
- User 피처와 동일한 로직 적용
- 스케일링 전후 값 로깅
- `scaler_params` 없을 때 ValueError 발생

---

## 📊 예상 결과

### 수정 전
```
useful: 2,123 (원본 값)
DeepFM 예측: 1.04점
```

### 수정 후
```
useful: 2,123 → 스케일링 → -0.07
DeepFM 예측: 3.5~4.5점 (정상 범위)
```

### 검증 방법

1. **로그 확인**:
   - Railway 로그에서 `[Scaler]` 태그로 로딩 확인
   - `[Scaling]` 태그로 스케일링 적용 확인

2. **예측 테스트**:
   ```bash
   cd /c/Users/yidj0205/Desktop/code/demo
   export RAILWAY_DATABASE_URL="postgresql://..."
   python scripts/test_new_model_api.py
   ```

3. **예상 로그**:
   ```
   [Scaler] ✅ 로딩 성공
   [Scaler] useful mean=3003.31, std=13183.66
   [Scaling] User 원본 값 - useful: 2123.00, ...
   [Scaling] User 스케일링 후 - useful: -0.07, ...
   ```

---

## 🚀 Railway 재배포 방법

이 수정사항은 **로컬 코드 변경**이므로 Railway에 재배포해야 클라우드 API에 적용됩니다.

### Git을 통한 배포 (권장)

```bash
# 변경사항 확인
git status

# 수정된 파일 스테이징
git add backend_model/prediction_service.py
git add DEEPFM_FIX_REPORT.md

# 커밋
git commit -m "fix: DeepFM input scaling issue - add logging and error handling"

# 푸시 (자동 배포)
git push origin main
```

### Railway CLI를 통한 배포

```bash
# Railway CLI 설치 (필요시)
npm install -g @railway/cli

# 로그인
railway login

# 배포
railway up
```

### 배포 후 확인

1. Railway 대시보드에서 배포 로그 확인
2. 모델 API URL에서 헬스체크:
   ```bash
   curl https://backendmodel-production-4594.up.railway.app/
   ```
3. 예측 테스트 실행

---

## 🔄 되돌리기 방법

문제가 발생하면 백업 파일로 복원:

```bash
# 백업 파일로 복원
cp backend_model/prediction_service.py.backup backend_model/prediction_service.py

# Git 커밋 및 푸시
git add backend_model/prediction_service.py
git commit -m "revert: rollback DeepFM scaling fix"
git push origin main
```

또는 Git revert:

```bash
# 최근 커밋 되돌리기
git revert HEAD

# 푸시
git push origin main
```

---

## 📝 추가 작업 필요사항

### 1. 텍스트 임베딩 수정 (별도 작업)
- **문제**: 텍스트 임베딩 100개가 모두 0
- **원인**: `text_embedding_service` 로딩 실패
- **해결**: 텍스트 임베딩 서비스 수정 또는 모델 재학습

### 2. 모니터링 강화
- Railway 로그에서 스케일링 로그 모니터링
- 예측 점수 분포 모니터링
- 이상치 감지 알림 설정

### 3. 성능 테스트
- 수정 후 DeepFM 예측 정확도 측정
- Multi-Tower와 비교 평가
- A/B 테스트 고려

---

## 📚 참고 자료

- **분석 스크립트**: `scripts/analyze_suwon_predictions.py`
- **입력 피처 분석**: `scripts/analyze_input_features.py`
- **테스트 스크립트**: `scripts/test_new_model_api.py`
- **Scaler 파라미터**: `models/scaler_params.json`
- **학습 전처리**: `scripts/step2_aggregate_features.py`

---

## ✅ 체크리스트

- [x] 원본 파일 백업 완료
- [x] 스케일러 로딩 로그 강화
- [x] User 피처 스케일링 로직 수정
- [x] Business 피처 스케일링 로직 수정
- [x] 코드 linter 통과
- [x] 문서화 완료
- [ ] Railway 재배포 (사용자 실행 필요)
- [ ] 예측 테스트 실행 (재배포 후)
- [ ] 결과 검증

---

**작성 완료일**: 2024
**다음 단계**: Railway 재배포 및 테스트 실행

