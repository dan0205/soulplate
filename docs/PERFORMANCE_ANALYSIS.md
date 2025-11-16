# API 성능 분석 가이드

이 가이드는 Railway 로그를 통해 API 성능을 분석하고 병목 지점을 찾는 방법을 설명합니다.

## 📊 로깅 시스템 개요

다음 3가지 레벨의 로깅이 추가되었습니다:

### 1. DB 쿼리 로깅 (`database.py`)
- **모든 쿼리 실행 시간 자동 측정**
- 100ms 이상 걸리는 슬로우 쿼리 자동 경고
- 이모지: 🐌 SLOW QUERY

### 2. API 응답 시간 로깅 (미들웨어)
- **모든 API 요청의 응답 시간 측정**
- 1초 이상 걸리는 API 자동 경고
- 이모지: 📊 (모든 요청), 🐢 SLOW API

### 3. 엔드포인트 상세 로깅
- **리뷰 조회 API** (`/api/businesses/{id}/reviews`)
  - Step 1: 비즈니스 조회
  - Step 2: 리뷰 조회
  - Step 3: N+1 쿼리 (user_review_count)
  - 이모지: 🔍, ⏱️, ⚠️ N+1 문제

- **지도 API** (`/api/businesses/map`)
  - Step 1: 비즈니스 쿼리 (위치 필터링)
  - Step 2: 데이터 변환 + AI 캐시 조회
  - 이모지: 🗺️, ⏱️

- **목록 API** (`/api/businesses`)
  - Step 1: 총 개수 조회
  - Step 2: 비즈니스 조회 (정렬)
  - Step 3: 데이터 변환
  - 이모지: 📋, ⏱️

---

## 🔍 Railway 로그 확인 방법

### 1. Railway 대시보드 접속
1. https://railway.app/ 로그인
2. 프로젝트 선택
3. `backend_web` 서비스 클릭

### 2. 로그 보기
1. "Deployments" 탭 클릭
2. 최신 배포 클릭
3. **"View Logs"** 버튼 클릭

### 3. 실시간 모니터링
- 로그는 실시간으로 업데이트됩니다
- 자동 스크롤 활성화 권장
- 필터/검색 기능 사용 가능

---

## 🔎 로그 분석 방법

### 패턴 1: 전체 API 응답 시간 확인

**검색어**: `📊` 또는 `SLOW API`

**예시 로그**:
```
📊 GET /api/businesses/map [200] 3.245s
🐢 SLOW API (3.245s): GET /api/businesses/map
```

**분석**:
- 어떤 API가 느린지 한눈에 파악
- 1초 이상은 `SLOW API`로 자동 표시
- 상태 코드도 함께 확인 (200, 404, 500 등)

---

### 패턴 2: 슬로우 쿼리 찾기

**검색어**: `🐌 SLOW QUERY`

**예시 로그**:
```
🐌 SLOW QUERY (0.542s): SELECT reviews.id, reviews.user_id FROM reviews WHERE reviews.user_id = %s
   Parameters: (123,)
```

**분석**:
- 어떤 쿼리가 느린지 확인
- 100ms 이상 쿼리만 표시
- 파라미터도 출력되어 디버깅 용이
- **→ 인덱스 추가 필요 신호**

---

### 패턴 3: N+1 쿼리 문제 찾기

**검색어**: `⚠️ N+1 문제` 또는 `N+1 Query`

**예시 로그**:
```
🔍 리뷰 조회 시작: business_id=KR_우만동족발집...
  ⏱️  Step 1 (비즈니스 조회): 0.045s
  ⏱️  Step 2 (리뷰 조회): 0.123s, 조회된 리뷰: 20개
    🐌 N+1 Query #1 (0.089s): user_id=5
    🐌 N+1 Query #2 (0.091s): user_id=12
    🐌 N+1 Query #3 (0.087s): user_id=8
    ... (총 20번)
  ⏱️  Step 3 (N+1 쿼리 포함): 1.845s
  ⚠️  N+1 문제: 20개 리뷰 = 20번 추가 쿼리 실행, 총 1.845s
✅ 리뷰 조회 완료: 2.013s
```

**분석**:
- Step 3가 대부분의 시간 차지
- 리뷰 개수만큼 쿼리 실행 (20개 = 20번)
- **→ JOIN이나 subquery로 한 번에 가져오기 필요**

---

### 패턴 4: 지도 API 느린 원인

**검색어**: `🗺️ 지도 API`

**예시 로그**:
```
🗺️ 지도 API 시작: lat=37.27, lng=127.04, radius=10.0km, limit=100
  ⏱️  Step 1 (비즈니스 쿼리): 2.134s, 조회: 45개
  ⏱️  Step 2 (데이터 변환 + AI 캐시): 0.234s
    - AI 캐시 쿼리: 45번, 0.201s
✅ 지도 API 완료: 2.368s
```

**분석**:
- Step 1 (비즈니스 쿼리)이 2초 이상
- 위치 기반 필터링이 느림
- **→ `latitude`, `longitude`에 인덱스 필요**

---

### 패턴 5: 목록 API 느린 원인

**검색어**: `📋 목록 API`

**예시 로그**:
```
📋 목록 API 시작: skip=0, limit=20, sort=review_count, search=None
  ⏱️  Step 1 (총 개수 조회): 0.456s, 총 1250개
  ⏱️  Step 2 (비즈니스 조회): 0.834s, 조회: 20개
  ⏱️  Step 3 (데이터 변환): 0.067s
✅ 목록 API 완료: 1.357s
```

**분석**:
- Step 1 (총 개수 조회)와 Step 2 (정렬 쿼리)가 느림
- **→ 정렬 컬럼(`review_count`)에 인덱스 필요**

---

## 🎯 병목 지점 판단 기준

### 1초 이하: ✅ 정상
- 추가 최적화 불필요

### 1~2초: ⚠️ 주의
- 사용 가능하지만 개선 권장
- 인덱스 추가 검토

### 2~5초: 🔴 심각
- 즉시 최적화 필요
- N+1 쿼리 또는 인덱스 누락

### 5초 이상: 💀 치명적
- 사용자 경험 매우 나쁨
- 긴급 수정 필요

---

## 🛠️ 일반적인 최적화 방법

### 1. 인덱스 추가

**문제**: 슬로우 쿼리 발생
```sql
-- 예: reviews 테이블 쿼리가 느림
SELECT * FROM reviews WHERE user_id = 123;  -- 0.5초
```

**해결**: 인덱스 추가
```sql
CREATE INDEX idx_reviews_user_id ON reviews(user_id);
```

**Railway에서 실행**:
```bash
# Railway CLI 사용
railway connect postgres
\c railway
CREATE INDEX idx_reviews_user_id ON reviews(user_id);
```

### 2. N+1 쿼리 해결

**문제**: 반복문에서 쿼리 실행
```python
for review in reviews:
    # 매번 DB 쿼리 실행 (N번)
    count = db.query(Review).filter(Review.user_id == review.user_id).count()
```

**해결**: 한 번에 가져오기
```python
# 모든 user_id를 한 번에 조회
from sqlalchemy import func
user_ids = [r.user_id for r in reviews]
counts = db.query(
    Review.user_id,
    func.count(Review.id)
).filter(
    Review.user_id.in_(user_ids)
).group_by(Review.user_id).all()

# 딕셔너리로 변환
counts_map = {user_id: count for user_id, count in counts}
```

### 3. 캐싱 추가

**문제**: 같은 데이터를 반복 조회
```python
# 매번 DB 조회
business = db.query(Business).filter(Business.id == id).first()
```

**해결**: Redis 캐싱
```python
import redis
cache = redis.Redis()

# 캐시 확인
cached = cache.get(f"business:{id}")
if cached:
    return json.loads(cached)

# 캐시 없으면 DB 조회 후 저장
business = db.query(Business).filter(Business.id == id).first()
cache.set(f"business:{id}", json.dumps(business_dict), ex=3600)
```

---

## 📝 체크리스트

### Railway 로그 확인 완료
- [ ] `📊` 검색 → 전체 API 응답 시간 확인
- [ ] `🐢 SLOW API` 검색 → 1초 이상 API 찾기
- [ ] `🐌 SLOW QUERY` 검색 → 100ms 이상 쿼리 찾기
- [ ] `⚠️ N+1 문제` 검색 → N+1 쿼리 문제 확인

### 병목 지점 파악
- [ ] 어떤 API가 가장 느린가?
- [ ] 어떤 Step이 가장 오래 걸리는가?
- [ ] N+1 쿼리가 발생하는가?
- [ ] 슬로우 쿼리는 어떤 테이블인가?

### 최적화 계획 수립
- [ ] 필요한 인덱스 목록 작성
- [ ] N+1 쿼리 해결 방법 결정
- [ ] 캐싱이 필요한 부분 식별

---

## 🚀 다음 단계

로그 분석이 완료되면:

1. **인덱스 추가** (가장 빠른 효과)
   - `docs/DB_OPTIMIZATION.md` 참고 (예정)

2. **N+1 쿼리 해결**
   - 코드 수정 필요

3. **캐싱 추가** (선택)
   - Redis 또는 메모리 캐시

4. **재측정**
   - 최적화 후 다시 로그 확인
   - 개선 효과 측정

---

## 💡 팁

### 1. 로그 필터링
Railway 로그에서 검색 기능 활용:
- `🐌` - 슬로우 쿼리만
- `🐢` - 슬로우 API만
- `⚠️` - 경고만
- `ERROR` - 에러만

### 2. 시간대별 분석
- 특정 시간에 느려지는지 확인
- 사용자 많을 때 vs 적을 때 비교

### 3. 로그 다운로드
- Railway에서 로그 export 가능
- 로컬에서 grep으로 상세 분석

---

**작성일**: 2025-11-16  
**최종 수정**: 2025-11-16

