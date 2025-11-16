-- ============================================================================
-- 성능 최적화를 위한 데이터베이스 인덱스 추가
-- ============================================================================
-- 작성일: 2025-11-16
-- 목적: API 응답 시간을 3~5초에서 0.3~0.5초로 단축
-- ============================================================================

-- pg_trgm 확장 활성화 (LIKE 검색 최적화)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- 1. reviews 테이블 인덱스
-- ============================================================================

-- user_id: N+1 쿼리 최적화 (user별 리뷰 수 조회)
CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id);

-- business_id: 비즈니스별 리뷰 조회
CREATE INDEX IF NOT EXISTS idx_reviews_business_id ON reviews(business_id);

-- created_at: 최신순 정렬
CREATE INDEX IF NOT EXISTS idx_reviews_created_at ON reviews(created_at DESC);

-- useful: 유용한 리뷰순 정렬
CREATE INDEX IF NOT EXISTS idx_reviews_useful ON reviews(useful DESC);

-- 복합 인덱스: business_id + created_at (리뷰 목록 조회 최적화)
CREATE INDEX IF NOT EXISTS idx_reviews_business_created ON reviews(business_id, created_at DESC);

-- ============================================================================
-- 2. businesses 테이블 인덱스
-- ============================================================================

-- latitude, longitude: 지도 API 위치 기반 검색
CREATE INDEX IF NOT EXISTS idx_businesses_lat_lng ON businesses(latitude, longitude);

-- review_count: 리뷰 개수순 정렬
CREATE INDEX IF NOT EXISTS idx_businesses_review_count ON businesses(review_count DESC);

-- name: 이름 검색 (LIKE '%keyword%')
CREATE INDEX IF NOT EXISTS idx_businesses_name_trgm ON businesses USING gin(name gin_trgm_ops);

-- categories: 카테고리 검색
CREATE INDEX IF NOT EXISTS idx_businesses_categories_trgm ON businesses USING gin(categories gin_trgm_ops);

-- city: 도시별 필터링
CREATE INDEX IF NOT EXISTS idx_businesses_city ON businesses(city);

-- business_id: 비즈니스 상세 조회
CREATE INDEX IF NOT EXISTS idx_businesses_business_id ON businesses(business_id);

-- ============================================================================
-- 3. user_business_predictions 테이블 인덱스
-- ============================================================================

-- user_id, business_id: AI 예측 캐시 조회 (복합 인덱스)
CREATE INDEX IF NOT EXISTS idx_predictions_user_business ON user_business_predictions(user_id, business_id);

-- deepfm_score: DeepFM 점수순 정렬
CREATE INDEX IF NOT EXISTS idx_predictions_deepfm ON user_business_predictions(user_id, deepfm_score DESC);

-- multitower_score: Multi-Tower 점수순 정렬
CREATE INDEX IF NOT EXISTS idx_predictions_multitower ON user_business_predictions(user_id, multitower_score DESC);

-- ============================================================================
-- 4. users 테이블 인덱스
-- ============================================================================

-- username: 로그인 및 사용자 조회
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- email: 이메일로 사용자 조회
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- yelp_user_id: Yelp 데이터 매칭
CREATE INDEX IF NOT EXISTS idx_users_yelp_user_id ON users(yelp_user_id);

-- ============================================================================
-- 인덱스 생성 완료
-- ============================================================================

-- 생성된 인덱스 확인
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

