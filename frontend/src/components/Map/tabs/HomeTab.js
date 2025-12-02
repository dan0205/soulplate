import React, { useState, useEffect } from 'react';
import { businessAPI } from '../../../services/api';

// ABSA 카테고리별 자연어 텍스트 매핑
const ABSA_TEXT = {
  food: {
    positive: '맛이 뛰어나요',
    negative: '맛이 아쉬워요',
    icon: '🍜'
  },
  service: {
    positive: '친절한 서비스',
    negative: '서비스가 아쉬워요',
    icon: '👨‍🍳'
  },
  atmosphere: {
    positive: '분위기가 좋아요',
    negative: '분위기가 아쉬워요',
    icon: '🏠'
  }
};

const HomeTab = ({ restaurant, onSwitchToReview }) => {
  const [reviewSummary, setReviewSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  // 리뷰 요약 데이터 로드
  useEffect(() => {
    const loadReviewSummary = async () => {
      if (!restaurant?.business_id) return;
      
      try {
        setLoading(true);
        const response = await businessAPI.getReviewSummary(restaurant.business_id);
        setReviewSummary(response.data);
      } catch (error) {
        console.error('리뷰 요약 로드 실패:', error);
        setReviewSummary(null);
      } finally {
        setLoading(false);
      }
    };

    loadReviewSummary();
  }, [restaurant?.business_id]);

  // ABSA 점수에서 강점/약점 추출 (최대 3개씩)
  const getStrengthsAndWeaknesses = () => {
    const strengths = [];
    const weaknesses = [];

    const absaData = [
      { key: 'food', score: restaurant.absa_food_avg },
      { key: 'service', score: restaurant.absa_service_avg },
      { key: 'atmosphere', score: restaurant.absa_atmosphere_avg }
    ];

    absaData.forEach(({ key, score }) => {
      if (score == null) return;
      
      const textInfo = ABSA_TEXT[key];
      if (score > 0) {
        strengths.push({
          icon: textInfo.icon,
          text: textInfo.positive,
          score
        });
      } else if (score < 0) {
        weaknesses.push({
          icon: textInfo.icon,
          text: textInfo.negative,
          score: Math.abs(score)
        });
      }
    });

    // 점수 순으로 정렬 후 최대 3개
    strengths.sort((a, b) => b.score - a.score);
    weaknesses.sort((a, b) => b.score - a.score);

    return {
      strengths: strengths.slice(0, 3),
      weaknesses: weaknesses.slice(0, 3)
    };
  };

  const { strengths, weaknesses } = getStrengthsAndWeaknesses();
  const hasABSAData = restaurant.absa_food_avg != null || 
                      restaurant.absa_service_avg != null || 
                      restaurant.absa_atmosphere_avg != null;
  const hasStrengths = strengths.length > 0;
  const hasWeaknesses = weaknesses.length > 0;
  const onlyOneCard = (hasStrengths && !hasWeaknesses) || (!hasStrengths && hasWeaknesses);

  return (
    <div className="home-tab">
      {/* 기본 정보 */}
      <div className="basic-info-section">
        <div className="info-item">
          <div className="info-label">📍 주소</div>
          <div className="info-value">{restaurant.address || '정보 없음'}</div>
        </div>

        <div className="info-item">
          <div className="info-label">🕐 영업시간</div>
          <div className="info-value">준비 중입니다</div>
        </div>

        <div className="info-item">
          <div className="info-label">📞 전화번호</div>
          <div className="info-value">{restaurant.phone || '정보 없음'}</div>
        </div>
      </div>

      {/* AI 브리핑 - 강점/약점 카드 */}
      <div className="absa-features">
        {hasABSAData ? (
          <div className={`strengths-weaknesses ${onlyOneCard ? 'single-card' : ''}`}>
            {/* 강점 카드 */}
            {hasStrengths && (
              <div className={`strength-card ${onlyOneCard ? 'full-width' : ''}`}>
                <div className="card-title">
                  <span>✓</span>
                  <span>이 가게의 강점</span>
                </div>
                <div className="card-items">
                  {strengths.map((item, idx) => (
                    <div key={idx} className="card-item">
                      <span className="card-icon">{item.icon}</span>
                      <span>{item.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 약점 카드 */}
            {hasWeaknesses && (
              <div className={`weakness-card ${onlyOneCard ? 'full-width' : ''}`}>
                <div className="card-title">
                  <span>!</span>
                  <span>아쉬운 점</span>
                </div>
                <div className="card-items">
                  {weaknesses.map((item, idx) => (
                    <div key={idx} className="card-item">
                      <span className="card-icon">{item.icon}</span>
                      <span>{item.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 둘 다 없는 경우 (모든 점수가 0인 경우) */}
            {!hasStrengths && !hasWeaknesses && (
              <div className="absa-neutral-message">
                <span>📊</span>
                <span>현재 분석 결과가 중립적입니다</span>
              </div>
            )}
          </div>
        ) : (
          <div className="absa-analyzing-message">
            <span>🔄</span>
            <span>AI가 리뷰를 분석 중입니다...</span>
          </div>
        )}
      </div>

      {/* 리뷰 요약 */}
      <div className="review-summary">
        {loading ? (
          <div className="loading-message">리뷰 정보 로딩 중...</div>
        ) : reviewSummary && reviewSummary.review_count > 0 ? (
          <>
            <div className="review-summary-content">
              {/* 별점 분포 그래프 */}
              <div className="rating-distribution">
                {[5, 4, 3, 2, 1].map(star => (
                  <div key={star} className="rating-bar-item">
                    <div className="rating-bar">
                      <div 
                        className="rating-bar-fill" 
                        style={{ width: `${reviewSummary.stars_distribution[star] || 0}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* 평균 평점 */}
              <div className="average-rating">
                <div className="average-score">{reviewSummary.avg_stars.toFixed(1)}</div>
                <div className="average-stars">
                  {'⭐'.repeat(Math.round(reviewSummary.avg_stars))}
                </div>
                <div className="review-count">({reviewSummary.review_count}개)</div>
              </div>
            </div>
          </>
        ) : (
          <div className="no-reviews-message">
            <span>📝</span>
            <span>아직 리뷰가 없어요</span>
          </div>
        )}
      </div>

      {/* 리뷰 미리보기 */}
      {reviewSummary && reviewSummary.recent_reviews && reviewSummary.recent_reviews.length > 0 && (
        <div className="review-preview">
          <div className="review-preview-list">
            {reviewSummary.recent_reviews.map(review => (
              <div key={review.id} className="review-preview-item">
                <div className="review-minimal-header">
                  <div className="review-minimal-header-left">
                    <span className="review-minimal-author">{review.username}</span>
                    {review.stars && (
                      <span className="review-stars">
                        {'⭐'.repeat(Math.floor(review.stars))}
                      </span>
                    )}
                  </div>
                </div>
                <div className="review-minimal-meta">
                  <span>{new Date(review.created_at).toLocaleDateString()}</span>
                </div>
                <p className="review-text">{review.text}</p>
                <div className="review-minimal-footer">
                  <span className="useful-count">👍 {review.useful}명이 도움됨</span>
                </div>
              </div>
            ))}
          </div>
          <button 
            className="btn-view-all-reviews"
            onClick={onSwitchToReview}
          >
            모든 리뷰 보기
          </button>
        </div>
      )}
    </div>
  );
};

export default HomeTab;
