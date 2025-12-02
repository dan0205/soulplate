import React, { useState, useEffect } from 'react';
import { businessAPI } from '../../../services/api';

// 17개 ABSA 특성에 대한 자연어 매핑
const ABSA_TEXT_MAP = {
  '맛': {
    positive: '맛이 뛰어나요',
    negative: '맛이 아쉬워요',
    icon: '🍜'
  },
  '짠맛': {
    positive: '간이 적당해요',
    negative: '너무 짜요',
    icon: '🧂'
  },
  '매운맛': {
    positive: '매운맛이 좋아요',
    negative: '너무 매워요',
    icon: '🌶️'
  },
  '단맛': {
    positive: '달콤해요',
    negative: '너무 달아요',
    icon: '🍯'
  },
  '느끼함': {
    positive: '느끼하지 않아요',
    negative: '느끼해요',
    icon: '🧈'
  },
  '담백함': {
    positive: '담백해요',
    negative: '담백함이 부족해요',
    icon: '🥗'
  },
  '고소함': {
    positive: '고소해요',
    negative: '고소함이 부족해요',
    icon: '🥜'
  },
  '품질/신선도': {
    positive: '재료가 신선해요',
    negative: '신선도가 아쉬워요',
    icon: '✨'
  },
  '양': {
    positive: '양이 푸짐해요',
    negative: '양이 적어요',
    icon: '🍽️'
  },
  '서비스': {
    positive: '친절한 서비스',
    negative: '서비스가 아쉬워요',
    icon: '👨‍🍳'
  },
  '가격': {
    positive: '가성비가 좋아요',
    negative: '가격이 비싸요',
    icon: '💰'
  },
  '쾌적함/청결도': {
    positive: '깔끔하고 청결해요',
    negative: '청결이 아쉬워요',
    icon: '🧹'
  },
  '소음': {
    positive: '조용해요',
    negative: '시끄러워요',
    icon: '🔇'
  },
  '분위기': {
    positive: '분위기가 좋아요',
    negative: '분위기가 아쉬워요',
    icon: '🏠'
  },
  '공간': {
    positive: '공간이 넓어요',
    negative: '공간이 좁아요',
    icon: '📐'
  },
  '주차': {
    positive: '주차가 편해요',
    negative: '주차가 어려워요',
    icon: '🚗'
  },
  '대기': {
    positive: '대기가 짧아요',
    negative: '대기가 길어요',
    icon: '⏱️'
  }
};

// 17개 특성 목록
const ASPECTS = [
  '맛', '짠맛', '매운맛', '단맛', '느끼함', '담백함', '고소함',
  '품질/신선도', '양', '서비스', '가격', '쾌적함/청결도',
  '소음', '분위기', '공간', '주차', '대기'
];

// 순점수 임계값 (이 값 이상/이하만 표시)
const THRESHOLD = 0.1;

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

  // 17개 특성에서 강점/약점 추출 (상위/하위 3개씩)
  const getStrengthsAndWeaknesses = () => {
    const absa = reviewSummary?.absa_features;
    if (!absa) return { strengths: [], weaknesses: [], hasData: false };

    // 각 특성의 순점수 계산 (긍정 - 부정)
    const scores = ASPECTS.map(aspect => {
      const positive = absa[`${aspect}_긍정`] || 0;
      const negative = absa[`${aspect}_부정`] || 0;
      const netScore = positive - negative;
      return { aspect, netScore };
    });

    // 강점: 순점수가 임계값 이상인 것들 중 상위 3개
    const strengths = scores
      .filter(item => item.netScore > THRESHOLD)
      .sort((a, b) => b.netScore - a.netScore)
      .slice(0, 3)
      .map(item => ({
        icon: ABSA_TEXT_MAP[item.aspect].icon,
        text: ABSA_TEXT_MAP[item.aspect].positive,
        score: item.netScore
      }));

    // 약점: 순점수가 -임계값 이하인 것들 중 하위 3개
    const weaknesses = scores
      .filter(item => item.netScore < -THRESHOLD)
      .sort((a, b) => a.netScore - b.netScore)
      .slice(0, 3)
      .map(item => ({
        icon: ABSA_TEXT_MAP[item.aspect].icon,
        text: ABSA_TEXT_MAP[item.aspect].negative,
        score: Math.abs(item.netScore)
      }));

    return {
      strengths,
      weaknesses,
      hasData: true
    };
  };

  const { strengths, weaknesses, hasData } = getStrengthsAndWeaknesses();
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
        {loading ? (
          <div className="absa-analyzing-message">
            <span>🔄</span>
            <span>AI 분석 정보 로딩 중...</span>
          </div>
        ) : hasData ? (
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

            {/* 둘 다 없는 경우 (모든 점수가 임계값 이하인 경우) */}
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
