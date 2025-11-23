/**
 * 최근 리뷰 페이지 - 모든 사용자의 리뷰와 답글을 최신순으로 표시
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { reviewAPI } from '../services/api';
import './Profile.css';

const RecentReviewsPage = () => {
  const navigate = useNavigate();
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [skip, setSkip] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [visibleCount, setVisibleCount] = useState(20);
  const [loadingMore, setLoadingMore] = useState(false);

  useEffect(() => {
    loadReviews(0, true);
  }, []);

  const loadReviews = async (skipValue = 0, isInitial = false) => {
    try {
      if (!isInitial) {
        setLoadingMore(true);
      } else {
        setLoading(true);
      }

      const response = await reviewAPI.getRecent({ skip: skipValue, limit: 20 });
      const newReviews = response.data;
      
      if (isInitial) {
        setReviews(newReviews);
      } else {
        setReviews(prev => [...prev, ...newReviews]);
      }
      
      if (newReviews.length < 20) {
        setHasMore(false);
      }
      
      setSkip(skipValue + newReviews.length);
    } catch (err) {
      console.error('Failed to load reviews:', err);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  };

  const handleLoadMore = (e) => {
    e.preventDefault();
    if (visibleCount < reviews.length) {
      setVisibleCount(prev => Math.min(prev + 20, reviews.length));
    } else if (hasMore) {
      loadReviews(skip, false);
      setVisibleCount(prev => prev + 20);
    }
  };

  const handleUserClick = (userId) => {
    navigate(`/profile/${userId}`);
  };

  const handleBusinessClick = (businessId) => {
    navigate('/', { state: { businessId } });
  };

  if (loading && reviews.length === 0) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>리뷰를 불러오는 중...</p>
      </div>
    );
  }

  return (
    <div className="profile-container">
      <div className="profile-header-actions">
        <div className="profile-logo" onClick={() => navigate('/')}>
          Soulplate
        </div>
      </div>

      <div className="reviews-section">
        <h2>최근 리뷰 📝</h2>
        {reviews.length === 0 ? (
          <p className="no-reviews">아직 작성된 리뷰가 없습니다.</p>
        ) : (
          <div style={{ padding: '0 20px' }}>
            {reviews.slice(0, visibleCount).map((review) => (
              <div key={review.id} className="review-minimal-item">
                <div className="review-minimal-header">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flex: 1, minWidth: 0 }}>
                    <h3 
                      className="review-minimal-title"
                      onClick={() => handleBusinessClick(review.business_id)}
                      style={{ cursor: 'pointer', margin: 0, wordWrap: 'break-word' }}
                    >
                      {review.business_name}
                    </h3>
                    <span 
                      className="review-minimal-author"
                      onClick={() => handleUserClick(review.user_id)}
                      style={{ 
                        fontSize: '14px', 
                        color: '#666', 
                        cursor: 'pointer',
                        textDecoration: 'none',
                        transition: 'all 0.2s',
                        display: 'inline-block',
                        width: 'fit-content'
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.color = '#ff6b6b';
                        e.target.style.textDecoration = 'underline';
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.color = '#666';
                        e.target.style.textDecoration = 'none';
                      }}
                    >
                      {review.username}
                    </span>
                  </div>
                  {review.stars && (
                    <div className="review-minimal-rating" style={{ flexShrink: 0 }}>
                      {'⭐'.repeat(Math.round(review.stars))}
                    </div>
                  )}
                </div>
                <div className="review-minimal-meta">
                  <span>{new Date(review.created_at).toLocaleDateString('ko-KR', { 
                    year: 'numeric', 
                    month: '2-digit', 
                    day: '2-digit' 
                  }).replace(/\. /g, '.').replace(/\.$/, '')}</span>
                  <span>👍 {review.useful || 0}명이 도움됨</span>
                </div>
                <p className="review-minimal-text">{review.text}</p>
              </div>
            ))}
            {(reviews.length > visibleCount || hasMore) && (
              <div className="review-load-more-link-minimal show">
                <a href="#" onClick={handleLoadMore}>더보기</a>
              </div>
            )}
            {loadingMore && (
              <div className="loading-more">
                <p>리뷰를 불러오는 중...</p>
              </div>
            )}
            {!hasMore && reviews.length > 0 && reviews.length <= visibleCount && (
              <div className="no-more-reviews">
                <p>모든 리뷰를 불러왔습니다</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default RecentReviewsPage;

