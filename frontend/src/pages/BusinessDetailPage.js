/**
 * 비즈니스 상세 페이지
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { businessAPI, reviewAPI } from '../services/api';
import AIPrediction from '../components/AIPrediction';
import { ABSAFeaturesDetailed } from '../components/ABSAFeatures';
import Avatar from '../components/Avatar';
import './BusinessDetail.css';

const BusinessDetailPage = () => {
  const { businessId } = useParams();
  // react router가 url의 동적 파라미터에서 businessId를 변수로 추출해준다 
  const navigate = useNavigate();
  
  const [business, setBusiness] = useState(null);
  // api에서 받아온 가게 정보 1개를 저장한다 
  const [reviews, setReviews] = useState([]);
  // api에서 받아온 리뷰 목록을 저장한다 
  const [loading, setLoading] = useState(true);
  // 가게 정보를 불러오는 중인지 나타냄 
  const [error, setError] = useState('');
  // 데이터 로딩 중 에러가 발생했는지
  const [hasMore, setHasMore] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [reviewSkip, setReviewSkip] = useState(0);
  const reviewsEndRef = useRef(null); 
  
  
  const [reviewForm, setReviewForm] = useState({
    stars: 5,
    text: ''
  });
  // 사용자가 리뷰 작성 폼에 입력 중인 별점과 텍스트를 실시간으로 저장한다 
  const [submitting, setSubmitting] = useState(false);
  // 리뷰 제출 버튼을 눌렀을 때, API에 전송 중인지 판단한다 

  useEffect(() => {
    loadBusinessDetails();
    // 리뷰 초기화 및 첫 로드
    setReviews([]);
    setReviewSkip(0);
    setHasMore(true);
    loadReviews(0, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [businessId]);
  // 페이지가 처음 열리거나, url의 businessId가 바뀔 때, useEffect를 실행한다 

  const loadBusinessDetails = async () => {
    try {
      const response = await businessAPI.get(businessId);
      setBusiness(response.data);
    } catch (err) {
      setError('Failed to load business details');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  // GET /api/businesses/{businessId} 호출하여 가게 정보를 business에 저장한다 

  const loadReviews = useCallback(async (skip = 0, isInitial = false) => {
    if (loadingMore && !isInitial) return;
    
    try {
      if (!isInitial) {
        setLoadingMore(true);
      }
      const limit = 20;
      const response = await businessAPI.getReviews(businessId, { skip, limit });
      const newReviews = response.data;
      
      if (isInitial) {
        setReviews(newReviews);
      } else {
        setReviews(prev => [...prev, ...newReviews]);
      }
      
      // 더 불러올 리뷰가 있는지 확인
      if (newReviews.length < limit) {
        setHasMore(false);
      }
      
      setReviewSkip(skip + newReviews.length);
    } catch (err) {
      console.error('Failed to load reviews:', err);
    } finally {
      setLoadingMore(false);
    }
  }, [businessId, loadingMore]);
  // GET /api/businesses/{businessId}/reviews 호출하여 리뷰 목록을 setReviews에 저장한다

  // 무한 스크롤 핸들러
  const handleScroll = useCallback(() => {
    if (loadingMore || !hasMore) return;
    
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;
    
    // 끝에서 200px 전에 도달하면 다음 페이지 로드
    if (scrollTop + windowHeight >= documentHeight - 200) {
      loadReviews(reviewSkip, false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reviewSkip, hasMore, loadingMore]);

  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  // useful 버튼 클릭 핸들러
  const handleUsefulClick = async (reviewId, currentUseful) => {
    try {
      // Optimistic update
      setReviews(prev => prev.map(review => 
        review.id === reviewId 
          ? { ...review, useful: (review.useful || 0) + 1 }
          : review
      ));
      
      await reviewAPI.incrementUseful(reviewId);
    } catch (err) {
      // 실패 시 롤백
      setReviews(prev => prev.map(review => 
        review.id === reviewId 
          ? { ...review, useful: currentUseful }
          : review
      ));
      console.error('Failed to increment useful:', err);
    }
  };

  const handleSubmitReview = async (e) => {
    e.preventDefault();
    // form이 제출될 때 브라우저가 새로고침되는 기본 동작을 막는다 
    setSubmitting(true);
    // 제출중 상태로 바꾸고, 버튼을 비활성화한다 
    
    try {
      await businessAPI.createReview(businessId, reviewForm);
      // POST /api/businesses/{businessId}/reviews 호출하여 리뷰를 생성한다  
      alert('Review submitted successfully! 🎉');
      setReviewForm({ stars: 5, text: '' });
      // 리뷰 목록 초기화 및 재로드
      setReviews([]);
      setReviewSkip(0);
      setHasMore(true);
      loadReviews(0, true);
      // from을 제출한 후, 방금 작성한 리뷰가 포함된 새 목록을 서버에서 다시 불러와 화면을 갱신한다 
      // 홈페이지로 돌아가면 추천이 업데이트됨
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to submit review');
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading business details...</p>
      </div>
    );
  } // 로딩 중일때 spinner를 보여준다 

  if (error || !business) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error || 'Business not found'}</p>
        <button onClick={() => navigate('/')}>Go Back</button>
      </div>
    );
  } // 에러가 났거나 business 데이터가 없으면 에러를 보여준다 

  return (
    <div className="business-detail-container">
      <button className="btn-back" onClick={() => navigate('/')}>← Back to Home</button>
      
      <div className="business-header">
        <h1>{business.name}</h1>
        <div className="business-info">
          <span className="reviews">📝 {business.review_count} reviews</span>
          <span className={business.is_open ? 'status-open' : 'status-closed'}>
            {business.is_open ? '🟢 Open' : '🔴 Closed'}
          </span>
        </div>
        <p className="categories">{business.categories || 'No category'}</p>
        <p className="address">
          📍 {business.address}, {business.city}, {business.state}
        </p>
      </div>

      {/* AI 예측 별점 섹션 */}
      {business.ai_prediction && (
        <AIPrediction prediction={business.ai_prediction} />
      )}

      {/* ABSA 특징 섹션 */}
      {business.absa_features && (
        <ABSAFeaturesDetailed 
          absaFeatures={business.absa_features}
          topFeatures={business.top_features}
        />
      )}

      <div className="review-section">
        <h2>Write a Review</h2>
        <form onSubmit={handleSubmitReview} className="review-form">
          <div className="form-group">
            <label>Rating</label>
            <div className="star-rating">
              {[1, 2, 3, 4, 5].map((star) => (
                <span
                  key={star}
                  className={star <= reviewForm.stars ? 'star filled' : 'star'}
                  onClick={() => setReviewForm({ ...reviewForm, stars: star })}
                >
                  ⭐
                </span>
              ))}
            </div>
          </div>
          
          <div className="form-group">
            <label>Your Review</label>
            <textarea
              value={reviewForm.text}
              onChange={(e) => setReviewForm({ ...reviewForm, text: e.target.value })}
              placeholder="Share your experience..."
              rows="4"
              required
            />
          </div>
          
          <button type="submit" className="btn-primary" disabled={submitting}>
            {submitting ? 'Submitting...' : 'Submit Review'}
          </button>
        </form>
      </div>

      <div className="reviews-section">
        <h2>Recent Reviews {reviews.length > 0 && `(${reviews.length})`}</h2>
        {reviews.length === 0 && !loadingMore ? (
          <p className="no-reviews">No reviews yet. Be the first to review!</p>
        ) : (
          <>
            <div className="reviews-list">
              {reviews.map((review) => (
                <div key={review.id} className="review-item">
                  <div className="review-header">
                    <div 
                      className="review-author-section"
                      onClick={() => navigate(`/profile/${review.user_id}`)}
                    >
                      <Avatar username={review.username} size="small" />
                      <span className="review-author">{review.username}</span>
                    </div>
                    <span className="review-stars">
                      {'⭐'.repeat(review.stars)}
                    </span>
                    <span className="review-date">
                      {new Date(review.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="review-text">{review.text}</p>
                  <div className="review-footer">
                    <button 
                      className="useful-button"
                      onClick={() => handleUsefulClick(review.id, review.useful || 0)}
                    >
                      👍 {review.useful || 0}
                    </button>
                  </div>
                </div>
              ))}
            </div>
            {loadingMore && (
              <div className="loading-more">
                <p>Loading more reviews...</p>
              </div>
            )}
            {!hasMore && reviews.length > 0 && (
              <div className="no-more-reviews">
                <p>No more reviews to load</p>
              </div>
            )}
            <div ref={reviewsEndRef} />
          </>
        )}
      </div>
    </div>
  );
};

export default BusinessDetailPage;

