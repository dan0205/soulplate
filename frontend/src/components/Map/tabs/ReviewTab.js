import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { businessAPI } from '../../../services/api';
import { useAuth } from '../../../context/AuthContext';

const ReviewTab = ({ businessId }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState('latest');
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  
  // 리뷰 작성 폼 상태
  const [isWriting, setIsWriting] = useState(false);
  const [newReview, setNewReview] = useState({ stars: 5, text: '' });
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    loadReviews();
  }, [businessId, sortBy]);

  const loadReviews = async (loadMore = false) => {
    try {
      setLoading(true);
      const currentPage = loadMore ? page + 1 : 1;
      const response = await businessAPI.getReviews(businessId, {
        sort: sortBy,
        skip: (currentPage - 1) * 10,  // offset → skip (백엔드 파라미터명과 일치)
        limit: 10
      });
      
      // API 응답이 배열로 직접 오는 경우 처리
      const reviewsData = Array.isArray(response.data) ? response.data : (response.data.reviews || []);
      
      if (loadMore) {
        setReviews([...reviews, ...reviewsData]);
        setPage(currentPage);
      } else {
        setReviews(reviewsData);
        setPage(1);
      }
      
      setHasMore(reviewsData.length === 10);
    } catch (error) {
      console.error('리뷰 로드 실패:', error);
      console.error('Error details:', error.response?.data || error.message);
      setReviews([]); // 에러 시 빈 배열로 설정
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitReview = async (e) => {
    e.preventDefault();
    
    if (!newReview.text.trim()) {
      alert('리뷰 내용을 입력해주세요.');
      return;
    }
    
    try {
      setSubmitting(true);
      await businessAPI.createReview(businessId, {
        stars: newReview.stars,
        text: newReview.text
      });
      
      // 폼 초기화
      setNewReview({ stars: 5, text: '' });
      setIsWriting(false);
      
      // 리뷰 목록 새로고침
      loadReviews();
      
      alert('리뷰가 성공적으로 작성되었습니다!');
    } catch (error) {
      console.error('리뷰 작성 실패:', error);
      alert(error.response?.data?.detail || '리뷰 작성에 실패했습니다.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleUserClick = (userId) => {
    if (userId) {
      navigate(`/profile/${userId}`);
    }
  };

  return (
    <div className="review-tab">
      {/* 상단 헤더 */}
      <div className="review-header">
        <button 
          className="btn-write-review"
          onClick={() => setIsWriting(!isWriting)}
        >
          {isWriting ? '✖ 취소' : '✍️ 리뷰 작성'}
        </button>
        <select 
          className="review-sort"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
        >
          <option value="latest">최신순</option>
          <option value="useful">추천순</option>
        </select>
      </div>

      {/* 리뷰 작성 폼 */}
      {isWriting && (
        <form className="review-write-form" onSubmit={handleSubmitReview}>
          <div className="form-group">
            <label>별점 선택</label>
            <div className="star-rating">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  type="button"
                  className={`star-btn ${star <= newReview.stars ? 'active' : ''}`}
                  onClick={() => setNewReview({ ...newReview, stars: star })}
                >
                  ⭐
                </button>
              ))}
              <span className="star-value">{newReview.stars}.0</span>
            </div>
          </div>
          
          <div className="form-group">
            <label>리뷰 내용</label>
            <textarea
              className="review-textarea"
              placeholder="이 음식점에 대한 솔직한 리뷰를 작성해주세요..."
              value={newReview.text}
              onChange={(e) => setNewReview({ ...newReview, text: e.target.value })}
              rows={5}
              required
            />
          </div>
          
          <div className="form-group">
            <label>사진 업로드</label>
            <div className="photo-upload-placeholder">
              📷 사진 업로드 기능은 준비중입니다
            </div>
          </div>
          
          <div className="form-actions">
            <button 
              type="submit" 
              className="btn-submit-review"
              disabled={submitting || !newReview.text.trim()}
            >
              {submitting ? '작성 중...' : '리뷰 등록'}
            </button>
          </div>
        </form>
      )}

      {/* 리뷰 리스트 */}
      {loading && reviews.length === 0 ? (
        <div className="loading">로딩 중...</div>
      ) : reviews.length === 0 ? (
        <div className="no-reviews">아직 리뷰가 없습니다</div>
      ) : (
        <>
          <div className="reviews-list">
            {reviews.map((review) => (
              <div key={review.id || review.review_id} className="review-item">
                <div className="review-header">
                  <div 
                    className="user-avatar clickable"
                    onClick={() => handleUserClick(review.user_id)}
                  >
                    {review.username ? review.username.charAt(0).toUpperCase() : 'U'}
                  </div>
                  <div className="user-info">
                    <span 
                      className="user-name clickable"
                      onClick={() => handleUserClick(review.user_id)}
                    >
                      {review.username || '익명'}
                    </span>
                    <span className="user-stats">리뷰 {review.user_total_reviews || 0}개</span>
                  </div>
                </div>
                
                <div className="review-rating">
                  {'⭐'.repeat(Math.floor(review.stars))} {review.stars}
                </div>
                
                {/* ABSA 감정 표시 */}
                {review.absa_sentiment && (
                  <div className="absa-sentiment">
                    {review.absa_sentiment.food !== undefined && (
                      <span className="sentiment-tag">
                        🍜{review.absa_sentiment.food > 0 ? '+' : ''}{review.absa_sentiment.food}
                      </span>
                    )}
                    {review.absa_sentiment.service !== undefined && (
                      <span className="sentiment-tag">
                        👨‍🍳{review.absa_sentiment.service > 0 ? '+' : ''}{review.absa_sentiment.service}
                      </span>
                    )}
                    {review.absa_sentiment.atmosphere !== undefined && (
                      <span className="sentiment-tag">
                        🏠{review.absa_sentiment.atmosphere > 0 ? '+' : ''}{review.absa_sentiment.atmosphere}
                      </span>
                    )}
                  </div>
                )}
                
                <p className="review-text">{review.text}</p>
                
                <div className="review-footer">
                  <span>👍 {review.useful || 0}</span>
                  <span>{new Date(review.created_at || review.date).toLocaleDateString()}</span>
                </div>
              </div>
            ))}
          </div>

          {hasMore && (
            <button 
              className="btn-load-more"
              onClick={() => loadReviews(true)}
              disabled={loading}
            >
              {loading ? '로딩 중...' : '더 보기'}
            </button>
          )}
        </>
      )}
    </div>
  );
};

export default ReviewTab;

