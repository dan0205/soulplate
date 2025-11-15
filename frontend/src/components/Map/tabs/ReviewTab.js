import React, { useState, useEffect } from 'react';
import { businessAPI } from '../../../services/api';

const ReviewTab = ({ businessId }) => {
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState('latest');
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  useEffect(() => {
    loadReviews();
  }, [businessId, sortBy]);

  const loadReviews = async (loadMore = false) => {
    try {
      setLoading(true);
      const currentPage = loadMore ? page + 1 : 1;
      const response = await businessAPI.getReviews(businessId, {
        sort: sortBy,
        offset: (currentPage - 1) * 10,
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

  return (
    <div className="review-tab">
      {/* 상단 헤더 */}
      <div className="review-header">
        <button className="btn-write-review">
          ✍️ 리뷰 작성
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
                  <div className="user-avatar">
                    {review.username ? review.username.charAt(0).toUpperCase() : 'U'}
                  </div>
                  <div className="user-info">
                    <span className="user-name">{review.username || '익명'}</span>
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

