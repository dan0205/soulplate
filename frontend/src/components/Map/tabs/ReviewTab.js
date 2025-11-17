import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { businessAPI, reviewAPI } from '../../../services/api';
import { useAuth } from '../../../context/AuthContext';

const ReviewTab = ({ businessId }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState('latest');
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  
  // 리뷰 작성/수정 상태
  const [writingMode, setWritingMode] = useState(null); // null | 'create' | 'edit' | 'reply'
  const [editingReview, setEditingReview] = useState(null);
  const [replyingTo, setReplyingTo] = useState(null);
  const [formData, setFormData] = useState({ stars: 5, text: '' });
  const [submitting, setSubmitting] = useState(false);
  
  // 답글 관련 상태
  const [expandedReplies, setExpandedReplies] = useState(new Set());
  const [repliesData, setRepliesData] = useState({}); // reviewId -> replies array
  
  // Kebab 메뉴 상태
  const [openMenu, setOpenMenu] = useState(null);

  useEffect(() => {
    loadReviews();
  }, [businessId, sortBy]);

  const loadReviews = async (loadMore = false) => {
    try {
      setLoading(true);
      const currentPage = loadMore ? page + 1 : 1;
      const response = await businessAPI.getReviews(businessId, {
        sort: sortBy,
        skip: (currentPage - 1) * 10,
        limit: 10
      });
      
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
      setReviews([]);
    } finally {
      setLoading(false);
    }
  };

  // 답글 토글
  const toggleReplies = async (reviewId) => {
    const newExpanded = new Set(expandedReplies);
    
    if (newExpanded.has(reviewId)) {
      newExpanded.delete(reviewId);
    } else {
      newExpanded.add(reviewId);
      
      // 답글 로드 (아직 로드하지 않은 경우)
      if (!repliesData[reviewId]) {
        try {
          const response = await reviewAPI.getReplies(reviewId);
          setRepliesData({ ...repliesData, [reviewId]: response.data });
        } catch (error) {
          console.error('답글 로드 실패:', error);
          setRepliesData({ ...repliesData, [reviewId]: [] });
        }
      }
    }
    
    setExpandedReplies(newExpanded);
  };

  // 리뷰/답글 작성
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.text.trim()) {
      alert('내용을 입력해주세요.');
      return;
    }
    
    try {
      setSubmitting(true);
      
      if (writingMode === 'create') {
        // 새 리뷰 작성
        await businessAPI.createReview(businessId, {
          stars: formData.stars,
          text: formData.text
        });
        alert('리뷰가 작성되었습니다!');
      } else if (writingMode === 'edit') {
        // 리뷰 수정
        await reviewAPI.update(editingReview.id, {
          stars: formData.stars,
          text: formData.text
        });
        alert('리뷰가 수정되었습니다!');
      } else if (writingMode === 'reply') {
        // 답글 작성
        await reviewAPI.createReply(replyingTo, {
          text: formData.text
        });
        alert('답글이 작성되었습니다!');
      }
      
      // 초기화
      setFormData({ stars: 5, text: '' });
      setWritingMode(null);
      setEditingReview(null);
      setReplyingTo(null);
      
      // 리뷰 목록 새로고침
      loadReviews();
    } catch (error) {
      console.error('작성/수정 실패:', error);
      alert(error.response?.data?.detail || '작업에 실패했습니다.');
    } finally {
      setSubmitting(false);
    }
  };

  // 리뷰 삭제
  const handleDelete = async (reviewId) => {
    if (!window.confirm('정말 삭제하시겠습니까? 답글도 함께 삭제됩니다.')) {
      return;
    }
    
    try {
      await reviewAPI.delete(reviewId);
      alert('삭제되었습니다.');
      loadReviews();
      setOpenMenu(null);
    } catch (error) {
      console.error('삭제 실패:', error);
      alert(error.response?.data?.detail || '삭제에 실패했습니다.');
    }
  };

  // 수정 시작
  const handleEditStart = (review) => {
    setEditingReview(review);
    setFormData({ stars: review.stars || 5, text: review.text });
    setWritingMode('edit');
    setOpenMenu(null);
    // 작성칸으로 스크롤
    setTimeout(() => {
      document.querySelector('.bottom-write-bar')?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  // 답글 시작
  const handleReplyStart = (reviewId) => {
    setReplyingTo(reviewId);
    setFormData({ stars: 5, text: '' });
    setWritingMode('reply');
    setOpenMenu(null);
    // 작성칸으로 스크롤
    setTimeout(() => {
      document.querySelector('.bottom-write-bar')?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  const handleUserClick = (userId) => {
    if (userId) {
      navigate(`/profile/${userId}`);
    }
  };

  // 메뉴 닫기 (외부 클릭)
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (!e.target.closest('.kebab-menu')) {
        setOpenMenu(null);
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  // Kebab 메뉴 컴포넌트
  const KebabMenu = ({ review }) => {
    const isOwner = user && user.id === review.user_id;
    const isOpen = openMenu === review.id;
    
    return (
      <div className="kebab-menu">
        <button 
          className="kebab-btn"
          onClick={(e) => {
            e.stopPropagation();
            setOpenMenu(isOpen ? null : review.id);
          }}
        >
          ⋮
        </button>
        {isOpen && (
          <div className="kebab-dropdown">
            {isOwner ? (
              <>
                <button onClick={() => handleEditStart(review)}>
                  ✏️ 수정
                </button>
                <button onClick={() => handleDelete(review.id)} className="danger">
                  🗑️ 삭제
                </button>
              </>
            ) : (
              <button onClick={() => handleReplyStart(review.id)}>
                💬 답글 달기
              </button>
            )}
          </div>
        )}
      </div>
    );
  };

  // Useful 클릭 핸들러
  const handleUsefulClick = async (reviewId) => {
    try {
      await reviewAPI.incrementUseful(reviewId);
      // 로컬 상태 업데이트
      setReviews(reviews.map(r => 
        r.id === reviewId ? { ...r, useful: (r.useful || 0) + 1 } : r
      ));
      // 답글도 업데이트
      const newRepliesData = { ...repliesData };
      Object.keys(newRepliesData).forEach(parentId => {
        newRepliesData[parentId] = newRepliesData[parentId].map(r =>
          r.id === reviewId ? { ...r, useful: (r.useful || 0) + 1 } : r
        );
      });
      setRepliesData(newRepliesData);
    } catch (error) {
      console.error('Useful 증가 실패:', error);
    }
  };

  // 리뷰 아이템 컴포넌트
  const ReviewItem = ({ review, isReply = false }) => (
    <div 
      className={`review-item ${isReply ? 'reply-item' : ''} ${replyingTo === review.id ? 'replying-target' : ''}`}
    >
      {/* 아바타 (왼쪽, 3줄 높이) */}
      <div 
        className="review-avatar clickable"
        onClick={() => handleUserClick(review.user_id)}
      >
        {review.username ? review.username.charAt(0).toUpperCase() : 'U'}
      </div>
      
      {/* 컨텐츠 (아바타 오른쪽) */}
      <div className="review-content">
        {/* 첫 줄: 이름 + 별점 + 케밥 */}
        <div className="review-first-line">
          <span 
            className="review-username clickable"
            onClick={() => handleUserClick(review.user_id)}
          >
            {review.username || '익명'}
          </span>
          {!isReply && review.stars && (
            <span className="review-stars">
              {'⭐'.repeat(Math.floor(review.stars))}
            </span>
          )}
          {/* Kebab 메뉴 (로그인한 경우만) */}
          {user && <KebabMenu review={review} />}
        </div>
        
        {/* 둘째 줄: 리뷰 텍스트 */}
        <p className="review-text">{review.text}</p>
        
        {/* 셋째 줄: useful + 날짜 */}
        <div className="review-footer">
          <button 
            className="useful-btn"
            onClick={() => handleUsefulClick(review.id)}
          >
            👍 {review.useful || 0}
          </button>
          <span className="review-date">
            {new Date(review.created_at || review.date).toLocaleDateString()}
          </span>
        </div>
        
        {/* 답글 토글 버튼 (최상위 리뷰만, 답글이 있는 경우) */}
        {!isReply && review.reply_count > 0 && (
          <button 
            className="toggle-replies-btn"
            onClick={() => toggleReplies(review.id)}
          >
            {expandedReplies.has(review.id) ? '▼' : '▶'} 답글 {review.reply_count}개
          </button>
        )}
        
        {/* 답글 목록 (review-content 안으로 이동) */}
        {!isReply && expandedReplies.has(review.id) && repliesData[review.id] && (
          <div className="replies-list">
            {repliesData[review.id].map(reply => (
              <ReviewItem key={reply.id} review={reply} isReply={true} />
            ))}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="review-tab">
      {/* 상단 헤더 */}
      <div className="review-header-top">
        <h3 className="review-section-title">리뷰</h3>
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
              <ReviewItem key={review.id || review.review_id} review={review} />
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
      
      {/* 하단 고정 작성칸 (position: fixed) */}
      {user && (
        <div 
          className={`bottom-write-bar-fixed ${writingMode ? 'expanded' : ''}`}
          onClick={() => {
            if (!writingMode) {
              setWritingMode('create');
              setFormData({ stars: 5, text: '' });
            }
          }}
        >
          {!writingMode ? (
            <div className="bottom-write-placeholder">
              <span className="placeholder-text">리뷰를 작성해주세요...</span>
              <span className="placeholder-icon">✍️</span>
            </div>
          ) : (
            <form className="bottom-write-form" onSubmit={handleSubmit} onClick={(e) => e.stopPropagation()}>
              {/* 모드 표시 */}
              <div className="write-form-header">
                {writingMode === 'create' && <h4>✍️ 리뷰 작성</h4>}
                {writingMode === 'edit' && <h4>✏️ 리뷰 수정</h4>}
                {writingMode === 'reply' && <h4>💬 답글 작성</h4>}
                <button 
                  type="button" 
                  className="btn-close"
                  onClick={() => {
                    setWritingMode(null);
                    setEditingReview(null);
                    setReplyingTo(null);
                    setFormData({ stars: 5, text: '' });
                  }}
                >
                  ✕
                </button>
              </div>
              
              {/* 별점 선택 (리뷰 작성/수정 시만) */}
              {writingMode !== 'reply' && (
                <div className="form-group">
                  <label>별점 선택</label>
                  <div className="star-rating">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        key={star}
                        type="button"
                        className={`star-btn ${star <= formData.stars ? 'active' : ''}`}
                        onClick={() => setFormData({ ...formData, stars: star })}
                      >
                        ⭐
                      </button>
                    ))}
                    <span className="star-value">{formData.stars}.0</span>
                  </div>
                </div>
              )}
              
              {/* 텍스트 입력 */}
              <div className="form-group">
                <label>{writingMode === 'reply' ? '답글 내용' : '리뷰 내용'}</label>
                <textarea
                  className="review-textarea"
                  placeholder={writingMode === 'reply' ? '답글을 입력해주세요...' : '이 음식점에 대한 솔직한 리뷰를 작성해주세요...'}
                  value={formData.text}
                  onChange={(e) => setFormData({ ...formData, text: e.target.value })}
                  rows={5}
                  required
                />
              </div>
              
              {/* 사진 업로드 (준비 중) */}
              {writingMode !== 'reply' && (
                <div className="form-group">
                  <label>사진 업로드</label>
                  <div className="photo-upload-placeholder">
                    📷 사진 업로드 기능은 준비중입니다
                  </div>
                </div>
              )}
              
              {/* 제출 버튼 */}
              <div className="form-actions">
                <button 
                  type="submit" 
                  className="btn-submit-review"
                  disabled={submitting || !formData.text.trim()}
                >
                  {submitting ? '작성 중...' : 
                   writingMode === 'edit' ? '수정 완료' : 
                   writingMode === 'reply' ? '답글 등록' : '리뷰 등록'}
                </button>
              </div>
            </form>
          )}
        </div>
      )}
      
      {!user && (
        <div className="login-required-message-fixed">
          리뷰를 작성하려면 로그인이 필요합니다.
        </div>
      )}
    </div>
  );
};

export default ReviewTab;
