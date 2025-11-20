/**
 * 사용자 프로필 페이지 (다른 사용자)
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { userAPI } from '../services/api';
import Avatar from '../components/Avatar';
import { getMBTIInfo } from '../utils/mbtiDescriptions';
import './Profile.css';

const UserProfilePage = () => {
  const { userId } = useParams();
  const navigate = useNavigate();
  
  const [profile, setProfile] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [reviewSkip, setReviewSkip] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [visibleReviewCount, setVisibleReviewCount] = useState(5);

  useEffect(() => {
    setProfile(null);
    setReviews([]);
    setReviewSkip(0);
    setHasMore(true);
    setLoading(true);
    setVisibleReviewCount(5);
    loadProfile();
    loadReviews(0, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  const loadProfile = async () => {
    try {
      const response = await userAPI.getUserProfile(userId);
      setProfile(response.data);
    } catch (err) {
      setError('Failed to load profile');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadReviews = useCallback(async (skip = 0, isInitial = false) => {
    if (loadingMore && !isInitial) return;
    
    try {
      if (!isInitial) {
        setLoadingMore(true);
      }
      const limit = 20;
      const response = await userAPI.getUserReviews(userId, { skip, limit });
      const newReviews = response.data;
      
      if (isInitial) {
        setReviews(newReviews);
        setVisibleReviewCount(Math.min(5, newReviews.length));
      } else {
        setReviews(prev => [...prev, ...newReviews]);
        setVisibleReviewCount(prev => prev + Math.min(5, newReviews.length));
      }
      
      if (newReviews.length < limit) {
        setHasMore(false);
      }
      
      setReviewSkip(skip + newReviews.length);
    } catch (err) {
      console.error('Failed to load reviews:', err);
    } finally {
      setLoadingMore(false);
    }
  }, [userId, loadingMore]);

  const handleLoadMoreReviews = (e) => {
    e.preventDefault();
    if (visibleReviewCount < reviews.length) {
      // 이미 로드된 리뷰 중에서 더 보여주기
      setVisibleReviewCount(prev => Math.min(prev + 5, reviews.length));
    } else if (hasMore) {
      // 더 많은 리뷰를 API에서 가져오기
      loadReviews(reviewSkip, false);
      setVisibleReviewCount(prev => prev + 5);
    }
  };

  const getTopABSAFeatures = (absaFeatures) => {
    if (!absaFeatures) return [];
    
    const features = Object.entries(absaFeatures).map(([key, value]) => ({
      key,
      value
    }));
    
    features.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    return features.slice(0, 5);
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading profile...</p>
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error || 'Profile not found'}</p>
        <button onClick={() => navigate(-1)}>Go Back</button>
      </div>
    );
  }

  const mbtiInfo = profile.taste_test_mbti_type ? getMBTIInfo(profile.taste_test_mbti_type) : null;

  return (
    <div className="profile-container">
      <div className="profile-header-actions">
        <div className="profile-logo" onClick={() => navigate('/')}>
          Soulplate
        </div>
      </div>
      
      <div className="profile-header">
        <Avatar username={profile.username} size="medium" />
        <div className="profile-info">
          <h1>{profile.username}</h1>
          <div className="profile-stats">
            <span className="stat-inline">Reviews: {profile.review_count}</span>
            <span className="stat-inline">Useful: {profile.useful}</span>
            <span className="stat-inline">Fans: {profile.fans}</span>
          </div>
        </div>
      </div>

      {profile.taste_test_completed && mbtiInfo && (
        <div className="taste-test-section">
          <h2>음식 취향</h2>
          <div className="mbti-box-red">
            <div className="mbti-type-large">
              {profile.taste_test_mbti_type}
            </div>
            <div className="mbti-type-name">
              {mbtiInfo.name}
            </div>
            <div className="mbti-description">
              {mbtiInfo.description}
            </div>
            {mbtiInfo.recommendations && mbtiInfo.recommendations.length > 0 && (
              <div className="mbti-recommendations">
                <div className="recommendations-title">📍 추천 장소</div>
                <ul>
                  {mbtiInfo.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="reviews-section">
        <h2>작성한 리뷰 ({reviews.length})</h2>
        {reviews.length === 0 && !loadingMore ? (
          <p className="no-reviews">작성한 리뷰가 없습니다.</p>
        ) : (
          <>
            <div style={{ padding: '0 20px' }}>
              {reviews.slice(0, visibleReviewCount).map((review) => (
                <div key={review.id} className="review-minimal-item">
                  <div className="review-minimal-header">
                    <h3 
                      className="review-minimal-title"
                      onClick={() => navigate('/', { state: { businessId: review.business.business_id } })}
                    >
                      {review.business.name}
                    </h3>
                    <div className="review-minimal-rating">
                      {'⭐'.repeat(review.stars)}
                    </div>
                  </div>
                  <div className="review-minimal-meta">
                    <span>{new Date(review.created_at).toLocaleDateString('ko-KR', { year: 'numeric', month: '2-digit', day: '2-digit' }).replace(/\. /g, '.').replace(/\.$/, '')}</span>
                    <span>👍 {review.useful || 0}명이 도움됨</span>
                  </div>
                  <p className="review-minimal-text">{review.text}</p>
                </div>
              ))}
              {(reviews.length > visibleReviewCount || (hasMore && !loadingMore)) && (
                <div className="review-load-more-link-minimal show">
                  <a href="#" onClick={handleLoadMoreReviews}>더보기</a>
                </div>
              )}
              {loadingMore && (
                <div className="loading-more">
                  <p>리뷰를 불러오는 중...</p>
                </div>
              )}
              {!hasMore && reviews.length > 0 && reviews.length <= visibleReviewCount && (
                <div className="no-more-reviews">
                  <p>모든 리뷰를 불러왔습니다</p>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default UserProfilePage;


