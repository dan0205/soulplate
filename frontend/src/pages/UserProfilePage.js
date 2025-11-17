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

  useEffect(() => {
    setProfile(null);
    setReviews([]);
    setReviewSkip(0);
    setHasMore(true);
    setLoading(true);
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
      } else {
        setReviews(prev => [...prev, ...newReviews]);
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
          🍽️ Soulplate
        </div>
      </div>
      
      <div className="profile-header">
        <Avatar username={profile.username} size="large" />
        <div className="profile-info">
          <h1>{profile.username}</h1>
          <div className="profile-stats">
            <div className="stat-item">
              <span className="stat-value">{profile.review_count}</span>
              <span className="stat-label">Reviews</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{profile.useful}</span>
              <span className="stat-label">Useful</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{profile.fans}</span>
              <span className="stat-label">Fans</span>
            </div>
          </div>
        </div>
      </div>

      {profile.taste_test_completed && mbtiInfo && (
        <div className="taste-test-section">
          <h2>🍽️ 음식 취향</h2>
          <div className="taste-test-card">
            <div className="mbti-box">
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
        </div>
      )}

      <div className="reviews-section">
        <h2>작성한 리뷰 ({reviews.length})</h2>
        {reviews.length === 0 && !loadingMore ? (
          <p className="no-reviews">작성한 리뷰가 없습니다.</p>
        ) : (
          <>
            <div className="user-reviews-list">
              {reviews.map((review) => (
                <div key={review.id} className="user-review-item">
                  <div className="review-business-info">
                    <h3 
                      className="business-name-link"
                      onClick={() => navigate(`/business/${review.business.business_id}`)}
                    >
                      {review.business.name}
                    </h3>
                    <div className="review-meta">
                      <span className="review-stars">{'⭐'.repeat(review.stars)}</span>
                      <span className="review-date">
                        {new Date(review.created_at).toLocaleDateString()}
                      </span>
                      <span className="review-useful">👍 {review.useful || 0}</span>
                    </div>
                  </div>
                  <p className="review-text">{review.text}</p>
                </div>
              ))}
            </div>
            {loadingMore && (
              <div className="loading-more">
                <p>리뷰를 불러오는 중...</p>
              </div>
            )}
            {hasMore && !loadingMore && (
              <div className="load-more-container">
                <button 
                  className="btn-load-more"
                  onClick={() => loadReviews(reviewSkip, false)}
                >
                  더보기
                </button>
              </div>
            )}
            {!hasMore && reviews.length > 0 && (
              <div className="no-more-reviews">
                <p>모든 리뷰를 불러왔습니다</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default UserProfilePage;

