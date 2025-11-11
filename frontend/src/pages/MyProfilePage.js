/**
 * 내 프로필 페이지
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { userAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import Avatar from '../components/Avatar';
import './Profile.css';

const MyProfilePage = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  
  const [profile, setProfile] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [reviewSkip, setReviewSkip] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);

  useEffect(() => {
    loadProfile();
    loadReviews(0, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadProfile = async () => {
    try {
      const response = await userAPI.getMyProfile();
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
      const response = await userAPI.getUserReviews(user.id, { skip, limit });
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
  }, [user, loadingMore]);

  const handleScroll = useCallback(() => {
    if (loadingMore || !hasMore) return;
    
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;
    
    if (scrollTop + windowHeight >= documentHeight - 200) {
      loadReviews(reviewSkip, false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reviewSkip, hasMore, loadingMore]);

  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

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
        <button onClick={() => navigate('/')}>Go Back</button>
      </div>
    );
  }

  const topFeatures = getTopABSAFeatures(profile.absa_features);

  return (
    <div className="profile-container">
      <div className="profile-header-actions">
        <button className="btn-back" onClick={() => navigate('/')}>← Back to Home</button>
        <button className="btn-logout" onClick={logout}>Logout</button>
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

      {topFeatures.length > 0 && (
        <div className="absa-section">
          <h2>내 리뷰 성향 분석</h2>
          <div className="absa-tags">
            {topFeatures.map((feature, idx) => (
              <span key={idx} className="absa-tag">
                {feature.key.replace('_', ' ')}: {(Math.abs(feature.value) * 100).toFixed(0)}%
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="reviews-section">
        <h2>내가 작성한 리뷰 ({reviews.length})</h2>
        {reviews.length === 0 && !loadingMore ? (
          <p className="no-reviews">아직 작성한 리뷰가 없습니다.</p>
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
                <p>Loading more reviews...</p>
              </div>
            )}
            {!hasMore && reviews.length > 0 && (
              <div className="no-more-reviews">
                <p>No more reviews to load</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default MyProfilePage;

