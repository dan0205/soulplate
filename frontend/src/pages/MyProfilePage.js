/**
 * 내 프로필 페이지
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { userAPI, tasteTestAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import Avatar from '../components/Avatar';
import { getMBTIInfo } from '../utils/mbtiDescriptions';
import ConfirmModal from '../components/ConfirmModal';
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
  const [showTestOptions, setShowTestOptions] = useState(false);
  const [showDeleteTestConfirm, setShowDeleteTestConfirm] = useState(false);

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

  const getTopABSAFeatures = (absaFeatures) => {
    if (!absaFeatures) return [];
    
    const features = Object.entries(absaFeatures).map(([key, value]) => ({
      key,
      value
    }));
    
    features.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    return features.slice(0, 5);
  };

  const handleStartTest = (testType) => {
    setShowTestOptions(false);
    navigate('/taste-test', { state: { testType } });
  };

  const handleDeleteTest = () => {
    setShowDeleteTestConfirm(true);
    setShowTestOptions(false);
  };

  const handleDeleteTestConfirm = async () => {
    try {
      await tasteTestAPI.delete();
      toast.dismiss();
      toast.success('취향 테스트 결과가 삭제되었습니다.');
      setShowDeleteTestConfirm(false);
      loadProfile();
      loadReviews(0, true);
    } catch (err) {
      console.error('취향 테스트 삭제 실패:', err);
      toast.dismiss();
      toast.error('삭제에 실패했습니다.');
      setShowDeleteTestConfirm(false);
    }
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
          <div className="taste-test-card">
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
        </div>
      )}

      <div className="taste-test-section">
        <h2>🍽️ 음식 취향 테스트</h2>
        <div className="taste-test-card">
          {profile.taste_test_completed ? (
            <>
              <div className="test-completed-badge">
                ✅ 취향 테스트 완료
              </div>
              {profile.review_count === 0 && (
                <p className="taste-test-hint">
                  💡 실제 리뷰를 작성하면 추천이 더 정확해져요!
                </p>
              )}
              <button 
                className="btn-retest"
                onClick={() => setShowTestOptions(!showTestOptions)}
              >
                🔄 재테스트하기
              </button>
            </>
          ) : (
            <>
              {profile.review_count === 0 ? (
                <p className="taste-test-desc">
                  아직 리뷰가 없으시네요! 취향 테스트로 시작해보세요.
                </p>
              ) : (
                <p className="taste-test-desc">
                  취향 테스트로 더 정확한 맛집 추천을 받아보세요!
                </p>
              )}
              <button 
                className="btn-start-test"
                onClick={() => setShowTestOptions(!showTestOptions)}
              >
                테스트 시작하기
              </button>
            </>
          )}
          
          {showTestOptions && (
            <div className="test-options">
              <button 
                className="test-option-btn quick"
                onClick={() => handleStartTest('quick')}
              >
                ⚡ 간단 테스트 (8문항, ~1분)
              </button>
              <button 
                className="test-option-btn deep"
                onClick={() => handleStartTest('deep')}
              >
                🔍 심화 테스트 (20문항, ~3-4분)
              </button>
              {profile.taste_test_completed && (
                <button 
                  className="test-option-btn delete"
                  onClick={handleDeleteTest}
                >
                  ❌ 기존 테스트 삭제
                </button>
              )}
            </div>
          )}
        </div>
      </div>

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
                <p>리뷰를 불러오는 중...</p>
              </div>
            )}
            {hasMore && !loadingMore && (
              <div style={{ textAlign: 'center', padding: '16px 0' }}>
                <button 
                  className="review-load-more-link"
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

      {/* 삭제 확인 모달 */}
      <ConfirmModal
        isOpen={showDeleteTestConfirm}
        title="기존 취향 테스트 결과를 삭제하시겠습니까?"
        message="삭제된 테스트 결과는 복구할 수 없습니다."
        confirmText="삭제"
        cancelText="취소"
        variant="danger"
        onConfirm={handleDeleteTestConfirm}
        onCancel={() => setShowDeleteTestConfirm(false)}
      />
    </div>
  );
};

export default MyProfilePage;

