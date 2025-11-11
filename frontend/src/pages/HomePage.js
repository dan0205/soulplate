/**
 * 홈 페이지 - 개인화 추천 표시
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { businessAPI } from '../services/api';
import './Home.css';

const HomePage = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const itemsPerPage = 20;
  
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    loadRecommendations();
  }, [currentPage]);

  const loadRecommendations = async () => {
    setLoading(true);
    setError('');
    
    try {
      // 비즈니스 목록 가져오기
      const skip = (currentPage - 1) * itemsPerPage;
      const response = await businessAPI.list({ skip, limit: itemsPerPage });
      
      // 응답 구조 확인: response.data가 { businesses, total, skip, limit } 형태
      const { businesses, total } = response.data;
      
      // 총 페이지 수 계산
      setTotalPages(Math.ceil(total / itemsPerPage));
      
      // 각 비즈니스를 표시 형식으로 변환
      const businessesWithPredictions = businesses.map(business => ({
        business: business,
        score: null,  // 추천 점수는 없음
        prediction: business.ai_prediction || null  // AI 예측
      }));
      
      setRecommendations(businessesWithPredictions);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load businesses');
      console.error('Error loading businesses:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (page) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const renderPageNumbers = () => {
    const pages = [];
    const maxVisible = 5;
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    
    // 조정: 끝에서 시작할 때
    if (endPage - startPage < maxVisible - 1) {
      startPage = Math.max(1, endPage - maxVisible + 1);
    }
    
    // 첫 페이지
    if (startPage > 1) {
      pages.push(
        <button
          key={1}
          className="pagination-button"
          onClick={() => handlePageChange(1)}
        >
          1
        </button>
      );
      if (startPage > 2) {
        pages.push(<span key="dots-start" className="pagination-dots">...</span>);
      }
    }
    
    // 페이지 번호들
    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <button
          key={i}
          className={`pagination-button ${i === currentPage ? 'active' : ''}`}
          onClick={() => handlePageChange(i)}
        >
          {i}
        </button>
      );
    }
    
    // 마지막 페이지
    if (endPage < totalPages) {
      if (endPage < totalPages - 1) {
        pages.push(<span key="dots-end" className="pagination-dots">...</span>);
      }
      pages.push(
        <button
          key={totalPages}
          className="pagination-button"
          onClick={() => handlePageChange(totalPages)}
        >
          {totalPages}
        </button>
      );
    }
    
    return pages;
  };

  const handleBusinessClick = (businessId) => {
    navigate(`/business/${businessId}`);
  };

  return (
    <div className="home-container">
      <header className="home-header">
        <h1>🚀 Two-Tower Recommendations</h1>
        <div className="user-info">
          <span>Welcome, {user?.username}!</span>
          <button onClick={logout} className="btn-logout">Logout</button>
        </div>
      </header>

      <main className="home-main">
        <div className="recommendations-header">
          <h2>🏪 Restaurant List</h2>
          <button onClick={loadRecommendations} className="btn-refresh" disabled={loading}>
            {loading ? 'Loading...' : '🔄 Refresh'}
          </button>
        </div>

        {error && (
          <div className="error-banner">
            {error}
            <button onClick={loadRecommendations}>Retry</button>
          </div>
        )}

        {loading ? (
          <div className="loading-container">
            <div className="spinner"></div>
            <p>Loading your personalized recommendations...</p>
          </div>
        ) : (
          <>
            <div className="recommendations-grid">
              {recommendations.length === 0 ? (
                <div className="no-results">
                  <p>No businesses available.</p>
                </div>
              ) : (
                recommendations.map((item, index) => (
                  <div
                    key={item.business.business_id}
                    className="business-card"
                    onClick={() => handleBusinessClick(item.business.business_id)}
                  >
                    <div className="card-rank">#{(currentPage - 1) * itemsPerPage + index + 1}</div>
                    <h3>{item.business.name}</h3>
                    <div className="card-info">
                      <span className="stars">⭐ {item.business.stars?.toFixed(1) || 'N/A'}</span>
                      <span className="reviews">📝 {item.business.review_count} reviews</span>
                    </div>
                    {item.business.ai_prediction && (
                      <div className="ai-prediction-inline">
                        AI 예상: {item.business.ai_prediction.deepfm_rating?.toFixed(1)} (DeepFM) / {item.business.ai_prediction.multitower_rating?.toFixed(1) || 'N/A'} (Multi-Tower)
                      </div>
                    )}
                    <p className="categories">{item.business.categories || 'No category'}</p>
                    <p className="location">📍 {item.business.city}, {item.business.state}</p>
                    {item.business.top_features && item.business.top_features.length > 0 && (
                      <div className="top-features">
                        {item.business.top_features.slice(0, 3).map((feature, idx) => (
                          <span key={idx} className="feature-tag">
                            {feature.aspect} ({Math.round(feature.score * 100)}%)
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
            
            {totalPages > 1 && (
              <div className="pagination-container">
                <button
                  className="pagination-button"
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                >
                  ← 이전
                </button>
                
                {renderPageNumbers()}
                
                <button
                  className="pagination-button"
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages}
                >
                  다음 →
                </button>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
};

export default HomePage;

