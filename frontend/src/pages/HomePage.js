/**
 * 홈 페이지 - 개인화 추천 표시
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { recommendationAPI } from '../services/api';
import './Home.css';

const HomePage = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    loadRecommendations();
  }, []);

  const loadRecommendations = async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await recommendationAPI.get({ top_k: 10 });
      setRecommendations(response.data.recommendations);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load recommendations');
      console.error('Error loading recommendations:', err);
    } finally {
      setLoading(false);
    }
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
          <h2>✨ Personalized Recommendations for You</h2>
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
          <div className="recommendations-grid">
            {recommendations.length === 0 ? (
              <div className="no-results">
                <p>No recommendations available.</p>
              </div>
            ) : (
              recommendations.map((item, index) => (
                <div
                  key={item.business.business_id}
                  className="business-card"
                  onClick={() => handleBusinessClick(item.business.business_id)}
                >
                  <div className="card-rank">#{index + 1}</div>
                  <h3>{item.business.name}</h3>
                  <div className="card-info">
                    <span className="stars">⭐ {item.business.stars?.toFixed(1) || 'N/A'}</span>
                    <span className="reviews">📝 {item.business.review_count} reviews</span>
                  </div>
                  <p className="categories">{item.business.categories || 'No category'}</p>
                  <p className="location">📍 {item.business.city}, {item.business.state}</p>
                  <div className="score">
                    <small>Match Score: {(item.score * 100).toFixed(1)}%</small>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default HomePage;

