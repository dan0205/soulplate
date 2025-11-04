/**
 * 비즈니스 상세 페이지
 */

import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { businessAPI } from '../services/api';
import './BusinessDetail.css';

const BusinessDetailPage = () => {
  const { businessId } = useParams();
  const navigate = useNavigate();
  
  const [business, setBusiness] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  const [reviewForm, setReviewForm] = useState({
    stars: 5,
    text: ''
  });
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    loadBusinessDetails();
    loadReviews();
  }, [businessId]);

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

  const loadReviews = async () => {
    try {
      const response = await businessAPI.getReviews(businessId);
      setReviews(response.data);
    } catch (err) {
      console.error('Failed to load reviews:', err);
    }
  };

  const handleSubmitReview = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    
    try {
      await businessAPI.createReview(businessId, reviewForm);
      alert('Review submitted successfully! 🎉');
      setReviewForm({ stars: 5, text: '' });
      loadReviews();
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
  }

  if (error || !business) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error || 'Business not found'}</p>
        <button onClick={() => navigate('/')}>Go Back</button>
      </div>
    );
  }

  return (
    <div className="business-detail-container">
      <button className="btn-back" onClick={() => navigate('/')}>← Back to Home</button>
      
      <div className="business-header">
        <h1>{business.name}</h1>
        <div className="business-info">
          <span className="stars">⭐ {business.stars?.toFixed(1) || 'N/A'}</span>
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
        <h2>Recent Reviews ({reviews.length})</h2>
        {reviews.length === 0 ? (
          <p className="no-reviews">No reviews yet. Be the first to review!</p>
        ) : (
          <div className="reviews-list">
            {reviews.map((review) => (
              <div key={review.id} className="review-item">
                <div className="review-header">
                  <span className="review-stars">
                    {'⭐'.repeat(review.stars)}
                  </span>
                  <span className="review-date">
                    {new Date(review.created_at).toLocaleDateString()}
                  </span>
                </div>
                <p className="review-text">{review.text}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default BusinessDetailPage;

