/**
 * AI 예측 별점 컴포넌트
 * DeepFM, Multi-Tower, 앙상블 예측 표시
 */

import React from 'react';
import './AIPrediction.css';

const AIPrediction = ({ prediction }) => {
  if (!prediction) return null;

  return (
    <div className="ai-prediction">
      <h3>🤖 AI 예상 별점</h3>
      <div className="predictions-grid">
        <div className="prediction-item">
          <span className="model-name">DeepFM:</span>
          <span className="rating">⭐ {prediction.deepfm_rating}</span>
        </div>
        <div className="prediction-item">
          <span className="model-name">Multi-Tower:</span>
          <span className="rating">⭐ {prediction.multitower_rating}</span>
        </div>
      </div>
    </div>
  );
};

export default AIPrediction;

