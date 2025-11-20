import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './TasteTestResult.css';

function TasteTestResultPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { result, testType } = location.state || {};

  if (!result) {
    return (
      <div className="result-container">
        <div className="error">결과를 찾을 수 없습니다.</div>
        <button onClick={() => navigate('/')}>홈으로</button>
      </div>
    );
  }

  const isQuickTest = testType === 'quick';

  return (
    <div className="result-container">
      <div className="result-card">
        <div className="result-header">
          <div className="result-badge">
            {isQuickTest ? '⚡ 간단 테스트 결과' : '🔍 심화 테스트 결과'}
          </div>
          <h1 className="mbti-type">{result.mbti_type}</h1>
          <h2 className="type-name">{result.type_name}</h2>
        </div>

        <div className="result-body">
          <div className="accuracy-section">
            {isQuickTest ? (
              <>
                <div className="accuracy-stars">★★★☆☆</div>
                <p className="accuracy-text">정확도: 중간</p>
                <p className="accuracy-hint">
                  💡 심화 테스트로 더 정확한 분석을 받아보세요!
                </p>
              </>
            ) : (
              <>
                <div className="accuracy-stars">★★★★★</div>
                <p className="accuracy-text">정확도: 높음</p>
              </>
            )}
          </div>

          <div className="description-section">
            <h3>📝 당신의 음식 취향</h3>
            <p className="description-text">{result.description}</p>
          </div>

          {result.recommendations && result.recommendations.length > 0 && (
            <div className="recommendations-section">
              <h3>🎯 추천 스타일</h3>
              <ul className="recommendations-list">
                {result.recommendations.map((rec, index) => (
                  <li key={index} className="recommendation-item">
                    <span className="recommendation-icon">✓</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="result-actions">
          {isQuickTest && (
            <button 
              className="btn-upgrade"
              onClick={() => navigate('/taste-test', { state: { testType: 'deep' } })}
            >
              🔍 심화 테스트로 업그레이드
            </button>
          )}
          
          <button 
            className="btn-home"
            onClick={() => navigate('/')}
          >
            🏠 홈으로 가기
          </button>

          <button 
            className="btn-retake"
            onClick={() => navigate('/taste-test', { state: { testType } })}
          >
            🔄 다시 하기
          </button>
        </div>
      </div>

      <div className="result-hint">
        <p>💡 리뷰를 작성하면 취향 분석이 더 정확해져요!</p>
      </div>
    </div>
  );
}

export default TasteTestResultPage;














