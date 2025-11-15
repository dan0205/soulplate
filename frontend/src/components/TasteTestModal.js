import React from 'react';
import { useNavigate } from 'react-router-dom';
import './TasteTestModal.css';

function TasteTestModal({ onClose }) {
  const navigate = useNavigate();

  const handleQuickTest = () => {
    navigate('/taste-test', { state: { testType: 'quick' } });
    onClose();
  };

  const handleDeepTest = () => {
    navigate('/taste-test', { state: { testType: 'deep' } });
    onClose();
  };

  const handleSkip = () => {
    localStorage.setItem('taste_test_skipped', 'true');
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        
        <div className="modal-header">
          <h2>🎯 잠깐만요!</h2>
          <p>음식 취향 테스트를 하시면<br/>맞춤 맛집 추천을 받을 수 있어요!</p>
        </div>

        <div className="modal-body">
          <div className="test-option" onClick={handleQuickTest}>
            <div className="test-icon">⚡</div>
            <div className="test-info">
              <h3>간단 테스트</h3>
              <p>8개 질문 · 약 1분</p>
              <span className="test-badge">빠르게 확인</span>
            </div>
          </div>

          <div className="test-option" onClick={handleDeepTest}>
            <div className="test-icon">🔍</div>
            <div className="test-info">
              <h3>심화 테스트</h3>
              <p>20개 질문 · 약 3-4분</p>
              <span className="test-badge">정확한 분석</span>
            </div>
          </div>
        </div>

        <button className="btn-skip-modal" onClick={handleSkip}>
          나중에 하기 →
        </button>

        <p className="modal-hint">💡 나중에 언제든 다시 할 수 있어요</p>
      </div>
    </div>
  );
}

export default TasteTestModal;




