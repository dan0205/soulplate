import React from 'react';
import ReactDOM from 'react-dom';
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
    sessionStorage.setItem('taste_test_skipped', 'true');
    onClose();
  };

  return ReactDOM.createPortal(
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        
        <div className="modal-header">
          <h2>잠깐만요!</h2>
          <p>음식 취향 테스트를 하시면<br/>맞춤 맛집 추천을 받을 수 있어요!</p>
        </div>

        <div className="modal-body">
          <div className="test-option" onClick={handleQuickTest}>
            <div className="test-icon">⚡</div>
            <div className="test-info">
              <h3>간단 테스트</h3>
              <p>빠르게 음식 취향을 파악하고 싶다면 간단 테스트를 추천해요. 8개의 핵심 질문으로 당신의 취향을 분석합니다.</p>
            </div>
          </div>

          <div className="test-option" onClick={handleDeepTest}>
            <div className="test-icon">🔍</div>
            <div className="test-info">
              <h3>심화 테스트</h3>
              <p>더 정확하고 상세한 맞춤 추천을 원한다면 심화 테스트를 선택하세요. 20개의 질문으로 깊이 있는 분석을 제공합니다.</p>
            </div>
          </div>
        </div>

        <button className="btn-skip-modal" onClick={handleSkip}>
          나중에 하기 →
        </button>

        <p className="modal-hint">💡 나중에 언제든 다시 할 수 있어요</p>
      </div>
    </div>,
    document.body
  );
}

export default TasteTestModal;












