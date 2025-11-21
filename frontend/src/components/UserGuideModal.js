import React, { useEffect } from 'react';
import ReactDOM from 'react-dom';
import './UserGuideModal.css';

function UserGuideModal({ onClose }) {
  // ESC 키로 모달 닫기
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [onClose]);

  const handleConfirm = () => {
    sessionStorage.setItem('user_guide_seen', 'true');
    onClose();
  };

  const handleClose = () => {
    sessionStorage.setItem('user_guide_seen', 'true');
    onClose();
  };

  return ReactDOM.createPortal(
    <div className="user-guide-modal-overlay">
      <div className="user-guide-modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="user-guide-modal-close" onClick={handleClose}>×</button>
        
        <div className="user-guide-modal-header">
          <h2>SoulPlate 사용법</h2>
          <p>서비스를 더 잘 활용하기 위한 간단한 가이드입니다</p>
        </div>

        <div className="user-guide-list">
          <div className="user-guide-item">
            <div className="user-guide-icon">🗺️</div>
            <div className="user-guide-content">
              <h3>지도에서 맛집 찾기</h3>
              <p>지도를 드래그하거나 확대/축소하여 원하는 지역의 맛집을 탐색할 수 있습니다. 마커를 클릭하면 상세 정보를 확인할 수 있어요.</p>
            </div>
          </div>

          <div className="user-guide-item">
            <div className="user-guide-icon">🤖</div>
            <div className="user-guide-content">
              <h3>AI 맞춤 추천</h3>
              <p>취향 테스트를 완료하면 AI가 당신의 취향을 분석하여 개인화된 맛집 추천을 제공합니다. 더 정확한 추천을 위해 심화 테스트도 추천해요!</p>
            </div>
          </div>

          <div className="user-guide-item">
            <div className="user-guide-icon">👤</div>
            <div className="user-guide-content">
              <h3>마이페이지</h3>
              <p>우측 상단의 프로필 버튼을 클릭하면 마이페이지로 이동할 수 있습니다. 마이페이지에서는 내가 작성한 리뷰 확인, 취향 테스트 결과 확인, 프로필 정보 수정 등을 할 수 있어요.</p>
            </div>
          </div>

          <div className="user-guide-item">
            <div className="user-guide-icon">📋</div>
            <div className="user-guide-content">
              <h3>하단 카드</h3>
              <p>화면 하단에 있는 카드를 위로 드래그하면 맛집 목록을 확인할 수 있습니다. 아래로 드래그하면 카드를 접어 지도를 전체적으로 볼 수 있고, 카드를 클릭하면 상세 정보를 확인할 수 있어요.</p>
            </div>
          </div>
        </div>

        <button className="user-guide-btn-confirm" onClick={handleConfirm}>
          확인했습니다
        </button>

        <p className="user-guide-modal-hint">💡 이 가이드는 세션당 한 번만 표시됩니다</p>
      </div>
    </div>,
    document.body
  );
}

export default UserGuideModal;

