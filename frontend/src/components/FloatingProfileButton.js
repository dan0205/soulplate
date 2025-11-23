import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ConfirmModal from './ConfirmModal';
import TasteTestModal from './TasteTestModal';
import SettingsModal from './SettingsModal';
import { useAuth } from '../context/AuthContext';
import './FloatingProfileButton.css';

const FloatingProfileButton = ({ username, onLogout }) => {
  const [showMenu, setShowMenu] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const [showTasteTestModal, setShowTasteTestModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const menuRef = useRef(null);
  const navigate = useNavigate();
  const { user, loadUser } = useAuth();

  // 외부 클릭 시 메뉴 닫기
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setShowMenu(false);
      }
    };

    if (showMenu) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('touchstart', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleClickOutside);
    };
  }, [showMenu]);

  const handleProfileClick = () => {
    setShowMenu(!showMenu);
  };

  const handleMyProfile = () => {
    setShowMenu(false);
    navigate('/my-profile');
  };

  const handleRecentReviews = () => {
    setShowMenu(false);
    navigate('/recent-reviews');
  };

  const handleTasteTest = () => {
    setShowMenu(false);
    setShowTasteTestModal(true);
  };

  const handleSettings = () => {
    setShowMenu(false);
    setShowSettingsModal(true);
  };

  const handleLogout = () => {
    setShowMenu(false);
    setShowLogoutConfirm(true);
  };

  const handleSettingsUpdate = async () => {
    // 사용자 정보 새로고침
    await loadUser();
  };

  const handleLogoutConfirm = () => {
    setShowLogoutConfirm(false);
    onLogout();
  };

  // 첫 글자만 표시
  const initial = username ? username.charAt(0).toUpperCase() : 'U';

  return (
    <div className="floating-profile-container" ref={menuRef}>
      <button 
        className="floating-profile-btn"
        onClick={handleProfileClick}
        aria-label="프로필 메뉴"
      >
        <span className="profile-initial">{initial}</span>
      </button>

      {showMenu && (
        <div className="profile-menu-popup">
          <div className="profile-menu-header">
            <div className="profile-menu-username">{username}</div>
          </div>
          <div className="profile-menu-divider" />
          <button 
            className="profile-menu-item"
            onClick={handleMyProfile}
          >
            <span className="menu-icon">👤</span>
            <span>내 프로필</span>
          </button>
          <button 
            className="profile-menu-item taste-test"
            onClick={handleTasteTest}
          >
            <span className="menu-icon">🍽️</span>
            <span>취향 테스트</span>
          </button>
          <button 
            className="profile-menu-item"
            onClick={handleRecentReviews}
          >
            <span className="menu-icon">📝</span>
            <span>최근 리뷰</span>
          </button>
          <button 
            className="profile-menu-item"
            onClick={handleSettings}
          >
            <span className="menu-icon">⚙️</span>
            <span>설정</span>
          </button>
          <button 
            className="profile-menu-item logout"
            onClick={handleLogout}
          >
            <span className="menu-icon">🚪</span>
            <span>로그아웃</span>
          </button>
        </div>
      )}

      {/* 로그아웃 확인 모달 */}
      <ConfirmModal
        isOpen={showLogoutConfirm}
        title="로그아웃 하시겠습니까?"
        message="다시 로그인하시면 계속 이용하실 수 있습니다."
        confirmText="로그아웃"
        cancelText="취소"
        variant="confirm"
        onConfirm={handleLogoutConfirm}
        onCancel={() => setShowLogoutConfirm(false)}
      />

      {/* 취향 테스트 모달 */}
      {showTasteTestModal && (
        <TasteTestModal onClose={() => setShowTasteTestModal(false)} />
      )}

      {/* 설정 모달 */}
      <SettingsModal
        isOpen={showSettingsModal}
        onClose={() => setShowSettingsModal(false)}
        currentUser={user}
        onUpdateSuccess={handleSettingsUpdate}
      />
    </div>
  );
};

export default FloatingProfileButton;

