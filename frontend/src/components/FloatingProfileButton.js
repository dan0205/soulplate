import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './FloatingProfileButton.css';

const FloatingProfileButton = ({ username, onLogout }) => {
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef(null);
  const navigate = useNavigate();

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

  const handleLogout = () => {
    setShowMenu(false);
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
            className="profile-menu-item logout"
            onClick={handleLogout}
          >
            <span className="menu-icon">🚪</span>
            <span>로그아웃</span>
          </button>
        </div>
      )}
    </div>
  );
};

export default FloatingProfileButton;

