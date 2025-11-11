/**
 * 사용자 아바타 컴포넌트
 */

import React from 'react';
import './Avatar.css';

const Avatar = ({ username, size = 'medium' }) => {
  // 사용자 이름의 첫 글자를 대문자로
  const initial = username ? username.charAt(0).toUpperCase() : '?';
  
  return (
    <div className={`avatar avatar-${size}`}>
      <span className="avatar-initial">{initial}</span>
    </div>
  );
};

export default Avatar;

