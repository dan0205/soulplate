/**
 * 로그인 페이지 - Google OAuth 전용
 */

import React from 'react';
import GoogleLoginButton from '../components/GoogleLoginButton';
import './Auth.css';

const LoginPage = () => {
  return (
    <div className="auth-container">
      <div className="auth-box">
        <h1>SoulPlate</h1>
        <p style={{ textAlign: 'center', marginBottom: '30px', color: '#666' }}>
          Google 계정으로 로그인하세요
        </p>
        
        <GoogleLoginButton />
      </div>
    </div>
  );
};

export default LoginPage;

