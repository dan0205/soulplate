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
        <div className="logo-container">
          <h1 className="logo">SoulPlate</h1>
        </div>

        <div className="tagline">당신의 음식 취향을 발견하세요</div>
        <div className="description">
          AI가 분석한 맞춤형 맛집 추천,<br />
          지금 바로 시작해보세요
        </div>

        <GoogleLoginButton />

        <div className="features">
          <div className="features-title">SoulPlate의 특별함</div>
          <div className="feature-list">
            <div className="feature-item">
              <div className="feature-icon"></div>
              <span>AI 기반 맞춤형 맛집 추천</span>
            </div>
            <div className="feature-item">
              <div className="feature-icon"></div>
              <span>당신만의 음식 취향 MBTI 분석</span>
            </div>
            <div className="feature-item">
              <div className="feature-icon"></div>
              <span>실시간 맛집 리뷰 및 평가</span>
            </div>
          </div>
        </div>

        <div className="footer">
          <div className="footer-links">
            <a href="#">이용약관</a>
            <a href="#">개인정보처리방침</a>
            <a href="#">문의하기</a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
