/**
 * 로그인 페이지 - Google OAuth 전용
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import GoogleLoginButton from '../components/GoogleLoginButton';
import { authAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import './Auth.css';

const LoginPage = () => {
  const navigate = useNavigate();
  const { handleOAuthCallback } = useAuth();
  const [isLoading, setIsLoading] = useState(false);

  const handleBrowseDemo = async () => {
    if (isLoading) return;
    setIsLoading(true);
    
    try {
      const response = await authAPI.browseDemoLogin();
      await handleOAuthCallback(response.data.access_token);
      toast.success('둘러보기 모드로 시작합니다!');
      navigate('/', { replace: true });
    } catch (error) {
      console.error('Browse demo login failed:', error);
      toast.error('둘러보기 시작에 실패했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

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

        <button 
          className="browse-demo-button"
          onClick={handleBrowseDemo}
          disabled={isLoading}
        >
          <span>👀</span>
          <span>{isLoading ? '로딩 중...' : '둘러보기'}</span>
        </button>

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
