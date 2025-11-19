/**
 * 스플래시 화면 컴포넌트
 * 세션별로 표시되는 스플래시 화면 (탭을 닫았다가 다시 열 때마다 표시)
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import './SplashScreen.css';

const SplashScreen = () => {
  const { loading } = useAuth();
  const [showSplash, setShowSplash] = useState(false);
  const [isFadingOut, setIsFadingOut] = useState(false);
  const [minTimeElapsed, setMinTimeElapsed] = useState(false);

  // 세션별 방문 감지 및 스플래시 표시 시작
  useEffect(() => {
    const hasVisited = sessionStorage.getItem('soulplate_first_visit');
    
    if (!hasVisited) {
      // 첫 방문이면 스플래시 표시
      setShowSplash(true);
      
      // 최소 표시 시간 2초 타이머
      const minTimeTimer = setTimeout(() => {
        setMinTimeElapsed(true);
      }, 2000);

      return () => {
        clearTimeout(minTimeTimer);
      };
    }
  }, []);

  // 페이드아웃 조건 체크: 최소 시간 경과 + 앱 초기화 완료
  useEffect(() => {
    if (showSplash && minTimeElapsed && !loading) {
      // 페이드아웃 시작
      setIsFadingOut(true);
      
      // 페이드아웃 애니메이션 완료 후 제거
      const fadeOutTimer = setTimeout(() => {
        setShowSplash(false);
        sessionStorage.setItem('soulplate_first_visit', 'false');
      }, 500); // 페이드아웃 애니메이션 시간

      return () => {
        clearTimeout(fadeOutTimer);
      };
    }
  }, [showSplash, minTimeElapsed, loading]);

  // 앱 초기화가 너무 오래 걸리는 경우 타임아웃 (5초)
  useEffect(() => {
    if (showSplash) {
      const timeoutTimer = setTimeout(() => {
        if (loading) {
          // 5초 후에도 로딩 중이면 강제로 페이드아웃
          setIsFadingOut(true);
          setTimeout(() => {
            setShowSplash(false);
            sessionStorage.setItem('soulplate_first_visit', 'false');
          }, 500);
        }
      }, 5000);

      return () => {
        clearTimeout(timeoutTimer);
      };
    }
  }, [showSplash, loading]);

  if (!showSplash) {
    return null;
  }

  return (
    <div 
      className={`splash-container ${isFadingOut ? 'fade-out' : ''}`}
    >
      <div className="splash-content">
        <h1 className="splash-title">SoulPlate</h1>
        <p className="splash-subtitle">AI 맛집 추천</p>
      </div>
      <div className="progress-container">
        <div className="progress-bar"></div>
      </div>
    </div>
  );
};

export default SplashScreen;

