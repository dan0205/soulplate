/**
 * 스플래시 화면 컴포넌트
 * 세션별로 표시되는 스플래시 화면 (탭을 닫았다가 다시 열 때마다 표시)
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import './SplashScreen.css';

const SplashScreen = ({ onComplete }) => {
  const { loading } = useAuth();
  const [isFadingOut, setIsFadingOut] = useState(false);
  const [minTimeElapsed, setMinTimeElapsed] = useState(false);
  const [progress, setProgress] = useState(0);

  // 진행률 업데이트: 2초 동안 0% → 100%
  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 5; // 100ms마다 5%씩 증가 (2초 / 20 = 100ms)
      });
    }, 100);

    return () => clearInterval(interval);
  }, []);

  // 최소 표시 시간 2초 타이머
  useEffect(() => {
    const minTimeTimer = setTimeout(() => {
      setMinTimeElapsed(true);
    }, 2000);

    return () => {
      clearTimeout(minTimeTimer);
    };
  }, []);

  // 진행률이 100%에 도달하지 않으면 강제로 100% 설정 (타임아웃)
  useEffect(() => {
    const progressTimeout = setTimeout(() => {
      setProgress(prev => {
        if (prev < 100) {
          return 100; // 강제로 100% 설정
        }
        return prev;
      });
    }, 2500); // 2.5초 후 강제 완료 (진행률 업데이트가 2초이므로 여유 시간)

    return () => {
      clearTimeout(progressTimeout);
    };
  }, []);

  // 페이드아웃 조건 체크: 진행률 100% + 최소 시간 경과 + 앱 초기화 완료
  useEffect(() => {
    if (progress >= 100 && minTimeElapsed && !loading) {
      // 페이드아웃 시작
      setIsFadingOut(true);
      
      // 페이드아웃 애니메이션 완료 후 제거
      const fadeOutTimer = setTimeout(() => {
        sessionStorage.setItem('soulplate_first_visit', 'false');
        if (onComplete) {
          onComplete();
        }
      }, 500); // 페이드아웃 애니메이션 시간

      return () => {
        clearTimeout(fadeOutTimer);
      };
    }
  }, [progress, minTimeElapsed, loading, onComplete]);

  // 앱 초기화가 너무 오래 걸리는 경우 타임아웃 (5초)
  useEffect(() => {
    const timeoutTimer = setTimeout(() => {
      if (loading) {
        // 5초 후에도 로딩 중이면 강제로 페이드아웃
        setIsFadingOut(true);
        setTimeout(() => {
          sessionStorage.setItem('soulplate_first_visit', 'false');
          if (onComplete) {
            onComplete();
          }
        }, 500);
      }
    }, 5000);

    return () => {
      clearTimeout(timeoutTimer);
    };
  }, [loading, onComplete]);

  return (
    <div 
      className={`splash-container ${isFadingOut ? 'fade-out' : ''}`}
    >
      <div className="splash-content">
        <h1 className="splash-title">SoulPlate</h1>
        <p className="splash-subtitle">AI 맛집 추천</p>
      </div>
      <div className="progress-container">
        <div 
          className="progress-bar"
          style={{ width: `${progress}%` }}
        ></div>
      </div>
    </div>
  );
};

export default SplashScreen;

