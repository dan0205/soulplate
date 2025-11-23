/**
 * 인증 Context
 */
// PrivateRout 컴포넌트의 핵심 두뇌 역할을 하는 AuthContext 
// React의 Context API를 사용해서, 어플리케이션 전체에서 로그인 상태, 사용자 정보,
// 로그인/로그아웃 기능을 공유할 수 있게 해주는 전역 상태 관리자이다 

import React, { createContext, useState, useContext, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { authAPI } from '../services/api';

const AuthContext = createContext(null);
// 인증과 관련된 모든 정보를 담을 수 있는 전역 상자를 만든다 
export const AuthProvider = ({ children }) => {
  // AuthContext 상자에 실제 값들을 채워서 앱 전체(children)에 제공하는 컴포넌트 
  // 이 AuthProvider 컴포넌트로 App.js에서 전체 앱을 감싸주면, 앱 내의 모든 컴포넌트가
  // 이 상자 안의 내용물을 꺼내쓸수있다 
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  // 현재 로그인한 사용자의 정보를 저장 = 객체 있으면 로그인, null이면 로그아웃 
  const [loading, setLoading] = useState(true);
  // true이면 현재 로그인 상태인지 확인 중 

  useEffect(() => {
    // AuthProvider가 처음 렌더링될 때, 단 한번 실행된다 
    const token = localStorage.getItem('access_token');
    if (token) {
      loadUser();
      // 토큰이 있을 때, loadUser 함수를 호출해서 이 토큰이 아직 유효한지 확인을 요청한다 
    } else {
      setLoading(false);
    }
  }, []);

  const loadUser = async () => {
    //토큰이 있을 때 실행되는 loadUser 
    try {
      const response = await authAPI.getMe();
      setUser(response.data);
      // 백엔드에서 사용자 정보를 받아와서 user 상태를 업데이트한다 
    } catch (error) {
      console.error('Failed to load user:', error);
      localStorage.removeItem('access_token');
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    setUser(null);
    navigate('/login', { replace: true });
  };

  const handleOAuthCallback = async (token) => {
    try {
      localStorage.setItem('access_token', token);
      const response = await authAPI.getMe();
      setUser(response.data);
      
      // profile_completed 체크
      if (!response.data.profile_completed) {
        navigate('/onboarding', { replace: true });
      } else {
        navigate('/', { replace: true });
      }
    } catch (error) {
      console.error('OAuth callback error:', error);
      localStorage.removeItem('access_token');
      throw error;
    }
  };

  const value = {
    user,
    loading,
    logout,
    handleOAuthCallback,
    loadUser, // 사용자 정보 재로드 함수 추가
    isAuthenticated: !!user,
  }; // 위에서 만든 모든 상태와 함수를 value라는 객체 하나로 묶는다 
  // isAuthenticated: !!user = user 객체가 있으면 true, 없으면 false 
  // 이 value 객체가 AuthContext.Provider를 통해 앱 전체에 공유된다 

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};
// 다른 컴포넌트들이 이 편의 창구(useAuth)를 통해 상자 안의 내용물을 꺼내쓸수있다 

export default AuthContext;

