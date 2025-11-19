/**
 * Private Route 컴포넌트
 * 인증이 필요한 페이지를 보호
 */
// react 어플리케이션에서 보안, 라우팅을 처리하는 부분이다 = 보안 요원 
// 인증(로그인)이 필요한 페이지를 보호하는 역할을 한다 

import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const PrivateRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  // AuthContext라는 전역 상태 관리자에서 현재 인증 상태를 가져오는 React Hook이다 
  // isAuthenticated = 사용자가 로그인했는지 여부 true/false 
  // loading = 현재 로그인 상태를 확인 중인지 여부 true/false 
  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'var(--vh)' }}>
        <h2>Loading...</h2>
      </div>
    );
  } // 앱이 토큰을 검증하기 전에 잠깐 시간벌기 

  return isAuthenticated ? children : <Navigate to="/login" />;
}; 
// isAuthenticated가 true라면 children을 그래도 표시
// 아니라면 Navigate를 반환해서 login 페이지로 즉시 강제 이동 

export default PrivateRoute;

