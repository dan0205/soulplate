/**
 * 메인 App 컴포넌트
 */

import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { AuthProvider } from './context/AuthContext';
import PrivateRoute from './components/PrivateRoute';
import SplashScreen from './components/SplashScreen';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import HomePage from './pages/HomePage';
import MyProfilePage from './pages/MyProfilePage';
import UserProfilePage from './pages/UserProfilePage';
import MBTIDetailPage from './pages/MBTIDetailPage';
import TasteTestPage from './pages/TasteTestPage';
import TasteTestResultPage from './pages/TasteTestResultPage';
import { useKakaoLoader } from 'react-kakao-maps-sdk';

function App() {
  // 카카오맵 SDK 로드
  useKakaoLoader({
    appkey: process.env.REACT_APP_KAKAO_MAP_KEY || '',
    libraries: ['services', 'clusterer'],
  });

  // 스플래시 표시 여부를 동기적으로 초기화
  const [showSplash, setShowSplash] = useState(() => {
    return !sessionStorage.getItem('soulplate_first_visit');
  });

  return (
    <Router>
      <AuthProvider>
        {showSplash ? (
          <SplashScreen onComplete={() => setShowSplash(false)} />
        ) : (
          <>
            <Toaster
              position="bottom-center"
              toastOptions={{
                duration: 2000,
                style: {
                  background: '#ff6b6b',
                  color: '#ffffff',
                  borderRadius: '8px',
                  padding: '12px 20px',
                  fontSize: '15px',
                  fontWeight: '500',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                },
                className: 'toast-custom',
              }}
              containerStyle={{
                bottom: '20px',
              }}
              reverseOrder={false}
            />
            <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route
            path="/"
            element={
              <PrivateRoute>
                <HomePage />
              </PrivateRoute>
            }
          />
          <Route
            path="/my-profile"
            element={
              <PrivateRoute>
                <MyProfilePage />
              </PrivateRoute>
            }
          />
          <Route
            path="/profile/mbti"
            element={
              <PrivateRoute>
                <MBTIDetailPage />
              </PrivateRoute>
            }
          />
          <Route
            path="/profile/:userId"
            element={
              <PrivateRoute>
                <UserProfilePage />
              </PrivateRoute>
            }
          />
          <Route
            path="/taste-test"
            element={
              <PrivateRoute>
                <TasteTestPage />
              </PrivateRoute>
            }
          />
          <Route
            path="/taste-test/result"
            element={
              <PrivateRoute>
                <TasteTestResultPage />
              </PrivateRoute>
            }
          />
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          </>
        )}
      </AuthProvider>
    </Router>
  );
}

export default App;
