/**
 * 메인 App 컴포넌트
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import PrivateRoute from './components/PrivateRoute';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import HomePage from './pages/HomePage';
import BusinessDetailPage from './pages/BusinessDetailPage';
import MyProfilePage from './pages/MyProfilePage';
import UserProfilePage from './pages/UserProfilePage';
import TasteTestPage from './pages/TasteTestPage';
import TasteTestResultPage from './pages/TasteTestResultPage';

function App() {
  return (
    <Router>
      <AuthProvider>
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
            path="/business/:businessId"
            element={
              <PrivateRoute>
                <BusinessDetailPage />
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
      </AuthProvider>
    </Router>
  );
}

export default App;
