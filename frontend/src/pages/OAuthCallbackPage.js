import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import toast from 'react-hot-toast';

const OAuthCallbackPage = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { handleOAuthCallback } = useAuth();
  const [isProcessing, setIsProcessing] = useState(true);

  useEffect(() => {
    const processCallback = async () => {
      const token = searchParams.get('token');
      const error = searchParams.get('error');

      if (error) {
        toast.error('로그인에 실패했습니다. 다시 시도해주세요.');
        setIsProcessing(false);
        navigate('/login', { replace: true });
        return;
      }

      if (token) {
        try {
          await handleOAuthCallback(token);
          toast.success('로그인되었습니다!');
          setIsProcessing(false);
        } catch (err) {
          toast.error('로그인 처리 중 오류가 발생했습니다.');
          setIsProcessing(false);
          navigate('/login', { replace: true });
        }
      } else {
        toast.error('인증 정보가 없습니다.');
        setIsProcessing(false);
        navigate('/login', { replace: true });
      }
    };

    processCallback();
  }, [searchParams, navigate, handleOAuthCallback]);

  if (!isProcessing) {
    return null;
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      backgroundColor: '#f5f5f5',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{
        padding: '40px',
        backgroundColor: 'white',
        borderRadius: '12px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
        textAlign: 'center'
      }}>
        <div style={{
          width: '48px',
          height: '48px',
          margin: '0 auto 20px',
          border: '4px solid #f3f3f3',
          borderTop: '4px solid #ff6b6b',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }}></div>
        <p style={{
          fontSize: '18px',
          fontWeight: '500',
          color: '#333',
          margin: '0'
        }}>로그인 중...</p>
        <p style={{
          fontSize: '14px',
          color: '#666',
          marginTop: '8px'
        }}>잠시만 기다려주세요</p>
      </div>
      
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default OAuthCallbackPage;







