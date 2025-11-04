/**
 * API 클라이언트 설정
 */

import axios from 'axios';

// Axios 인스턴스 생성
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request 인터셉터: 토큰 자동 추가
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response 인터셉터: 401 에러 처리
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // 토큰 만료 또는 무효
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API 함수들
export const authAPI = {
  register: (data) => api.post('/auth/register', data),
  login: (username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    return api.post('/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  getMe: () => api.get('/auth/me'),
};

export const businessAPI = {
  list: (params) => api.get('/businesses', { params }),
  get: (businessId) => api.get(`/businesses/${businessId}`),
  getReviews: (businessId, params) =>
    api.get(`/businesses/${businessId}/reviews`, { params }),
  createReview: (businessId, data) =>
    api.post(`/businesses/${businessId}/reviews`, data),
};

export const recommendationAPI = {
  get: (params) => api.get('/recommendations', { params }),
};

export default api;

