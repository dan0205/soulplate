/**
 * API 클라이언트 설정
 */
// react 앱에서 백엔드 FastAPI 서버와 통신하는 모든 API 요청을 중앙에서 관리하는
// API 클라이언트 파일이다 
// axios 라이브러리를 기반으로 하며 목적은 다음과 같다 
// 중앙화: 모든 api 요청의 기본 주소를 한곳에서 관리
// 자동화: 모든 요청에 자동으로 JWT 토큰을 추가한다
// 오류처리: 토큰이 만료될때 자동으로 로그아웃
// 추상화: 복잡한 API호출을 간단한 함수로 만들어 제공 

import axios from 'axios';

// Axios 인스턴스 생성
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});
// API 요청을 위한 기본 설정을 갖춘 axios 인스턴스를 생성한다 
// baseURL: 'http://localhost:8000/api' = 매우 편리한 기능. /로 시작하는 상대 경로만 적어도,
// axios가 자동으로 baseURL 뒤에 붙여서 전체 요청 URL을 만들어준다 
// headers: { 'Content-Type': 'application/json' } = 서버에 보내는 데이터의 기본 형식을 JSON으로 지정한다 

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
// react 앱이 백엔드로 api 요청을 보내기 직전에 해당 요청을 가로채서 자동으로 JWT를 헤더에 추가한다 
// 인증이 필요한 모든 API를 호출할 때마다 헤더를 수동으로 추가하는 코드를 작성할 필요가 없다 

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
// 백엔드로부터 API 응답을 받은 직후에 해당 응답을 가로채서 에러처리를 수행한다 
// 토큰 문제가 발생하면 사용자에게 에러를 보여주지 않고 로그인 페이지로 강제 이동 

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
  getForMap: (params) => api.get('/businesses/map', { params }),
  getInBounds: (params) => api.get('/businesses/in-bounds', { params }),
  get: (businessId) => api.get(`/businesses/${businessId}`),
  getReviews: (businessId, params) =>
    api.get(`/businesses/${businessId}/reviews`, { params }),
  createReview: (businessId, data) =>
    api.post(`/businesses/${businessId}/reviews`, data),
};

export const reviewAPI = {
  incrementUseful: (reviewId) => api.put(`/reviews/${reviewId}/useful`),
};

export const userAPI = {
  getMyProfile: () => api.get('/users/me/profile'),
  getUserProfile: (userId) => api.get(`/users/${userId}/profile`),
  getUserReviews: (userId, params) => api.get(`/users/${userId}/reviews`, { params }),
  getStatus: () => api.get('/user/status'),
};

export const tasteTestAPI = {
  getQuestions: (testType = 'quick') => api.get('/taste-test/questions', { params: { test_type: testType } }),
  submit: (data) => api.post('/taste-test/submit', data),
  delete: () => api.delete('/taste-test'),
};

export const recommendationAPI = {
  get: (params) => api.get('/recommendations', { params }),
};

export default api;

