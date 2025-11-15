/**
 * 홈 페이지 - 개인화 추천 표시
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { businessAPI, userAPI } from '../services/api';
import TasteTestModal from '../components/TasteTestModal';
import MapView from '../components/Map/MapView';
import MapToggle from '../components/Map/MapToggle';
import MapBottomSheet from '../components/Map/MapBottomSheet';
import FloatingProfileButton from '../components/FloatingProfileButton';
import './Home.css';

const HomePage = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [sortBy, setSortBy] = useState('');
  const [showTasteTestModal, setShowTasteTestModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [viewMode, setViewMode] = useState('map'); // 'map' or 'list'
  const [mapRestaurants, setMapRestaurants] = useState([]); // 지도용 레스토랑 데이터
  const [selectedRestaurant, setSelectedRestaurant] = useState(null); // 선택된 레스토랑 (하단 시트용)
  const itemsPerPage = 20;
  
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  // Debounce 검색어 (300ms 지연)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchQuery);
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  // 검색어 변경 시 페이지를 1로 리셋
  useEffect(() => {
    if (debouncedSearch !== '') {
      setCurrentPage(1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSearch]);

  // 검색어 또는 페이지, 정렬 변경 시 데이터 로드
  useEffect(() => {
    if (viewMode === 'list') {
      loadRecommendations();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPage, sortBy, debouncedSearch, viewMode]);

  // 지도 뷰일 때는 MapView 내부에서 초기 위치 기반으로 자동 로드됨

  useEffect(() => {
    checkUserStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const checkUserStatus = async () => {
    try {
      const response = await userAPI.getStatus();
      const { should_show_test_popup } = response.data;
      
      // 로컬 스토리지에서 "나중에 하기" 확인
      const skipped = localStorage.getItem('taste_test_skipped');
      
      if (should_show_test_popup && !skipped) {
        setShowTasteTestModal(true);
      }
    } catch (err) {
      console.error('사용자 상태 확인 실패:', err);
    }
  };

  const loadRecommendations = async () => {
    setLoading(true);
    setError('');
    
    try {
      // 비즈니스 목록 가져오기
      const skip = (currentPage - 1) * itemsPerPage;
      const params = { skip, limit: itemsPerPage };
      if (sortBy) {
        params.sort_by = sortBy;
      }
      if (debouncedSearch) {
        params.search = debouncedSearch;
      }
      const response = await businessAPI.list(params);
      
      // 응답 구조 확인: response.data가 { businesses, total, skip, limit } 형태
      const { businesses, total } = response.data;
      
      // 총 페이지 수 계산
      setTotalPages(Math.ceil(total / itemsPerPage));
      
      // 각 비즈니스를 표시 형식으로 변환
      const businessesWithPredictions = businesses.map(business => ({
        business: business,
        score: null,  // 추천 점수는 없음
        prediction: business.ai_prediction || null  // AI 예측
      }));
      
      setRecommendations(businessesWithPredictions);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load businesses');
      console.error('Error loading businesses:', err);
    } finally {
      setLoading(false);
    }
  };

  // 지도용 레스토랑 로드
  const loadMapRestaurants = async (lat, lng) => {
    setLoading(true);
    setError('');
    
    try {
      const response = await businessAPI.getForMap({
        lat,
        lng,
        radius: 10, // 10km 반경
        limit: 100
      });
      
      const { businesses } = response.data;
      
      // 위도/경도가 null인 레스토랑 필터링
      const validBusinesses = businesses.filter(
        b => b.latitude !== null && b.longitude !== null
      );
      
      // 지도용 데이터로 변환
      const mapData = validBusinesses.map(business => ({
        business: business,
        score: null,
        prediction: business.ai_prediction || null
      }));
      
      setMapRestaurants(mapData);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load map restaurants');
      console.error('Error loading map restaurants:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (page) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const renderPageNumbers = () => {
    const pages = [];
    const maxVisible = 5;
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    
    // 조정: 끝에서 시작할 때
    if (endPage - startPage < maxVisible - 1) {
      startPage = Math.max(1, endPage - maxVisible + 1);
    }
    
    // 첫 페이지
    if (startPage > 1) {
      pages.push(
        <button
          key={1}
          className="pagination-button"
          onClick={() => handlePageChange(1)}
        >
          1
        </button>
      );
      if (startPage > 2) {
        pages.push(<span key="dots-start" className="pagination-dots">...</span>);
      }
    }
    
    // 페이지 번호들
    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <button
          key={i}
          className={`pagination-button ${i === currentPage ? 'active' : ''}`}
          onClick={() => handlePageChange(i)}
        >
          {i}
        </button>
      );
    }
    
    // 마지막 페이지
    if (endPage < totalPages) {
      if (endPage < totalPages - 1) {
        pages.push(<span key="dots-end" className="pagination-dots">...</span>);
      }
      pages.push(
        <button
          key={totalPages}
          className="pagination-button"
          onClick={() => handlePageChange(totalPages)}
        >
          {totalPages}
        </button>
      );
    }
    
    return pages;
  };

  const handleBusinessClick = (businessId) => {
    navigate(`/business/${businessId}`);
  };

  const handleSortChange = (newSortBy) => {
    setSortBy(newSortBy);
    setCurrentPage(1); // 정렬 변경 시 1페이지로 리셋
  };

  const handleSearchChange = (e) => {
    setSearchQuery(e.target.value);
  };

  const clearSearch = () => {
    setSearchQuery('');
    setDebouncedSearch('');
  };

  // 지도 위치 변경 핸들러
  const handleMapLocationChange = (lat, lng) => {
    loadMapRestaurants(lat, lng);
  };

  // 지도용 레스토랑 데이터 변환
  const dataToUse = viewMode === 'map' ? mapRestaurants : recommendations;
  const restaurantsForMap = dataToUse.map(item => ({
    id: item.business.business_id,
    name: item.business.name,
    latitude: item.business.latitude,
    longitude: item.business.longitude,
    stars: item.business.stars,
    ai_prediction: item.business.ai_prediction?.deepfm_rating || item.business.stars,
    categories: item.business.categories,
    address: item.business.address || `${item.business.city}, ${item.business.state}`,
    review_count: item.business.review_count,
    absa_food_avg: item.business.absa_food_avg,
    absa_service_avg: item.business.absa_service_avg,
    absa_atmosphere_avg: item.business.absa_atmosphere_avg,
  }));

  return (
    <div className="home-container">
      {/* 플로팅 프로필 버튼 */}
      <FloatingProfileButton username={user?.username} onLogout={logout} />

      {/* 지도/리스트 토글 버튼 */}
      <MapToggle viewMode={viewMode} onToggle={setViewMode} />

      {/* 지도 뷰 */}
      {viewMode === 'map' ? (
        <>
          <MapView 
            restaurants={restaurantsForMap}
            onRestaurantSelect={setSelectedRestaurant}
            onLocationChange={handleMapLocationChange}
            loading={loading}
          />
          <MapBottomSheet 
            restaurant={selectedRestaurant}
            onClose={() => setSelectedRestaurant(null)}
          />
        </>
      ) : (
        <main className="home-main">
        <div className="recommendations-header">
          <h2>🏪 Restaurant List</h2>
          
          <div className="search-section">
            <div className="search-input-wrapper">
              <input 
                type="text"
                className="search-input"
                placeholder="🔍 음식점 이름, 카테고리, 지역 검색..."
                value={searchQuery}
                onChange={handleSearchChange}
              />
              {searchQuery && (
                <button className="clear-search-btn" onClick={clearSearch}>
                  ✕
                </button>
              )}
            </div>
          </div>

          <div className="header-actions">
            <div className="sort-buttons">
              <button 
                className={`sort-btn ${sortBy === '' ? 'active' : ''}`}
                onClick={() => handleSortChange('')}
              >
                기본
              </button>
              <button 
                className={`sort-btn ${sortBy === 'deepfm' ? 'active' : ''}`}
                onClick={() => handleSortChange('deepfm')}
              >
                DeepFM 별점순
              </button>
              <button 
                className={`sort-btn ${sortBy === 'multitower' ? 'active' : ''}`}
                onClick={() => handleSortChange('multitower')}
              >
                Multi-Tower 별점순
              </button>
              <button 
                className={`sort-btn ${sortBy === 'review_count' ? 'active' : ''}`}
                onClick={() => handleSortChange('review_count')}
              >
                리뷰 많은순
              </button>
            </div>
            <button onClick={loadRecommendations} className="btn-refresh" disabled={loading}>
              {loading ? 'Loading...' : '🔄 Refresh'}
            </button>
          </div>
        </div>

        {error && (
          <div className="error-banner">
            {error}
            <button onClick={loadRecommendations}>Retry</button>
          </div>
        )}

        {loading ? (
          <div className="loading-container">
            <div className="spinner"></div>
            <p>Loading your personalized recommendations...</p>
          </div>
        ) : (
          <>
            <div className="recommendations-grid">
              {recommendations.length === 0 ? (
                <div className="no-results">
                  <p>{searchQuery ? `"${searchQuery}"에 대한 검색 결과가 없습니다.` : 'No businesses available.'}</p>
                </div>
              ) : (
                recommendations.map((item, index) => (
                  <div
                    key={item.business.business_id}
                    className="business-card"
                    onClick={() => handleBusinessClick(item.business.business_id)}
                  >
                    <div className="card-rank">#{(currentPage - 1) * itemsPerPage + index + 1}</div>
                    <h3>{item.business.name}</h3>
                    <div className="card-info">
                      <span className="reviews">📝 {item.business.review_count} reviews</span>
                    </div>
                    {item.business.ai_prediction ? (
                      <div className="ai-prediction-inline">
                        🤖 AI 예상: {item.business.ai_prediction.deepfm_rating?.toFixed(1)} (DeepFM) / {item.business.ai_prediction.multitower_rating?.toFixed(1) || 'N/A'} (Multi-Tower)
                      </div>
                    ) : user ? (
                      <div className="ai-prediction-inline" style={{background: '#fff3cd', color: '#856404'}}>
                        ⏳ AI 예측 계산 중... (백그라운드 처리)
                      </div>
                    ) : (
                      <div className="ai-prediction-inline" style={{background: '#f0f0f0', color: '#666'}}>
                        ⚠️ AI 예측을 사용하려면 로그인이 필요합니다
                      </div>
                    )}
                    <p className="categories">{item.business.categories || 'No category'}</p>
                    <p className="location">📍 {item.business.city}, {item.business.state}</p>
                    {item.business.top_features && item.business.top_features.length > 0 && (
                      <div className="top-features">
                        {item.business.top_features.slice(0, 3).map((feature, idx) => (
                          <span key={idx} className="feature-tag">
                            {feature.aspect} ({Math.round(feature.score * 100)}%)
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
            
            {totalPages > 1 && (
              <div className="pagination-container">
                <button
                  className="pagination-button"
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                >
                  ← 이전
                </button>
                
                {renderPageNumbers()}
                
                <button
                  className="pagination-button"
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages}
                >
                  다음 →
                </button>
              </div>
            )}
          </>
        )}
        </main>
      )}

      {showTasteTestModal && (
        <TasteTestModal onClose={() => setShowTasteTestModal(false)} />
      )}
    </div>
  );
};

export default HomePage;

