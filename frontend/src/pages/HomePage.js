/**
 * 홈 페이지 - 지도 기반 통합 UI
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { businessAPI, userAPI } from '../services/api';
import TasteTestModal from '../components/TasteTestModal';
import MapView from '../components/Map/MapView';
import MapBottomSheet from '../components/Map/MapBottomSheet';
import FloatingProfileButton from '../components/FloatingProfileButton';
import FloatingSearchBar from '../components/Map/FloatingSearchBar';
import { calculateDistance } from '../utils/distance';
import './Home.css';

const HomePage = () => {
  const [loading, setLoading] = useState(true);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [error, setError] = useState('');
  const [sortBy, setSortBy] = useState('deepfm');
  const [showTasteTestModal, setShowTasteTestModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [mapRestaurants, setMapRestaurants] = useState([]);
  const [selectedRestaurant, setSelectedRestaurant] = useState(null);
  const [displayedCount, setDisplayedCount] = useState(20);
  const [userLocation, setUserLocation] = useState(null);
  const [currentBounds, setCurrentBounds] = useState(null);
  
  const LOAD_MORE_COUNT = 20;
  
  const { user, logout, loading: authLoading } = useAuth();

  // Debounce 검색어 (300ms 지연)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchQuery);
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  // 검색어 변경 시 지도 범위 재로드
  useEffect(() => {
    if (currentBounds) {
      loadMapRestaurants(currentBounds);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSearch]);

  // 사용자 위치 가져오기
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
        },
        (error) => {
          console.log('위치 권한 거부 또는 오류:', error);
        }
      );
    }
  }, []);

  // 사용자 인증이 완료된 후 취향 테스트 모달 표시 여부 확인
  useEffect(() => {
    // 인증 로딩이 완료되고 사용자 정보가 있을 때만 상태 확인
    if (!authLoading && user) {
      checkUserStatus();
    }
  }, [authLoading, user]);

  const checkUserStatus = async () => {
    try {
      console.log('[취향 테스트 모달] 사용자 상태 확인 시작');
      const response = await userAPI.getStatus();
      const { should_show_test_popup, is_new_user, has_taste_test, review_count } = response.data;
      
      console.log('[취향 테스트 모달] API 응답:', {
        should_show_test_popup,
        is_new_user,
        has_taste_test,
        review_count
      });
      
      const skipped = localStorage.getItem('taste_test_skipped');
      console.log('[취향 테스트 모달] 건너뛰기 상태:', skipped);
      
      if (should_show_test_popup && !skipped) {
        console.log('[취향 테스트 모달] 모달 표시 조건 충족 - 모달 표시');
        setShowTasteTestModal(true);
      } else {
        console.log('[취향 테스트 모달] 모달 표시 조건 불충족:', {
          should_show_test_popup,
          skipped: !!skipped
        });
      }
    } catch (err) {
      console.error('[취향 테스트 모달] 사용자 상태 확인 실패:', {
        error: err,
        message: err.message,
        response: err.response?.data,
        status: err.response?.status
      });
    }
  };

  // 지도 범위 기반 레스토랑 로드
  const loadMapRestaurants = useCallback(async (bounds) => {
    setLoading(true);
    setError('');
    setCurrentBounds(bounds);
    
    try {
      const response = await businessAPI.getInBounds({
        north: bounds.north,
        south: bounds.south,
        east: bounds.east,
        west: bounds.west,
        limit: 200,
        search: debouncedSearch || undefined
      });
      
      const { businesses } = response.data;
      
      // 거리 계산 추가
      const businessesWithDistance = businesses.map(b => ({
        ...b,
        distance: userLocation 
          ? calculateDistance(userLocation.lat, userLocation.lng, b.latitude, b.longitude)
          : null
      }));
      
      setMapRestaurants(businessesWithDistance);
      setDisplayedCount(20); // 표시 개수 리셋
      
      if (isInitialLoading) {
        setIsInitialLoading(false);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load restaurants');
      console.error('Error loading restaurants:', err);
    } finally {
      setLoading(false);
    }
  }, [debouncedSearch, userLocation, isInitialLoading]);

  // 지도 bounds 변경 핸들러
  const handleMapBoundsChange = useCallback((bounds) => {
    loadMapRestaurants(bounds);
  }, [loadMapRestaurants]);

  // 더보기 버튼 핸들러
  const handleLoadMore = () => {
    setDisplayedCount(prev => Math.min(prev + LOAD_MORE_COUNT, mapRestaurants.length));
  };

  // 정렬 변경 핸들러
  const handleSortChange = (newSortBy) => {
    setSortBy(newSortBy);
    setDisplayedCount(20); // 리셋
  };

  // 정렬 함수
  const sortRestaurants = useMemo(() => {
    return (restaurants, sortBy) => {
      const sorted = [...restaurants];
      switch(sortBy) {
        case 'deepfm':
          return sorted.sort((a, b) => 
            (b.ai_prediction?.deepfm_rating || b.stars) - 
            (a.ai_prediction?.deepfm_rating || a.stars)
          );
        case 'multitower':
          return sorted.sort((a, b) => 
            (b.ai_prediction?.multitower_rating || b.ai_prediction?.deepfm_rating || b.stars) - 
            (a.ai_prediction?.multitower_rating || a.ai_prediction?.deepfm_rating || a.stars)
          );
        case 'distance':
          return sorted.sort((a, b) => (a.distance || 999) - (b.distance || 999));
        case 'review_count':
          return sorted.sort((a, b) => b.review_count - a.review_count);
        default:
          return sorted;
      }
    };
  }, []);

  // 정렬된 레스토랑 목록
  const sortedRestaurants = useMemo(() => {
    return sortRestaurants(mapRestaurants, sortBy);
  }, [mapRestaurants, sortBy, sortRestaurants]);

  // 레스토랑 선택 핸들러
  const handleRestaurantSelect = (restaurant) => {
    setSelectedRestaurant(restaurant);
  };

  // 레스토랑 선택 해제 핸들러
  const handleRestaurantClose = () => {
    setSelectedRestaurant(null);
  };

  // 검색 핸들러
  const handleSearch = (query) => {
    setSearchQuery(query);
  };

  return (
    <div className="home-container">
      {/* 플로팅 프로필 버튼 */}
      <FloatingProfileButton username={user?.username} onLogout={logout} />

      {/* 플로팅 검색창 */}
      <FloatingSearchBar 
        onSearch={handleSearch}
        placeholder="🔍 음식점 이름, 카테고리, 지역 검색..."
        defaultValue={searchQuery}
      />

      {/* 지도 (항상 표시) */}
      <MapView 
        restaurants={sortedRestaurants}
        onRestaurantSelect={handleRestaurantSelect}
        onBoundsChange={handleMapBoundsChange}
        loading={loading}
        isInitialLoading={isInitialLoading}
      />

      {/* 하단 시트 (통합) */}
      <MapBottomSheet 
        restaurants={sortedRestaurants}
        displayedCount={displayedCount}
        onLoadMore={handleLoadMore}
        selectedRestaurant={selectedRestaurant}
        onSelectRestaurant={handleRestaurantSelect}
        onClose={handleRestaurantClose}
        sortBy={sortBy}
        onSortChange={handleSortChange}
      />

      {/* 취향 테스트 모달 */}
      {showTasteTestModal && (
        <TasteTestModal onClose={() => setShowTasteTestModal(false)} />
      )}
    </div>
  );
};

export default HomePage;
