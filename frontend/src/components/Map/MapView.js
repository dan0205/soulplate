import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Map, MapMarker, CustomOverlayMap } from 'react-kakao-maps-sdk';
import './Map.css';

const MapView = ({ restaurants, onRestaurantSelect, onLocationChange, loading, isInitialLoading }) => {
  const [center, setCenter] = useState({ lat: 37.5665, lng: 126.9780 }); // 서울 중심 기본 위치
  const [userLocation, setUserLocation] = useState(null);
  const [mapLevel, setMapLevel] = useState(3);
  const debounceTimerRef = useRef(null);
  const initialLoadRef = useRef(false);
  const mapRef = useRef(null); // Map 객체 저장용 ref
  
  // 아주대학교 좌표
  const AJOU_UNIVERSITY = { lat: 37.2809, lng: 127.0447 };

  // 사용자 위치 가져오기 및 초기 API 호출 (한 번만 실행)
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const userPos = {
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          };
          setUserLocation(userPos);
          setCenter(userPos);
          
          // 초기 로드: 사용자 위치 기준으로 마커 로드
          if (onLocationChange) {
            onLocationChange(userPos.lat, userPos.lng);
          }
        },
        (error) => {
          console.log('위치 권한 거부 또는 오류:', error);
          
          // 폴백: 서울 중심으로 마커 로드
          if (onLocationChange) {
            onLocationChange(center.lat, center.lng);
          }
        }
      );
    } else {
      // geolocation 미지원: 서울 중심으로 마커 로드
      if (onLocationChange) {
        onLocationChange(center.lat, center.lng);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // 빈 배열: 초기 한 번만 실행

  // AI 점수에 따른 마커 색상
  const getMarkerColor = (aiScore) => {
    if (aiScore >= 4.5) return '#FF4444'; // 빨강 (강력 추천)
    if (aiScore >= 4.0) return '#FF8800'; // 주황 (추천)
    if (aiScore >= 3.5) return '#FFD700'; // 노랑 (괜찮음)
    return '#CCCCCC'; // 회색 (보통)
  };

  // 마커 클릭 핸들러
  const handleMarkerClick = (restaurant) => {
    if (onRestaurantSelect) {
      onRestaurantSelect(restaurant);
    }
  };

  // 지도 이동 핸들러 (디바운싱)
  const handleMapCenterChanged = useCallback((map) => {
    // 기존 타이머 취소
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // 0.5초 후 새 데이터 로드 (상단 프로그레스 바로 빠른 업데이트 가능)
    debounceTimerRef.current = setTimeout(() => {
      const newCenter = map.getCenter();
      const lat = newCenter.getLat();
      const lng = newCenter.getLng();
      
      if (onLocationChange) {
        onLocationChange(lat, lng);
      }
    }, 500);
  }, [onLocationChange]);

  // 컴포넌트 언마운트 시 타이머 정리
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  // 커스텀 마커 컴포넌트
  const CustomMarker = ({ restaurant }) => {
    const color = getMarkerColor(restaurant.ai_prediction || restaurant.stars);
    const score = (restaurant.ai_prediction || restaurant.stars).toFixed(1);
    
    return (
      <div
        style={{
          backgroundColor: color,
          color: 'white',
          border: '3px solid white',
          borderRadius: '50%',
          width: '50px',
          height: '50px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 'bold',
          fontSize: '14px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
          cursor: 'pointer',
          transition: 'transform 0.2s',
        }}
        onClick={() => handleMarkerClick(restaurant)}
        onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
        onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
      >
        {score}
      </div>
    );
  };

  // 내 위치로 이동하는 핸들러
  const handleGoToMyLocation = () => {
    if (mapRef.current && userLocation) {
      // Kakao Map의 panTo() 메서드를 사용하여 부드럽게 이동
      const moveLatLon = new window.kakao.maps.LatLng(userLocation.lat, userLocation.lng);
      mapRef.current.panTo(moveLatLon);
    } else if (!userLocation) {
      alert('위치 정보를 가져올 수 없습니다.');
    }
  };

  // 아주대학교로 이동하는 핸들러
  const handleGoToAjouUniversity = () => {
    if (mapRef.current) {
      // Kakao Map의 panTo() 메서드를 사용하여 부드럽게 이동
      const moveLatLon = new window.kakao.maps.LatLng(AJOU_UNIVERSITY.lat, AJOU_UNIVERSITY.lng);
      mapRef.current.panTo(moveLatLon);
    }
  };

  return (
    <div className="map-container">
      {/* 초기 로딩: 전체 화면 오버레이 + 스피너 */}
      {loading && isInitialLoading && (
        <div className="map-loading-overlay">
          <div className="spinner"></div>
        </div>
      )}
      
      {/* 재로딩: 상단 프로그레스 바만 표시 */}
      {loading && !isInitialLoading && (
        <div className="map-progress-bar">
          <div className="progress-bar-fill"></div>
        </div>
      )}
      
      <Map
        center={center}
        style={{ width: '100%', height: '100vh' }}
        level={mapLevel}
        onCreate={(map) => { mapRef.current = map; }}
        onZoomChanged={(map) => setMapLevel(map.getLevel())}
        onCenterChanged={handleMapCenterChanged}
      >
        {/* 사용자 위치 마커 */}
        {userLocation && (
          <MapMarker
            position={userLocation}
            image={{
              src: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAiIGhlaWdodD0iMzAiIHZpZXdCb3g9IjAgMCAzMCAzMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxNSIgY3k9IjE1IiByPSIxMCIgZmlsbD0iIzY2N2VlYSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIzIi8+PC9zdmc+',
              size: { width: 30, height: 30 },
            }}
          />
        )}

        {/* 레스토랑 마커들 */}
        {restaurants && restaurants.map((restaurant) => (
          <CustomOverlayMap
            key={restaurant.id}
            position={{ lat: restaurant.latitude, lng: restaurant.longitude }}
            yAnchor={0.5}
          >
            <CustomMarker restaurant={restaurant} />
          </CustomOverlayMap>
        ))}
      </Map>
      
      {/* 아주대학교 바로가기 버튼 */}
      <button className="ajou-university-btn" onClick={handleGoToAjouUniversity} title="아주대학교로 이동">
        🏫
      </button>
      
      {/* 내 위치 버튼 */}
      {userLocation && (
        <button className="my-location-btn" onClick={handleGoToMyLocation} title="내 위치로 이동">
          📍
        </button>
      )}
    </div>
  );
};

export default MapView;

