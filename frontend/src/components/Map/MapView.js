import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Map, MapMarker, CustomOverlayMap } from 'react-kakao-maps-sdk';
import './Map.css';

const MapView = ({ restaurants, onRestaurantSelect, onBoundsChange, onLocationChange, loading, isInitialLoading }) => {
  const [center, setCenter] = useState({ lat: 37.5665, lng: 126.9780 }); // 서울 중심 기본 위치
  const [userLocation, setUserLocation] = useState(null);
  const [mapLevel, setMapLevel] = useState(3);
  const debounceTimerRef = useRef(null);
  const initialLoadRef = useRef(false);
  const mapRef = useRef(null); // Map 객체 저장용 ref
  
  // 아주대학교 좌표
  const AJOU_UNIVERSITY = { lat: 37.2809, lng: 127.0447 };

  // 사용자 위치 가져오기 (위치만 설정, API 호출은 지도 생성 후 자동)
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
        },
        (error) => {
          console.log('위치 권한 거부 또는 오류:', error);
        }
      );
    }
  }, [])

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

  // 지도 bounds 변경 핸들러 (드래그 끝, 줌 변경 시)
  const handleBoundsChange = useCallback((map) => {
    // 기존 타이머 취소
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // 0.5초 후 새 데이터 로드
    debounceTimerRef.current = setTimeout(() => {
      // Bounds 정보 추출
      const bounds = map.getBounds();
      const sw = bounds.getSouthWest();
      const ne = bounds.getNorthEast();
      
      const boundsData = {
        north: ne.getLat(),
        south: sw.getLat(),
        east: ne.getLng(),
        west: sw.getLng()
      };
      
      if (onBoundsChange) {
        onBoundsChange(boundsData);
      } else if (onLocationChange) {
        // 호환성을 위해 onLocationChange도 지원
        const newCenter = map.getCenter();
        onLocationChange(newCenter.getLat(), newCenter.getLng());
      }
    }, 500);
  }, [onBoundsChange, onLocationChange]);

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
        onCreate={(map) => { 
          mapRef.current = map;
          // 지도 생성 후 초기 bounds 전달
          setTimeout(() => {
            handleBoundsChange(map);
          }, 100);
        }}
        onDragEnd={(map) => {
          // 드래그가 끝났을 때만 호출
          handleBoundsChange(map);
        }}
        onZoomChanged={(map) => {
          setMapLevel(map.getLevel());
          // 줌 변경이 끝났을 때만 호출
          handleBoundsChange(map);
        }}
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

