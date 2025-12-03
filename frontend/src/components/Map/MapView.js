import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Map, MapMarker, CustomOverlayMap } from 'react-kakao-maps-sdk';
import toast from 'react-hot-toast';
import './Map.css';

const MapView = ({ restaurants, onRestaurantSelect, onBoundsChange, onLocationChange, loading, isInitialLoading, initialCenter }) => {
  const [center, setCenter] = useState(initialCenter || { lat: 37.5665, lng: 126.9780 }); // 서울 중심 기본 위치
  const [userLocation, setUserLocation] = useState(null);
  const [mapLevel, setMapLevel] = useState(3);
  const debounceTimerRef = useRef(null);
  const initialLoadRef = useRef(false);
  const mapRef = useRef(null); // Map 객체 저장용 ref
  const lastBoundsRef = useRef(null); // 마지막 bounds 저장용 ref
  
  // 아주대학교 좌표
  const AJOU_UNIVERSITY = { lat: 37.2809, lng: 127.0447 };

  // initialCenter prop 변경 시 지도 중심 이동
  useEffect(() => {
    if (initialCenter && mapRef.current) {
      const moveLatLon = new window.kakao.maps.LatLng(initialCenter.lat, initialCenter.lng);
      mapRef.current.panTo(moveLatLon);
      setCenter(initialCenter);
    }
  }, [initialCenter]);

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
          // initialCenter가 없을 때만 사용자 위치로 설정
          if (!initialCenter) {
            setCenter(userPos);
          }
        },
        (error) => {
          console.log('위치 권한 거부 또는 오류:', error);
        }
      );
    }
  }, [initialCenter])

  // AI 점수에 따른 마커 색상 (5단계 빨간색 계열)
  const getMarkerColor = (aiScore) => {
    if (aiScore >= 4.5) return '#ff2929'; // 5단계: 진한 빨강 (4.5~5.0)
    if (aiScore >= 4.0) return '#ff4a4a'; // 4단계: 중간 빨강 (4.0~4.49)
    if (aiScore >= 3.5) return '#ff6b6b'; // 3단계: 기준 빨강 (3.5~3.99)
    if (aiScore >= 3.0) return '#ff9292'; // 2단계: 연한 빨강 (3.0~3.49)
    return '#ffb3b3'; // 1단계: 회색에 가까운 연한 빨강 (0~2.99)
  };

  // AI 점수에 따른 텍스트 색상 (마커 색상 기반, 가독성을 위해 약간 진한 톤)
  const getTextColor = (aiScore) => {
    if (aiScore >= 4.5) return '#cc0000'; // 5단계: 진한 빨강 텍스트
    if (aiScore >= 4.0) return '#cc1a1a'; // 4단계
    if (aiScore >= 3.5) return '#cc3333'; // 3단계
    if (aiScore >= 3.0) return '#cc4d4d'; // 2단계
    return '#cc6666'; // 1단계: 연한 빨강 텍스트
  };

  // 줌 레벨에 따른 최소 점수 (확대할수록 낮은 점수 음식점도 표시)
  const getMinScoreByLevel = (level) => {
    if (level <= 1) return 0;    // 레벨 1: 모든 음식점
    if (level <= 2) return 3.0;  // 레벨 2: 3.0 이상
    if (level <= 3) return 3.5;  // 레벨 3: 3.5 이상
    return 4.0;                   // 레벨 4+: 4.0 이상
  };

  // 레스토랑의 AI 점수 계산 (필터링용)
  const getAiScore = (restaurant) => {
    const deepfm = restaurant.ai_prediction?.deepfm_rating;
    const multitower = restaurant.ai_prediction?.multitower_rating;
    return (deepfm !== undefined && multitower !== undefined) 
      ? (deepfm + multitower) / 2 
      : (deepfm !== undefined ? deepfm : (multitower !== undefined ? multitower : 2.5));
  };

  // 줌 레벨에 따른 마커 크기 계산
  const getMarkerSize = (level) => {
    const minSize = 15; // 확대했을 때 (레벨 낮음)
    const maxSize = 35; // 축소했을 때 (레벨 높음)
    const minLevel = 1;
    const maxLevel = 14;
    
    // 정비례 관계 (레벨이 높을수록 = 축소할수록 마커 크게)
    const size = minSize + ((level - minLevel) / (maxLevel - minLevel)) * (maxSize - minSize);
    //return Math.max(minSize, Math.min(maxSize, size));
    return 25;
  };

  // 마커 클릭 핸들러
  const handleMarkerClick = (restaurant) => {
    if (onRestaurantSelect) {
      onRestaurantSelect(restaurant);
    }
  };

  // 지도 bounds 변경 핸들러 (드래그 끝, 줌 변경 시)
  const handleBoundsChange = useCallback((map) => {
    console.log('🔵 handleBoundsChange 호출됨', {
      timestamp: new Date().toISOString(),
      caller: new Error().stack.split('\n')[2]?.trim() || 'unknown'
    });
    
    // 기존 타이머 취소
    if (debounceTimerRef.current) {
      console.log('⏸️  기존 타이머 취소');
      clearTimeout(debounceTimerRef.current);
    }

    // 0.5초 후 새 데이터 로드
    debounceTimerRef.current = setTimeout(() => {
      console.log('⏰ Debounce 타이머 실행 - API 호출 시작');
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
      
      // 이전 bounds와 비교 (소수점 6자리까지 비교)
      const boundsEqual = (b1, b2) => {
        if (!b1 || !b2) return false;
        return (
          Math.abs(b1.north - b2.north) < 0.000001 &&
          Math.abs(b1.south - b2.south) < 0.000001 &&
          Math.abs(b1.east - b2.east) < 0.000001 &&
          Math.abs(b1.west - b2.west) < 0.000001
        );
      };
      
      if (boundsEqual(boundsData, lastBoundsRef.current)) {
        console.log('⏭️ 동일한 bounds - API 호출 건너뜀');
        return;
      }
      
      console.log('📊 API 호출할 bounds:', boundsData);
      lastBoundsRef.current = boundsData;
      
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

  // 커스텀 마커 컴포넌트 (지도핀 모양 + 이름 표시)
  const CustomMarker = ({ restaurant, mapLevel }) => {
    // DeepFM과 Multi-Tower의 평균값 사용, 없으면 기본값 2.5
    const deepfm = restaurant.ai_prediction?.deepfm_rating;
    const multitower = restaurant.ai_prediction?.multitower_rating;
    const aiScore = (deepfm !== undefined && multitower !== undefined) 
      ? (deepfm + multitower) / 2 
      : (deepfm !== undefined ? deepfm : (multitower !== undefined ? multitower : 2.5));
    const color = getMarkerColor(aiScore);
    const textColor = getTextColor(aiScore);
    const size = getMarkerSize(mapLevel);
    const showName = true; // 모든 줌 레벨에서 이름 표시
    
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          cursor: 'pointer',
          transition: 'transform 0.2s ease',
        }}
        onClick={() => handleMarkerClick(restaurant)}
        onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.15)'}
        onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
      >
        <svg
          width={size}
          height={size}
          viewBox="0 0 40 50"
          xmlns="http://www.w3.org/2000/svg"
          style={{
            filter: 'drop-shadow(0 2px 6px rgba(0,0,0,0.3))',
          }}
        >
          {/* 지도핀 모양 (물방울 형태) */}
          <path
            d="M20 0 C10 0, 2 8, 2 18 C2 28, 10 35, 20 50 C30 35, 38 28, 38 18 C38 8, 30 0, 20 0 Z"
            fill={color}
            stroke="white"
            strokeWidth="2.5"
          />
        </svg>
        {/* 가게 이름 (줌 레벨 4 이하에서만 표시) */}
        {showName && (
          <div
            style={{
              marginTop: '4px',
              padding: '4px 8px',
              borderRadius: '6px',
              fontSize: '12px',
              fontWeight: '600',
              whiteSpace: 'nowrap',
              background: 'rgba(255, 255, 255, 0.95)',
              boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
              border: '1px solid rgba(0,0,0,0.1)',
              color: textColor,
              maxWidth: '120px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              pointerEvents: 'none',
            }}
          >
            {restaurant.name}
          </div>
        )}
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
      toast.dismiss();
      toast.error('위치 정보를 가져올 수 없습니다.');
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
        style={{ width: '100%', height: 'var(--vh)' }}
        level={mapLevel}
        onCreate={(map) => { 
          // 이미 초기화되었으면 무시
          if (initialLoadRef.current) {
            console.log('⏭️ onCreate 무시됨 (이미 초기화됨)');
            return;
          }
          
          console.log('🟢 onCreate 호출됨!', new Date().toISOString());
          initialLoadRef.current = true; // 플래그 설정
          mapRef.current = map;
          // 지도 생성 후 초기 bounds 전달
          setTimeout(() => {
            console.log('🟢 onCreate의 setTimeout 실행');
            handleBoundsChange(map);
          }, 100);
        }}
        onDragEnd={(map) => {
          console.log('🟡 onDragEnd 호출됨!', new Date().toISOString());
          // 드래그가 끝났을 때만 호출
          handleBoundsChange(map);
        }}
        onZoomChanged={(map) => {
          const level = map.getLevel();
          console.log('🟠 onZoomChanged 호출됨!', {
            timestamp: new Date().toISOString(),
            level: level,
            previousLevel: mapLevel
          });
          setMapLevel(level);
          // 줌 변경이 끝났을 때만 호출
          handleBoundsChange(map);
        }}
        onIdle={(map) => {
          // 지도 이동/줌이 완전히 끝났을 때 호출 (panTo 포함)
          console.log('🟣 onIdle 호출됨!', new Date().toISOString());
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

        {/* 레스토랑 마커들 (줌 레벨에 따라 필터링) */}
        {restaurants && restaurants
          .filter((restaurant) => getAiScore(restaurant) >= getMinScoreByLevel(mapLevel))
          .map((restaurant) => (
            <CustomOverlayMap
              key={restaurant.id}
              position={{ lat: restaurant.latitude, lng: restaurant.longitude }}
              yAnchor={1}
            >
              <CustomMarker restaurant={restaurant} mapLevel={mapLevel} />
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

