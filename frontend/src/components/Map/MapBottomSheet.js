import React, { useState, useEffect, useRef } from 'react';
import { BottomSheet } from 'react-spring-bottom-sheet';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import toast from 'react-hot-toast';
import 'react-spring-bottom-sheet/dist/style.css';
import 'react-tabs/style/react-tabs.css';
import './Map.css';
import HomeTab from './tabs/HomeTab';
import MenuTab from './tabs/MenuTab';
import ReviewTab from './tabs/ReviewTab';
import PhotoTab from './tabs/PhotoTab';
import SortDropdown from './SortDropdown';
import { formatDistance } from '../../utils/distance';

// AI 점수 기반 색상 (공통 함수)
const getMarkerColor = (score) => {
  if (score > 4.0) return '#ff6b6b'; // 연한 빨강 (높은 점수)
  if (score > 3.0) return '#FFB74D'; // 연한 주황 (중간 점수)
  return '#FFF176'; // 연한 노랑 (낮은 점수)
};

const RestaurantListItem = ({ restaurant, onClick }) => {
  // DeepFM과 Multi-Tower의 평균값 사용, 없으면 기본값 3.0
  const deepfm = restaurant.ai_prediction?.deepfm_rating;
  const multitower = restaurant.ai_prediction?.multitower_rating;
  const aiScore = (deepfm !== undefined && multitower !== undefined) 
    ? (deepfm + multitower) / 2 
    : (deepfm !== undefined ? deepfm : (multitower !== undefined ? multitower : 3.0));
  
  // 리스트 카드용 DeepFM 점수 (배지 색상용)
  const deepfmScore = restaurant.ai_prediction?.deepfm_rating || restaurant.stars || 3.0;

  return (
    <div className="restaurant-list-item" onClick={onClick}>
      <div 
        className="list-item-marker" 
        style={{ backgroundColor: getMarkerColor(aiScore) }}
      >
        {aiScore.toFixed(1)}
      </div>
      <div className="list-item-content">
        <div className="list-item-header">
          <h4>{restaurant.name}</h4>
          <div className="list-item-badges">
            <span 
              className="badge-deepfm"
              style={{ backgroundColor: getMarkerColor(deepfmScore) }}
            >
              {restaurant.ai_prediction?.deepfm_rating?.toFixed(1) || restaurant.stars.toFixed(1)}
            </span>
          </div>
        </div>
        <div className="list-item-meta">
          <span>{restaurant.categories}</span>
          {restaurant.distance !== null && restaurant.distance !== undefined && (
            <span> · {formatDistance(restaurant.distance)}</span>
          )}
          {restaurant.review_count > 0 && (
            <span> · 리뷰 {restaurant.review_count}개</span>
          )}
        </div>
      </div>
    </div>
  );
};

const MapBottomSheet = ({ 
  restaurants = [],
  displayedCount = 20,
  onLoadMore,
  selectedRestaurant,
  onSelectRestaurant,
  onClose,
  sortBy = 'deepfm',
  onSortChange
}) => {
  // 3가지 모드: hint, list, detail
  const [sheetMode, setSheetMode] = useState('hint');
  const [snapIndex, setSnapIndex] = useState(0); // 0: 10%, 1: 50%, 2: 100%
  const sheetRef = useRef(null);

  // selectedRestaurant 변경 시 detail 모드로 전환 + 10%일 때 50%로 자동 확장
  useEffect(() => {
    console.log('🎯 [useEffect1] 트리거됨 - selectedRestaurant:', !!selectedRestaurant, 'snapIndex:', snapIndex, 'sheetMode:', sheetMode);
    
    if (selectedRestaurant) {
      setSheetMode('detail');
      // 10% 상태에서 마커 클릭 시 50%로 확장
      // ⚠️ sheetMode가 'hint'일 때만 자동 확장 (사용자가 수동으로 내린 경우 제외)
      if (snapIndex === 0 && sheetMode === 'hint' && sheetRef.current) {
        console.log('🚀 [useEffect1] 조건 충족! 50%로 자동 확장 예약');
        setTimeout(() => {
          console.log('🚀 [useEffect1] 50%로 확장 실행!');
          sheetRef.current.snapTo(({ snapPoints }) => snapPoints[1]);
        }, 100);
      } else {
        console.log('⛔ [useEffect1] 자동 확장 조건 불충족 - snapIndex:', snapIndex, 'sheetMode:', sheetMode);
      }
    } else if (sheetMode === 'detail') {
      // 선택 해제 시 list 모드로
      console.log('🔄 [useEffect1] 선택 해제 → list 모드');
      setSheetMode('list');
    }
  }, [selectedRestaurant, snapIndex, sheetMode]);

  // ResizeObserver로 snap 상태 감지
  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const height = entry.contentRect.height;
        const windowHeight = window.innerHeight;
        const ratio = height / windowHeight;
        
        // 🔍 디버깅: ratio 값 출력
        console.log('📏 [ResizeObserver] ratio:', ratio.toFixed(3), 'height:', height, 'windowHeight:', windowHeight);
        
        let newSnapIndex;
        // snap index 업데이트 (임계값 조정: 선택 매장 콘텐츠가 많을 때도 10%로 스냅되도록)
        if (ratio < 0.3) {
          newSnapIndex = 0; // 10%
          console.log('🔽 [ResizeObserver] ratio < 0.3 감지! → 10%로 스냅 시도');
          // 🔧 수정: ratio가 12%~25% 사이에서 멈춰있으면 강제로 10%로 스냅
          // (snapIndex 상태와 무관하게 실제 위치 기반으로 판단)
          if (ratio > 0.12 && ratio < 0.25 && sheetRef.current) {
            console.log('✅ [ResizeObserver] 중간에 멈춤 감지 → 강제 snapTo(10%) 실행!');
            sheetRef.current.snapTo(({ snapPoints }) => snapPoints[0]);
          } else if (snapIndex !== 0 && sheetRef.current) {
            console.log('✅ [ResizeObserver] snapTo(10%) 실행!');
            sheetRef.current.snapTo(({ snapPoints }) => snapPoints[0]);
          } else {
            console.log('⚠️ [ResizeObserver] snapTo 실행 안됨 - snapIndex:', snapIndex, 'ratio:', ratio.toFixed(3));
          }
        } else if (ratio < 0.7) {
          newSnapIndex = 1; // 50%
          console.log('🔽 [ResizeObserver] ratio 0.3~0.7 → 50% 상태');
        } else {
          newSnapIndex = 2; // 100%
          console.log('🔽 [ResizeObserver] ratio >= 0.7 → 100% 상태');
        }
        
        if (newSnapIndex !== snapIndex) {
          console.log('🔄 [ResizeObserver] snapIndex 변경:', snapIndex, '→', newSnapIndex);
        }
        setSnapIndex(newSnapIndex);
        
        // 🆕 detail 모드에서 10%로 드래그했을 때 선택 해제
        // 🔧 수정: snapIndex === 1 조건 제거 (빠른 드래그 시 조건을 놓치는 문제 해결)
        if (newSnapIndex === 0 && sheetMode === 'detail' && onClose) {
          console.log('🔽 [ResizeObserver] detail → 10% 드래그 감지 → 선택 해제');
          onClose(); // selectedRestaurant를 null로 만듦
        }
        
        // 🔥 10%일 때는 list/detail → hint로 전환
        if (newSnapIndex === 0 && (sheetMode === 'list' || sheetMode === 'detail')) {
          console.log('🔄 [ResizeObserver] sheetMode:', sheetMode, '→ hint');
          setSheetMode('hint');
        }
        // 🔥 snap이 50% 이상이고 hint 모드면 자동으로 list 모드로 전환
        else if (newSnapIndex >= 1 && sheetMode === 'hint') {
          console.log('🔄 [ResizeObserver] sheetMode: hint → list');
          setSheetMode('list');
        }
      }
    });

    const findSheetElement = () => {
      const selectors = [
        '[data-rsbs-overlay]',
        '[data-rsbs-scroll]',
        '[data-rsbs-root]',
      ];
      
      for (const selector of selectors) {
        const element = document.querySelector(selector);
        if (element && element.offsetHeight > 0) {
          observer.observe(element);
          return true;
        }
      }
      return false;
    };

    const timeout = setTimeout(() => {
      if (!findSheetElement()) {
        setTimeout(findSheetElement, 100);
      }
    }, 50);

    return () => {
      clearTimeout(timeout);
      observer.disconnect();
    };
  }, [sheetMode, snapIndex, onClose]); // sheetMode, snapIndex, onClose 변경 감지

  const handleHintClick = () => {
    setSheetMode('list');
  };

  const handleRestaurantClick = (restaurant) => {
    if (onSelectRestaurant) {
      onSelectRestaurant(restaurant);
    }
  };

  const handleSortChange = (newSortBy) => {
    if (onSortChange) {
      onSortChange(newSortBy);
    }
  };

  const handleTabSwitch = (mode) => {
    setSheetMode(mode);
    if (mode === 'list' && onClose) {
      onClose(); // 전체 리스트로 전환 시 선택 해제
    }
  };

  // 표시할 레스토랑 목록
  const displayedRestaurants = restaurants.slice(0, displayedCount);
  const hasMore = displayedCount < restaurants.length;
  const remainingCount = restaurants.length - displayedCount;

  // DeepFM과 Multi-Tower 점수 (detail 모드용)
  const deepfmScore = selectedRestaurant?.ai_prediction?.deepfm_rating || selectedRestaurant?.stars || 3.0;
  const multitowerScore = selectedRestaurant?.ai_prediction?.multitower_rating || selectedRestaurant?.ai_prediction?.deepfm_rating || selectedRestaurant?.stars || 3.0;

  // snap 상태 확인
  const isHintSnap = snapIndex === 0; // 10%
  const isHalfSnap = snapIndex === 1; // 50%
  const isFullSnap = snapIndex === 2; // 100%

  return (
    <BottomSheet
      ref={sheetRef}
      open={true}
      onDismiss={() => {
        // 0% 상태 방지: 시트가 닫히려고 하면 10%로 복원
        if (sheetRef.current) {
          setTimeout(() => {
            sheetRef.current.snapTo(({ snapPoints }) => snapPoints[0]);
          }, 50);
        }
        // detail 모드에서 닫으려고 하면 list 모드로 전환
        if (sheetMode === 'detail' && onClose) {
          onClose();
          setSheetMode('list');
        }
      }}
      defaultSnap={({ maxHeight }) => maxHeight * 0.1}
      snapPoints={({ maxHeight }) => [
        maxHeight * 0.1,  // 10% (힌트)
        maxHeight * 0.5,  // 50% (기본)
        maxHeight,        // 100% (전체)
      ]}
      blocking={false}
      expandOnContentDrag={true}
      className="map-bottom-sheet"
    >
      <div className={`bottom-sheet-content snap-${snapIndex === 0 ? '10' : snapIndex === 1 ? '50' : '100'}`}>
        {/* HINT 모드 (10%) */}
        {sheetMode === 'hint' && (
          <div className="sheet-hint" onClick={handleHintClick}>
            <p>⬆️ 주변 맛집 {restaurants.length}곳 보기</p>
          </div>
        )}

        {/* LIST 모드 (50%/100%) */}
        {sheetMode === 'list' && (
          <>
            <div className="sheet-header">
              <div className="header-left">
                <h3>📋 전체 리스트</h3>
                <p className="region-info">
                  📍 이 지역 {restaurants.length}곳
                </p>
              </div>
              <SortDropdown
                value={sortBy}
                onChange={handleSortChange}
                options={[
                  { value: 'deepfm', label: 'DeepFM 순' },
                  { value: 'multitower', label: 'Multi-Tower 순' },
                  { value: 'distance', label: '거리 순' },
                  { value: 'review_count', label: '리뷰 많은 순' }
                ]}
              />
            </div>

            <div className="restaurant-list">
              {restaurants.length === 0 ? (
                <div className="empty-state">
                  <p>📍 이 지역에는 레스토랑이 없습니다</p>
                  <p className="hint">지도를 이동하거나 줌아웃 해보세요</p>
                </div>
              ) : (
                <>
                  {displayedRestaurants.map(r => (
                    <RestaurantListItem
                      key={r.business_id}
                      restaurant={r}
                      onClick={() => handleRestaurantClick(r)}
                    />
                  ))}

                  {hasMore && (
                    <div style={{ textAlign: 'center', padding: '16px 0' }}>
                      <button 
                        className="review-load-more-link"
                        onClick={onLoadMore}
                      >
                        더보기 ({remainingCount}개 남음)
                      </button>
                    </div>
                  )}

                  {!hasMore && restaurants.length > 20 && (
                    <div className="list-end-message">
                      ✓ 전체 {restaurants.length}개 레스토랑 확인 완료
                    </div>
                  )}
                </>
              )}
            </div>
          </>
        )}

        {/* DETAIL 모드 (50%/100%) */}
        {sheetMode === 'detail' && selectedRestaurant && (
          <>
            {/* 탭 헤더 */}
            <div className="sheet-tabs">
              <button 
                className={`tab-btn ${sheetMode === 'detail' ? 'active' : ''}`}
                onClick={() => handleTabSwitch('detail')}
              >
                📍 선택 매장
              </button>
              <button 
                className={`tab-btn ${sheetMode === 'list' ? 'active' : ''}`}
                onClick={() => handleTabSwitch('list')}
              >
                📋 전체 리스트
              </button>
            </div>

            {/* 공통 헤더: 가게 이름 + AI 점수 */}
            <div 
              className="sheet-header-common"
              onTouchStart={(e) => e.stopPropagation()}
              onTouchMove={(e) => e.stopPropagation()}
              onTouchEnd={(e) => e.stopPropagation()}
              onTouchCancel={(e) => e.stopPropagation()}
            >
              <h2>{selectedRestaurant.name}</h2>
              <div className="ai-scores">
                <span 
                  className="score-badge deepfm"
                  style={{ backgroundColor: getMarkerColor(deepfmScore) }}
                >
                  DeepFM {deepfmScore.toFixed(1)}
                </span>
                <span 
                  className="score-badge multitower"
                  style={{ backgroundColor: getMarkerColor(multitowerScore) }}
                >
                  Multi {multitowerScore.toFixed(1)}
                </span>
              </div>
            </div>

            {/* 50% 전용 콘텐츠 */}
            <div className="content-50-only">
              <div className="restaurant-meta">
                <span className="category">{selectedRestaurant.categories}</span>
                {selectedRestaurant.review_count > 0 && (
                  <span className="review-count"> · 리뷰 {selectedRestaurant.review_count}개</span>
                )}
              </div>

              {selectedRestaurant.address && (
                <div className="restaurant-address">
                  📍 {selectedRestaurant.address}
                </div>
              )}

              {selectedRestaurant.distance !== null && selectedRestaurant.distance !== undefined && (
                <div className="restaurant-distance">
                  🚶 {formatDistance(selectedRestaurant.distance)}
                </div>
              )}

              <div className="photo-placeholder">
                사진 없음
              </div>

              <div className="action-buttons">
                <button 
                  className="action-btn"
                  onClick={() => window.open(`https://map.kakao.com/link/to/${selectedRestaurant.name},${selectedRestaurant.latitude},${selectedRestaurant.longitude}`, '_blank')}
                >
                  🚗 길찾기
                </button>
                <button 
                  className="action-btn"
                  onClick={() => {
                    if (selectedRestaurant.phone) {
                      window.location.href = `tel:${selectedRestaurant.phone}`;
                    } else {
                      toast.dismiss();
                      toast('전화번호 정보가 없습니다');
                    }
                  }}
                >
                  📞 전화
                </button>
              </div>
            </div>

            {/* 100% 전용 콘텐츠: 탭 */}
            <div className="content-100-only">
              <Tabs>
                <TabList>
                  <Tab>홈</Tab>
                  <Tab>메뉴</Tab>
                  <Tab>리뷰</Tab>
                  <Tab>사진</Tab>
                </TabList>

                <TabPanel>
                  <HomeTab restaurant={selectedRestaurant} />
                </TabPanel>

                <TabPanel>
                  <MenuTab />
                </TabPanel>

                <TabPanel>
                  <ReviewTab businessId={selectedRestaurant.business_id} />
                </TabPanel>

                <TabPanel>
                  <PhotoTab />
                </TabPanel>
              </Tabs>
            </div>
          </>
        )}
      </div>
    </BottomSheet>
  );
};

export default MapBottomSheet;
