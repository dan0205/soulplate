import React, { useState, useEffect, useRef } from 'react';
import { BottomSheet } from 'react-spring-bottom-sheet';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-spring-bottom-sheet/dist/style.css';
import 'react-tabs/style/react-tabs.css';
import './Map.css';
import HomeTab from './tabs/HomeTab';
import MenuTab from './tabs/MenuTab';
import ReviewTab from './tabs/ReviewTab';
import PhotoTab from './tabs/PhotoTab';
import SortDropdown from './SortDropdown';
import { formatDistance } from '../../utils/distance';

const RestaurantListItem = ({ restaurant, onClick }) => {
  const aiScore = restaurant.ai_prediction?.deepfm_rating || restaurant.stars || 0;
  
  // AI 점수 기반 색상
  const getMarkerColor = (score) => {
    if (score >= 4.5) return '#FF4444';
    if (score >= 4.0) return '#FF8800';
    if (score >= 3.5) return '#FFD700';
    return '#CCCCCC';
  };

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
            <span className="badge-deepfm">
              {restaurant.ai_prediction?.deepfm_rating?.toFixed(1) || restaurant.stars.toFixed(1)}
            </span>
          </div>
        </div>
        <div className="list-item-meta">
          <span>{restaurant.categories}</span>
          {restaurant.distance !== null && restaurant.distance !== undefined && (
            <span> · {formatDistance(restaurant.distance)}</span>
          )}
          <span> · 리뷰 {restaurant.review_count || 0}개</span>
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

  // selectedRestaurant 변경 시 detail 모드로 전환
  useEffect(() => {
    if (selectedRestaurant) {
      setSheetMode('detail');
    } else if (sheetMode === 'detail') {
      // 선택 해제 시 list 모드로
      setSheetMode('list');
    }
  }, [selectedRestaurant]);

  // ResizeObserver로 snap 상태 감지
  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const height = entry.contentRect.height;
        const windowHeight = window.innerHeight;
        const ratio = height / windowHeight;
        
        // snap index 업데이트
        if (ratio < 0.2) {
          setSnapIndex(0); // 10%
        } else if (ratio < 0.7) {
          setSnapIndex(1); // 50%
        } else {
          setSnapIndex(2); // 100%
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
  }, [sheetMode]);

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
  const deepfmScore = selectedRestaurant?.ai_prediction?.deepfm_rating || selectedRestaurant?.stars || 0;
  const multitowerScore = selectedRestaurant?.ai_prediction?.multitower_rating || selectedRestaurant?.ai_prediction?.deepfm_rating || selectedRestaurant?.stars || 0;

  // snap 상태 확인
  const isHintSnap = snapIndex === 0; // 10%
  const isHalfSnap = snapIndex === 1; // 50%
  const isFullSnap = snapIndex === 2; // 100%

  return (
    <BottomSheet
      open={true}
      onDismiss={() => {
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
            <div className="drag-handle"></div>
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
                    <button className="btn-load-more" onClick={onLoadMore}>
                      더보기 ({remainingCount}개 남음)
                    </button>
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
            <div className="sheet-header-common">
              <h2>{selectedRestaurant.name}</h2>
              <div className="ai-scores">
                <span className="score-badge deepfm">
                  DeepFM {deepfmScore.toFixed(1)}
                </span>
                <span className="score-badge multitower">
                  Multi {multitowerScore.toFixed(1)}
                </span>
              </div>
            </div>

            {/* 50% 전용 콘텐츠 */}
            <div className="content-50-only">
              <div className="restaurant-meta">
                <span className="category">{selectedRestaurant.categories}</span>
                {selectedRestaurant.review_count && (
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
                  onClick={() => alert('전화번호 준비 중입니다')}
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
                  <ReviewTab businessId={selectedRestaurant.id} />
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
