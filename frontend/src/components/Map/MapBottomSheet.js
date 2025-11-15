import React from 'react';
import { BottomSheet } from 'react-spring-bottom-sheet';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-spring-bottom-sheet/dist/style.css';
import 'react-tabs/style/react-tabs.css';
import './Map.css';
import HomeTab from './tabs/HomeTab';
import MenuTab from './tabs/MenuTab';
import ReviewTab from './tabs/ReviewTab';
import PhotoTab from './tabs/PhotoTab';

const MapBottomSheet = ({ restaurant, onClose }) => {
  if (!restaurant) return null;

  // DeepFM과 Multi-Tower 점수 추출
  const deepfmScore = restaurant.ai_prediction || restaurant.stars || 0;
  const multitowerScore = restaurant.multitower_rating || deepfmScore;

  const getMarkerColor = (aiScore) => {
    if (aiScore >= 4.5) return '#FF4444';
    if (aiScore >= 4.0) return '#FF8800';
    if (aiScore >= 3.5) return '#FFD700';
    return '#CCCCCC';
  };

  return (
    <BottomSheet
      open={!!restaurant}
      onDismiss={onClose}
      defaultSnap={({ maxHeight }) => maxHeight * 0.5}
      snapPoints={({ maxHeight }) => [
        maxHeight * 0.5, // 50% (기본)
        maxHeight, // 100% (전체 화면)
      ]}
      blocking={false}
      expandOnContentDrag={true}
      className="map-bottom-sheet"
    >
      <div className="bottom-sheet-content">
        {/* 드래그 핸들 */}
        <div className="bottom-sheet-handle" />

        {/* 고정 헤더: 음식점 이름 */}
        <div className="sheet-header-fixed">
          <div className="restaurant-name">
            <h2>{restaurant.name}</h2>
          </div>

          {/* AI 점수 표시 */}
          <div className="ai-scores">
            <span className="score-badge deepfm">
              DeepFM {deepfmScore.toFixed(1)}
            </span>
            <span className="score-badge multitower">
              Multi {multitowerScore.toFixed(1)}
            </span>
          </div>

          {/* 카테고리 + 리뷰 수 */}
          <div className="restaurant-meta">
            <span className="category">{restaurant.categories}</span>
            {restaurant.review_count && (
              <span className="review-count"> · 리뷰 {restaurant.review_count}개</span>
            )}
          </div>

          {/* 주소 */}
          {restaurant.address && (
            <div className="restaurant-address">
              📍 {restaurant.address}
            </div>
          )}

          {/* 빈 사진 영역 */}
          <div className="photo-placeholder">
            사진 없음
          </div>

          {/* 액션 버튼 (50% 상태용) */}
          <div className="action-buttons">
            <button 
              className="action-btn"
              onClick={() => window.open(`https://map.kakao.com/link/to/${restaurant.name},${restaurant.latitude},${restaurant.longitude}`, '_blank')}
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

        {/* 100% 카드: 탭 구조 */}
        <div className="sheet-content-scroll">
          <Tabs>
            <TabList>
              <Tab>홈</Tab>
              <Tab>메뉴</Tab>
              <Tab>리뷰</Tab>
              <Tab>사진</Tab>
            </TabList>

            <TabPanel>
              <HomeTab restaurant={restaurant} />
            </TabPanel>

            <TabPanel>
              <MenuTab />
            </TabPanel>

            <TabPanel>
              <ReviewTab businessId={restaurant.id} />
            </TabPanel>

            <TabPanel>
              <PhotoTab />
            </TabPanel>
          </Tabs>
        </div>
      </div>
    </BottomSheet>
  );
};

export default MapBottomSheet;

