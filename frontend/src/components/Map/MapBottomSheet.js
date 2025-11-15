import React, { useState, useEffect } from 'react';
import { BottomSheet } from 'react-spring-bottom-sheet';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-spring-bottom-sheet/dist/style.css';
import 'react-tabs/style/react-tabs.css';
import './Map.css';
import HomeTab from './tabs/HomeTab';
import MenuTab from './tabs/MenuTab';
import ReviewTab from './tabs/ReviewTab';
import PhotoTab from './tabs/PhotoTab';

const MapBottomSheet = ({ restaurant, onClose, initialSnap = 0.5 }) => {
  // Hook은 항상 최상단에서 호출
  const [snapIndex, setSnapIndex] = useState(initialSnap === 1.0 ? 1 : 0); // 0: 50%, 1: 100%

  // 조건부 렌더링은 Hook 호출 이후에
  if (!restaurant) return null;

  // DeepFM과 Multi-Tower 점수 추출
  const deepfmScore = restaurant.ai_prediction || restaurant.stars || 0;
  const multitowerScore = restaurant.multitower_rating || deepfmScore;

  // 50% 상태인지 확인
  const isHalfSnap = snapIndex === 0;

  return (
    <BottomSheet
      open={!!restaurant}
      onDismiss={onClose}
      defaultSnap={({ maxHeight }) => maxHeight * initialSnap}
      snapPoints={({ maxHeight }) => [
        maxHeight * 0.5, // 50% (기본)
        maxHeight, // 100% (전체 화면)
      ]}
      onSpringEnd={(event) => {
        // 이벤트 디버깅
        console.log('onSpringEnd event:', event);
        if (event.type === 'SNAP') {
          // spring의 현재 값으로 snap 상태 판단
          const height = event.spring?.get();
          console.log('Current height:', height);
          // 높이가 70% 이상이면 100% snap으로 간주
          if (height && typeof height === 'number') {
            setSnapIndex(height > 0.7 ? 1 : 0);
          }
        }
      }}
      blocking={false}
      expandOnContentDrag={true}
      header={false} // 자동 헤더 비활성화
      className="map-bottom-sheet"
    >
      <div className={`bottom-sheet-content ${isHalfSnap ? 'snap-50' : 'snap-100'}`}>
        {/* 공통 헤더: 가게 이름 + AI 점수 (항상 표시) */}
        <div className="sheet-header-common">
          <h2>{restaurant.name}</h2>
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
            <span className="category">{restaurant.categories}</span>
            {restaurant.review_count && (
              <span className="review-count"> · 리뷰 {restaurant.review_count}개</span>
            )}
          </div>

          {restaurant.address && (
            <div className="restaurant-address">
              📍 {restaurant.address}
            </div>
          )}

          <div className="photo-placeholder">
            사진 없음
          </div>

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

