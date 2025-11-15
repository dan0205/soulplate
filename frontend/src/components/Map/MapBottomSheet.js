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
  if (!restaurant) return null;

  const [snapIndex, setSnapIndex] = useState(0); // 0: 50%, 1: 100%

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
        if (event.type === 'SNAP') {
          // snapIndex 계산: 50%면 0, 100%면 1
          const currentHeight = event.source;
          setSnapIndex(currentHeight > 0.7 ? 1 : 0);
        }
      }}
      blocking={false}
      expandOnContentDrag={true}
      header={false} // 자동 헤더 비활성화
      className="map-bottom-sheet"
    >
      <div className="bottom-sheet-content">
        {isHalfSnap ? (
          /* 50% 카드: 전체 정보 표시 */
          <div className="sheet-50-content">
            <div className="restaurant-name">
              <h2>{restaurant.name}</h2>
            </div>

            <div className="ai-scores">
              <span className="score-badge deepfm">
                DeepFM {deepfmScore.toFixed(1)}
              </span>
              <span className="score-badge multitower">
                Multi {multitowerScore.toFixed(1)}
              </span>
            </div>

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
        ) : (
          /* 100% 카드: 간소화된 헤더 + 탭 */
          <div className="sheet-100-content">
            <div className="sheet-header-minimal">
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
        )}
      </div>
    </BottomSheet>
  );
};

export default MapBottomSheet;

