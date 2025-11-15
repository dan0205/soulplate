import React from 'react';
import { BottomSheet } from 'react-spring-bottom-sheet';
import 'react-spring-bottom-sheet/dist/style.css';
import './Map.css';

const MapBottomSheet = ({ restaurant, onClose }) => {
  if (!restaurant) return null;

  const getMarkerColor = (aiScore) => {
    if (aiScore >= 4.5) return '#FF4444';
    if (aiScore >= 4.0) return '#FF8800';
    if (aiScore >= 3.5) return '#FFD700';
    return '#CCCCCC';
  };

  const aiScore = restaurant.ai_prediction || restaurant.stars || 0;
  const markerColor = getMarkerColor(aiScore);

  return (
    <BottomSheet
      open={!!restaurant}
      onDismiss={onClose}
      defaultSnap={({ maxHeight }) => maxHeight * 0.5}
      snapPoints={({ maxHeight }) => [
        maxHeight * 0.5, // 50% (기본)
        maxHeight * 0.95, // 95% (거의 전체)
      ]}
      blocking={false}
      expandOnContentDrag={true}
      className="map-bottom-sheet"
    >
      <div className="bottom-sheet-content">
        {/* 드래그 핸들 */}
        <div className="bottom-sheet-handle" />

        {/* 레스토랑 정보 */}
        <div className="bottom-sheet-header">
          <h2>{restaurant.name}</h2>
          <div className="restaurant-badges">
            <span 
              className="ai-score-badge" 
              style={{ backgroundColor: markerColor }}
            >
              AI {aiScore.toFixed(1)}
            </span>
            <span className="stars-badge">
              ⭐ {(restaurant.stars || 0).toFixed(1)}
            </span>
          </div>
        </div>

        <div className="bottom-sheet-body">
          {/* 기본 정보 */}
          <div className="restaurant-info">
            {restaurant.address && (
              <div className="info-row">
                <span className="info-icon">📍</span>
                <span className="info-text">{restaurant.address}</span>
              </div>
            )}
            {restaurant.categories && (
              <div className="info-row">
                <span className="info-icon">🍽️</span>
                <span className="info-text">{restaurant.categories}</span>
              </div>
            )}
            {restaurant.review_count && (
              <div className="info-row">
                <span className="info-icon">💬</span>
                <span className="info-text">리뷰 {restaurant.review_count}개</span>
              </div>
            )}
          </div>

          {/* 액션 버튼 */}
          <div className="action-buttons">
            <button 
              className="action-btn"
              onClick={() => window.open(`https://map.kakao.com/link/to/${restaurant.name},${restaurant.latitude},${restaurant.longitude}`, '_blank')}
            >
              🚗 길찾기
            </button>
            <button 
              className="action-btn primary"
              onClick={() => window.open(`/business/${restaurant.id}`, '_blank')}
            >
              📋 상세보기 (새 탭)
            </button>
          </div>

          {/* ABSA 특징 (있는 경우) */}
          {(restaurant.absa_food_avg || restaurant.absa_service_avg || restaurant.absa_atmosphere_avg) && (
            <div className="absa-features">
              <h3>리뷰 분석</h3>
              <div className="feature-bars">
                {restaurant.absa_food_avg && (
                  <div className="feature-bar">
                    <span className="feature-label">음식 맛</span>
                    <div className="feature-progress">
                      <div 
                        className="feature-fill" 
                        style={{ width: `${(restaurant.absa_food_avg + 1) * 50}%` }}
                      />
                    </div>
                    <span className="feature-value">{restaurant.absa_food_avg.toFixed(1)}</span>
                  </div>
                )}
                {restaurant.absa_service_avg && (
                  <div className="feature-bar">
                    <span className="feature-label">서비스</span>
                    <div className="feature-progress">
                      <div 
                        className="feature-fill" 
                        style={{ width: `${(restaurant.absa_service_avg + 1) * 50}%` }}
                      />
                    </div>
                    <span className="feature-value">{restaurant.absa_service_avg.toFixed(1)}</span>
                  </div>
                )}
                {restaurant.absa_atmosphere_avg && (
                  <div className="feature-bar">
                    <span className="feature-label">분위기</span>
                    <div className="feature-progress">
                      <div 
                        className="feature-fill" 
                        style={{ width: `${(restaurant.absa_atmosphere_avg + 1) * 50}%` }}
                      />
                    </div>
                    <span className="feature-value">{restaurant.absa_atmosphere_avg.toFixed(1)}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </BottomSheet>
  );
};

export default MapBottomSheet;

