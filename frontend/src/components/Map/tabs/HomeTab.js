import React from 'react';
import toast from 'react-hot-toast';

const HomeTab = ({ restaurant }) => {
  return (
    <div className="home-tab">
      {/* 액션 버튼 */}
      <div className="action-buttons">
        <button 
          className="action-btn"
          onClick={() => window.open(`https://map.kakao.com/link/to/${restaurant.name},${restaurant.latitude},${restaurant.longitude}`, '_blank')}
        >
          🚗 길찾기
        </button>
        <button 
          className="action-btn"
          onClick={() => {
            toast.dismiss();
            toast('전화번호 준비 중입니다');
          }}
        >
          📞 전화
        </button>
      </div>

      {/* 기본 정보 */}
      <div className="basic-info-section">
        <div className="info-item">
          <div className="info-label">📍 주소</div>
          <div className="info-value">{restaurant.address || '정보 없음'}</div>
        </div>

        <div className="info-item">
          <div className="info-label">🕐 영업시간</div>
          <div className="info-value">준비 중입니다</div>
        </div>

        <div className="info-item">
          <div className="info-label">📞 전화번호</div>
          <div className="info-value">준비 중입니다</div>
        </div>
      </div>

      {/* AI 브리핑 (ABSA) */}
      {(restaurant.absa_food_avg != null || restaurant.absa_service_avg != null || restaurant.absa_atmosphere_avg != null) && (
        <div className="absa-features">
          <h3>🤖 AI 브리핑</h3>
          <div className="feature-bars">
            {restaurant.absa_food_avg != null && (
              <div className="feature-bar">
                <span className="feature-label">🍜 음식 맛</span>
                <div className="feature-progress">
                  <div 
                    className="feature-fill" 
                    style={{ width: `${(restaurant.absa_food_avg + 1) * 50}%` }}
                  />
                </div>
                <span className="feature-value">{restaurant.absa_food_avg.toFixed(1)}</span>
              </div>
            )}
            {restaurant.absa_service_avg != null && (
              <div className="feature-bar">
                <span className="feature-label">👨‍🍳 서비스</span>
                <div className="feature-progress">
                  <div 
                    className="feature-fill" 
                    style={{ width: `${(restaurant.absa_service_avg + 1) * 50}%` }}
                  />
                </div>
                <span className="feature-value">{restaurant.absa_service_avg.toFixed(1)}</span>
              </div>
            )}
            {restaurant.absa_atmosphere_avg != null && (
              <div className="feature-bar">
                <span className="feature-label">🏠 분위기</span>
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
  );
};

export default HomeTab;

