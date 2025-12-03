/**
 * 취향 테스트 결과 페이지 (둘러보기 모드용)
 * - 결과를 location.state에서 받아서 표시
 * - 저장되지 않음 안내
 * - 다른 취향 탐색하기 버튼
 */

import React, { useState } from 'react';
import ReactDOM from 'react-dom';
import { useNavigate, useLocation } from 'react-router-dom';
import { getMBTIInfo, MBTI_TYPE_DESCRIPTIONS } from '../utils/mbtiDescriptions';
import './MBTIDetailPage.css';
import './Profile.css';

const TasteTestResultPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { result, testType, isDemo } = location.state || {};
  
  const [showOtherTypes, setShowOtherTypes] = useState(false);
  const [showTypeModal, setShowTypeModal] = useState(false);
  const [selectedType, setSelectedType] = useState(null);

  // 결과가 없으면 홈으로 리다이렉트
  if (!result) {
    navigate('/', { replace: true });
    return null;
  }

  const mbtiInfo = getMBTIInfo(result.mbti_type);
  const axisScores = result.axis_scores;
  const otherTypes = Object.keys(MBTI_TYPE_DESCRIPTIONS).filter(
    type => type !== result.mbti_type
  );

  const toggleOtherTypes = () => {
    setShowOtherTypes(!showOtherTypes);
  };

  const openTypeModal = (typeCode) => {
    setSelectedType(typeCode);
    setShowTypeModal(true);
  };

  const closeTypeModal = () => {
    setShowTypeModal(false);
    setSelectedType(null);
  };

  return (
    <div className="mbti-detail-container">
      <div className="profile-header-actions">
        <div className="profile-logo" onClick={() => navigate('/')}>
          Soulplate
        </div>
      </div>

      {/* 저장 안 됨 안내 (데모 모드) */}
      {isDemo && (
        <div className="demo-notice" style={{
          background: '#fff3cd',
          border: '1px solid #ffc107',
          borderRadius: '12px',
          padding: '16px',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          color: '#856404'
        }}>
          <span style={{ fontSize: '24px' }}>⚠️</span>
          <div style={{ flex: 1, fontSize: '14px', lineHeight: '1.5' }}>
            <strong>둘러보기 모드에서는 결과가 저장되지 않습니다.</strong><br />
            로그인하면 결과를 저장하고 AI 추천을 받을 수 있습니다.
          </div>
        </div>
      )}

      {/* MBTI 기본 정보 카드 */}
      <div className="mbti-card-detailed">
        <div className="mbti-card-header-detailed">
          <div className="mbti-type-badge">{result.mbti_type}</div>
          <div className="mbti-type-title">
            <span className="mbti-emoji">{mbtiInfo.emoji || '🍽️'}</span>
            <span className="mbti-name">{mbtiInfo.name}</span>
          </div>
          {mbtiInfo.catchphrase && (
            <div className="mbti-catchphrase">"{mbtiInfo.catchphrase}"</div>
          )}
          <div className="mbti-description">{mbtiInfo.description}</div>
        </div>
        
        <div className="mbti-card-body-detailed">
          {mbtiInfo.recommend && mbtiInfo.recommend.length > 0 && (
            <div className="mbti-info-section">
              <div className="mbti-info-title mbti-recommend">
                <i className="fas fa-thumbs-up"></i> 👍 추천 메뉴 & 장소
              </div>
              <div className="mbti-info-content">
                <ul>
                  {mbtiInfo.recommend.map((rec, idx) => (
                    <li key={idx} dangerouslySetInnerHTML={{ __html: rec.replace(': ', ':</strong> ').replace(/^([^:]+):/, '<strong>$1:</strong>') }} />
                  ))}
                </ul>
              </div>
            </div>
          )}
          
          {mbtiInfo.avoid && mbtiInfo.avoid.length > 0 && (
            <div className="mbti-info-section">
              <div className="mbti-info-title mbti-avoid">
                <i className="fas fa-ban"></i> 🚫 피해야 할 식당
              </div>
              <div className="mbti-info-content">
                <ul>
                  {mbtiInfo.avoid.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 4개 축 확률 분석 */}
      {axisScores ? (
        <div className="probability-view">
          <h3 className="probability-title">내 음식 성향 분석표</h3>
          
          <div className="trait-group trait-flavor">
            <div className="trait-info">
              <span>맛의 강도</span>
              <span className="trait-percentage highlight">
                {axisScores.flavor_intensity.S}% 강렬함
              </span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${axisScores.flavor_intensity.S}%` }}></div>
              <div className="bar-circle" style={{ left: `${axisScores.flavor_intensity.S}%` }}></div>
            </div>
            <div className="trait-labels">
              <span className="label-left">강렬함 (Strong)</span>
              <span className="label-right">부드러움 (Mild)</span>
            </div>
          </div>

          <div className="trait-group trait-env">
            <div className="trait-info">
              <span>식사 환경</span>
              <span className="trait-percentage highlight">
                {axisScores.dining_environment.A}% 분위기
              </span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${axisScores.dining_environment.A}%` }}></div>
              <div className="bar-circle" style={{ left: `${axisScores.dining_environment.A}%` }}></div>
            </div>
            <div className="trait-labels">
              <span className="label-left">분위기 (Ambiance)</span>
              <span className="label-right">효율 (Optimized)</span>
            </div>
          </div>

          <div className="trait-group trait-price">
            <div className="trait-info">
              <span>가격 민감도</span>
              <span className="trait-percentage highlight">
                {axisScores.price_sensitivity.P}% 프리미엄
              </span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${axisScores.price_sensitivity.P}%` }}></div>
              <div className="bar-circle" style={{ left: `${axisScores.price_sensitivity.P}%` }}></div>
            </div>
            <div className="trait-labels">
              <span className="label-left">프리미엄 (Premium)</span>
              <span className="label-right">가성비 (Cost-effective)</span>
            </div>
          </div>

          <div className="trait-group trait-social">
            <div className="trait-info">
              <span>동행 선호도</span>
              <span className="trait-percentage highlight">
                {axisScores.dining_company.O}% 혼자
              </span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${100 - axisScores.dining_company.O}%` }}></div>
              <div className="bar-circle" style={{ left: `${100 - axisScores.dining_company.O}%` }}></div>
            </div>
            <div className="trait-labels">
              <span className="label-left">함께 (Together)</span>
              <span className="label-right">혼자 (Solo)</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="probability-view">
          <h3 className="probability-title">내 음식 성향 분석표</h3>
          <div className="no-axis-data">
            <p style={{ textAlign: 'center', color: '#666', padding: '40px 20px' }}>
              확률 분석 데이터가 없습니다.
            </p>
          </div>
        </div>
      )}

      {/* 다른 취향 탐색하기 버튼 */}
      <button className="btn-explore-types-main" onClick={toggleOtherTypes}>
        🔍 다른 취향 탐색하기
      </button>

      {/* 16개 타입 그리드 */}
      <div className={`other-types-grid ${showOtherTypes ? 'show' : ''}`}>
        {otherTypes.map((typeCode) => {
          const typeInfo = getMBTIInfo(typeCode);
          return (
            <div
              key={typeCode}
              className="other-type-card"
              onClick={() => openTypeModal(typeCode)}
            >
              <div className="other-type-code">{typeCode}</div>
              <div className="other-type-name">{typeInfo.name}</div>
            </div>
          );
        })}
      </div>

      {/* 타입 상세 모달 */}
      {showTypeModal && selectedType && ReactDOM.createPortal(
        <div 
          className={`type-detail-modal ${showTypeModal ? 'show' : ''}`}
          onClick={(e) => {
            if (e.target.classList.contains('type-detail-modal')) {
              closeTypeModal();
            }
          }}
        >
          <div className="type-detail-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeTypeModal}>×</button>
            <div className="modal-mbti-box">
              <div className="modal-mbti-header">
                <div className="modal-mbti-type">{selectedType}</div>
                <div className="modal-mbti-title">
                  <span className="modal-mbti-emoji">{getMBTIInfo(selectedType).emoji || '🍽️'}</span>
                  <span className="modal-mbti-name">{getMBTIInfo(selectedType).name}</span>
                </div>
                {getMBTIInfo(selectedType).catchphrase && (
                  <div className="modal-mbti-catchphrase">"{getMBTIInfo(selectedType).catchphrase}"</div>
                )}
                <div className="modal-mbti-description">
                  {getMBTIInfo(selectedType).description}
                </div>
              </div>
              
              {getMBTIInfo(selectedType).recommend && getMBTIInfo(selectedType).recommend.length > 0 && (
                <div className="modal-info-section">
                  <div className="modal-info-title modal-recommend">
                    👍 추천 메뉴 & 장소
                  </div>
                  <div className="modal-info-content">
                    <ul>
                      {getMBTIInfo(selectedType).recommend.map((rec, idx) => (
                        <li key={idx} dangerouslySetInnerHTML={{ __html: rec.replace(': ', ':</strong> ').replace(/^([^:]+):/, '<strong>$1:</strong>') }} />
                      ))}
                    </ul>
                  </div>
                </div>
              )}
              
              {getMBTIInfo(selectedType).avoid && getMBTIInfo(selectedType).avoid.length > 0 && (
                <div className="modal-info-section">
                  <div className="modal-info-title modal-avoid">
                    🚫 피해야 할 식당
                  </div>
                  <div className="modal-info-content">
                    <ul>
                      {getMBTIInfo(selectedType).avoid.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
};

export default TasteTestResultPage;
