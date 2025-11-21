/**
 * MBTI 상세 페이지
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { userAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { getMBTIInfo, MBTI_TYPE_DESCRIPTIONS } from '../utils/mbtiDescriptions';
import './MBTIDetailPage.css';
import './Profile.css';

const MBTIDetailPage = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedAxis, setSelectedAxis] = useState(null);
  const [showOtherTypes, setShowOtherTypes] = useState(false);
  const [showTypeModal, setShowTypeModal] = useState(false);
  const [selectedType, setSelectedType] = useState(null);
  const [showRetestOptions, setShowRetestOptions] = useState(false);

  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    try {
      const response = await userAPI.getMyProfile();
      setProfile(response.data);
    } catch (err) {
      console.error('Failed to load profile:', err);
      toast.error('프로필을 불러오는데 실패했습니다.');
    } finally {
      setLoading(false);
    }
  };

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

  const handleStartQuickTest = () => {
    setShowRetestOptions(false);
    navigate('/taste-test', { state: { testType: 'quick' } });
  };

  const handleStartDeepTest = () => {
    setShowRetestOptions(false);
    navigate('/taste-test', { state: { testType: 'deep' } });
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(window.location.origin + '/profile/mbti');
    toast.success('링크가 복사되었습니다!');
  };

  // 로딩 상태
  if (loading) {
    return (
      <div className="mbti-detail-container">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>로딩 중...</p>
        </div>
      </div>
    );
  }

  // 미완료 사용자 처리
  if (!profile?.taste_test_completed) {
    return (
      <div className="mbti-detail-container">
        <div className="profile-header-actions">
          <div className="profile-logo" onClick={() => navigate('/')}>
            Soulplate
          </div>
        </div>
        <div className="empty-state-cta">
          <h2>아직 음식 MBTI 테스트를 하지 않으셨네요!</h2>
          <p>간단한 질문으로 당신의 음식 취향을 알아보세요</p>
          <div className="empty-state-buttons">
            <button 
              className="btn-start-test" 
              onClick={() => navigate('/taste-test', { state: { testType: 'quick' } })}
            >
              ⚡ 간단 테스트 시작
            </button>
            <button 
              className="btn-start-test" 
              onClick={() => navigate('/taste-test', { state: { testType: 'deep' } })}
            >
              🔍 심화 테스트 시작
            </button>
          </div>
        </div>
      </div>
    );
  }

  const mbtiInfo = getMBTIInfo(profile.taste_test_mbti_type);
  const otherTypes = Object.keys(MBTI_TYPE_DESCRIPTIONS).filter(
    type => type !== profile?.taste_test_mbti_type
  );

  return (
    <div className="mbti-detail-container">
      <div className="profile-header-actions">
        <div className="profile-logo" onClick={() => navigate('/')}>
          Soulplate
        </div>
      </div>

      {/* 뒤로가기 버튼 */}
      <button className="btn-back-detail" onClick={() => navigate(-1)}>
        <i className="fas fa-arrow-left"></i> 뒤로가기
      </button>

      {/* MBTI 기본 정보 카드 */}
      <div className="mbti-card-detailed">
        <div className="mbti-card-header-detailed">
          <div className="mbti-type-badge">{profile.taste_test_mbti_type}</div>
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
                <i className="fas fa-thumbs-up"></i> 추천 메뉴 & 장소
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
                <i className="fas fa-ban"></i> 피해야 할 식당
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

      {/* 공유 버튼 */}
      <div className="share-buttons">
        <button className="btn-share" onClick={copyToClipboard}>
          <i className="fas fa-link"></i> URL 복사
        </button>
      </div>

      {/* 4개 축 확률 분석 */}
      {profile.taste_test_axis_scores && (
        <div className="probability-view">
          <h3 className="probability-title">🎯 내 음식 취향 비율 분석</h3>
          
          <div 
            className={`axis-item ${selectedAxis === 'flavor_intensity' ? 'expanded' : ''}`}
            onClick={() => setSelectedAxis(selectedAxis === 'flavor_intensity' ? null : 'flavor_intensity')}
          >
            <div className="axis-name">맛 강도 (Flavor Intensity)</div>
            <div className="axis-bar-container">
              <div 
                className="axis-left" 
                style={{ width: `${profile.taste_test_axis_scores.flavor_intensity.S}%` }}
              >
                S {profile.taste_test_axis_scores.flavor_intensity.S}%
              </div>
              <div 
                className="axis-right" 
                style={{ width: `${profile.taste_test_axis_scores.flavor_intensity.M}%` }}
              >
                M {profile.taste_test_axis_scores.flavor_intensity.M}%
              </div>
            </div>
            <div className="axis-labels">
              <span>강렬한 맛 (Strong)</span>
              <span>부드러운 맛 (Mild)</span>
            </div>
            {selectedAxis === 'flavor_intensity' && (
              <div className="axis-detail-expanded">
                <p>
                  {profile.taste_test_axis_scores.flavor_intensity.S >= 50
                    ? "강렬한 맛을 선호하며 맵고 짠 자극적인 음식을 즐깁니다. 순한 맛보다는 개성 있고 강한 풍미를 추구합니다."
                    : "부드럽고 담백한 맛을 선호하며 건강한 식단을 중시합니다. 자극적인 음식보다는 은은하고 섬세한 맛을 좋아합니다."}
                </p>
              </div>
            )}
          </div>

          <div 
            className={`axis-item ${selectedAxis === 'dining_environment' ? 'expanded' : ''}`}
            onClick={() => setSelectedAxis(selectedAxis === 'dining_environment' ? null : 'dining_environment')}
          >
            <div className="axis-name">식사 환경 (Dining Environment)</div>
            <div className="axis-bar-container">
              <div 
                className="axis-left" 
                style={{ width: `${profile.taste_test_axis_scores.dining_environment.A}%` }}
              >
                A {profile.taste_test_axis_scores.dining_environment.A}%
              </div>
              <div 
                className="axis-right" 
                style={{ width: `${profile.taste_test_axis_scores.dining_environment.O}%` }}
              >
                O {profile.taste_test_axis_scores.dining_environment.O}%
              </div>
            </div>
            <div className="axis-labels">
              <span>분위기 중시 (Atmosphere)</span>
              <span>효율 중시 (Optimized)</span>
            </div>
            {selectedAxis === 'dining_environment' && (
              <div className="axis-detail-expanded">
                <p>
                  {profile.taste_test_axis_scores.dining_environment.A >= 50
                    ? "식사 공간의 분위기와 인테리어를 중요하게 생각합니다. 감성적이고 아름다운 공간에서 식사하는 것을 선호합니다."
                    : "식사의 효율성과 실용성을 중시합니다. 빠르고 편리하게 맛있는 음식을 먹는 것이 중요합니다."}
                </p>
              </div>
            )}
          </div>

          <div 
            className={`axis-item ${selectedAxis === 'price_sensitivity' ? 'expanded' : ''}`}
            onClick={() => setSelectedAxis(selectedAxis === 'price_sensitivity' ? null : 'price_sensitivity')}
          >
            <div className="axis-name">가격 민감도 (Price Sensitivity)</div>
            <div className="axis-bar-container">
              <div 
                className="axis-left" 
                style={{ width: `${profile.taste_test_axis_scores.price_sensitivity.P}%` }}
              >
                P {profile.taste_test_axis_scores.price_sensitivity.P}%
              </div>
              <div 
                className="axis-right" 
                style={{ width: `${profile.taste_test_axis_scores.price_sensitivity.C}%` }}
              >
                C {profile.taste_test_axis_scores.price_sensitivity.C}%
              </div>
            </div>
            <div className="axis-labels">
              <span>프리미엄 선호 (Premium)</span>
              <span>가성비 중시 (Cost-effective)</span>
            </div>
            {selectedAxis === 'price_sensitivity' && (
              <div className="axis-detail-expanded">
                <p>
                  {profile.taste_test_axis_scores.price_sensitivity.P >= 50
                    ? "가격보다 품질과 경험을 중시합니다. 프리미엄 재료와 서비스를 위해 기꺼이 더 지불할 의향이 있습니다."
                    : "합리적인 가격과 가성비를 중요하게 생각합니다. 저렴하면서도 맛있는 음식을 찾는 것을 즐깁니다."}
                </p>
              </div>
            )}
          </div>

          <div 
            className={`axis-item ${selectedAxis === 'dining_company' ? 'expanded' : ''}`}
            onClick={() => setSelectedAxis(selectedAxis === 'dining_company' ? null : 'dining_company')}
          >
            <div className="axis-name">동행 선호도 (Dining Company)</div>
            <div className="axis-bar-container">
              <div 
                className="axis-left" 
                style={{ width: `${profile.taste_test_axis_scores.dining_company.A}%` }}
              >
                A {profile.taste_test_axis_scores.dining_company.A}%
              </div>
              <div 
                className="axis-right" 
                style={{ width: `${profile.taste_test_axis_scores.dining_company.O}%` }}
              >
                O {profile.taste_test_axis_scores.dining_company.O}%
              </div>
            </div>
            <div className="axis-labels">
              <span>함께 (All together)</span>
              <span>혼자 (On my own)</span>
            </div>
            {selectedAxis === 'dining_company' && (
              <div className="axis-detail-expanded">
                <p>
                  {profile.taste_test_axis_scores.dining_company.A >= 50
                    ? "친구나 가족과 함께 식사하는 것을 좋아합니다. 왁자지껄한 분위기에서 음식을 나누며 즐기는 것을 선호합니다."
                    : "혼자만의 시간을 즐기며 식사합니다. 조용히 자신만의 페이스로 음식을 즐기는 것을 좋아합니다."}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 다시 테스트하기 버튼 */}
      <div className="retest-section">
        <button 
          className="btn-retest-main" 
          onClick={() => setShowRetestOptions(!showRetestOptions)}
        >
          🔄 다시 테스트하기
        </button>
        
        {showRetestOptions && (
          <div className="retest-options-inline">
            <button className="retest-option-btn" onClick={handleStartQuickTest}>
              ⚡ 간단 테스트 (8문항, ~1분)
            </button>
            <button className="retest-option-btn" onClick={handleStartDeepTest}>
              🔍 심화 테스트 (20문항, ~3-4분)
            </button>
          </div>
        )}
      </div>

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
      {showTypeModal && selectedType && (
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
                    <i className="fas fa-thumbs-up"></i> 추천 메뉴 & 장소
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
                    <i className="fas fa-ban"></i> 피해야 할 식당
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
        </div>
      )}
    </div>
  );
};

export default MBTIDetailPage;

