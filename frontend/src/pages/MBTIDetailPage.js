/**
 * MBTI 상세 페이지
 */

import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom';
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
      console.log('Profile data:', response.data);
      console.log('taste_test_axis_scores:', response.data.taste_test_axis_scores);
      console.log('taste_test_completed:', response.data.taste_test_completed);
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

      {/* 4개 축 확률 분석 */}
      {profile.taste_test_axis_scores ? (
        <div className="probability-view">
          <h3 className="probability-title">내 음식 성향 분석표</h3>
          
          <div className="trait-group trait-flavor">
            <div className="trait-info">
              <span>맛의 강도</span>
              <span className="trait-percentage highlight">
                {profile.taste_test_axis_scores.flavor_intensity.S}% 강렬함
              </span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${profile.taste_test_axis_scores.flavor_intensity.S}%` }}></div>
              <div className="bar-circle" style={{ left: `${profile.taste_test_axis_scores.flavor_intensity.S}%` }}></div>
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
                {profile.taste_test_axis_scores.dining_environment.A}% 분위기
              </span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${profile.taste_test_axis_scores.dining_environment.A}%` }}></div>
              <div className="bar-circle" style={{ left: `${profile.taste_test_axis_scores.dining_environment.A}%` }}></div>
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
                {profile.taste_test_axis_scores.price_sensitivity.P}% 프리미엄
              </span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${profile.taste_test_axis_scores.price_sensitivity.P}%` }}></div>
              <div className="bar-circle" style={{ left: `${profile.taste_test_axis_scores.price_sensitivity.P}%` }}></div>
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
                {profile.taste_test_axis_scores.dining_company.O}% 혼자
              </span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${100 - profile.taste_test_axis_scores.dining_company.O}%` }}></div>
              <div className="bar-circle" style={{ left: `${100 - profile.taste_test_axis_scores.dining_company.O}%` }}></div>
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
              확률 분석 데이터가 없습니다. 다시 테스트를 진행해주세요.
            </p>
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
        </div>,
        document.body
      )}
    </div>
  );
};

export default MBTIDetailPage;

