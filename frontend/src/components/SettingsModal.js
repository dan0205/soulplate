/**
 * 설정 모달 - 프로필 정보 수정
 */

import React, { useState, useEffect, useRef } from 'react';
import ReactDOM from 'react-dom';
import { authAPI } from '../services/api';
import toast from 'react-hot-toast';
import './SettingsModal.css';

const SettingsModal = ({ isOpen, onClose, currentUser, onUpdateSuccess }) => {
  const [formData, setFormData] = useState({
    username: '',
    age: '',
    gender: ''
  });
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSuccessBanner, setShowSuccessBanner] = useState(false);
  const modalRef = useRef(null);

  // currentUser 정보로 초기화
  useEffect(() => {
    if (currentUser) {
      setFormData({
        username: currentUser.username || '',
        age: currentUser.age ? String(currentUser.age) : '',
        gender: currentUser.gender || ''
      });
    }
  }, [currentUser]);

  // ESC 키로 모달 닫기
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isOpen) {
        handleClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // Username 유효성 검사
  const validateUsername = (value) => {
    if (value.length < 2) {
      return '최소 2자 이상 입력해주세요';
    }
    if (value.length > 50) {
      return '최대 50자까지 입력 가능합니다';
    }
    const validPattern = /^[a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ_\- ]+$/;
    if (!validPattern.test(value)) {
      return '영문/한글/숫자/_/-/공백만 사용 가능합니다';
    }
    return null;
  };

  // Age 유효성 검사
  const validateAge = (value) => {
    if (!value) {
      return '나이를 입력해주세요';
    }
    const age = parseInt(value);
    if (age < 14) {
      return '14세 이상만 가입 가능합니다';
    }
    if (age > 120) {
      return '올바른 나이를 입력해주세요';
    }
    return null;
  };

  // Gender 유효성 검사
  const validateGender = (value) => {
    if (!value) {
      return '성별을 선택해주세요';
    }
    return null;
  };

  // 입력 핸들러
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));

    // 실시간 유효성 검사
    let error = null;
    if (name === 'username') {
      error = validateUsername(value);
    } else if (name === 'age') {
      error = validateAge(value);
    } else if (name === 'gender') {
      error = validateGender(value);
    }

    setErrors(prev => ({
      ...prev,
      [name]: error
    }));
  };

  // 폼 제출 가능 여부
  const isFormValid = () => {
    return (
      !validateUsername(formData.username) &&
      !validateAge(formData.age) &&
      !validateGender(formData.gender)
    );
  };

  // 변경사항 확인
  const hasChanges = () => {
    if (!currentUser) return false;
    return (
      formData.username !== currentUser.username ||
      String(formData.age) !== String(currentUser.age) ||
      formData.gender !== currentUser.gender
    );
  };

  // 모달 닫기
  const handleClose = () => {
    if (hasChanges()) {
      if (!window.confirm('변경사항을 저장하지 않고 나가시겠습니까?')) {
        return;
      }
    }
    onClose();
  };

  // 오버레이 클릭 시 모달 닫기
  const handleOverlayClick = (e) => {
    if (e.target === modalRef.current) {
      handleClose();
    }
  };

  // 폼 제출
  const handleSubmit = async (e) => {
    e.preventDefault();

    // 최종 유효성 검사
    const usernameError = validateUsername(formData.username);
    const ageError = validateAge(formData.age);
    const genderError = validateGender(formData.gender);

    if (usernameError || ageError || genderError) {
      setErrors({
        username: usernameError,
        age: ageError,
        gender: genderError
      });
      return;
    }

    setIsSubmitting(true);

    try {
      await authAPI.completeProfile({
        username: formData.username,
        age: parseInt(formData.age),
        gender: formData.gender
      });

      // 성공 배너 표시
      setShowSuccessBanner(true);
      toast.success('프로필이 성공적으로 업데이트되었습니다!');

      // 사용자 정보 새로고침
      if (onUpdateSuccess) {
        await onUpdateSuccess();
      }

      // 3초 후 모달 닫기
      setTimeout(() => {
        setShowSuccessBanner(false);
        onClose();
      }, 2000);
    } catch (error) {
      console.error('Profile update error:', error);
      if (error.response?.data?.detail === 'Username already exists') {
        toast.error('이미 사용 중인 닉네임입니다');
        setErrors(prev => ({
          ...prev,
          username: '이미 사용 중인 닉네임입니다'
        }));
      } else {
        toast.error('프로필 업데이트에 실패했습니다');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return ReactDOM.createPortal(
    <div 
      className="settings-modal-overlay" 
      ref={modalRef}
      onClick={handleOverlayClick}
    >
      <div className="settings-modal-content">
        <button className="close-btn" onClick={handleClose}>×</button>

        <div className="header">
          <div className="logo">설정</div>
          <div className="subtitle">프로필 정보를 수정할 수 있습니다</div>
        </div>

        {/* 성공 배너 */}
        {showSuccessBanner && (
          <div className="success-banner">
            ✓ 프로필이 성공적으로 업데이트되었습니다!
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {/* Username */}
          <div className="form-group">
            <label className="form-label">
              사용자 이름
              <span className="character-count">
                {formData.username.length}/50
              </span>
            </label>
            <input
              type="text"
              className={`form-input ${errors.username ? 'invalid' : formData.username.length >= 2 ? 'valid' : ''}`}
              name="username"
              placeholder="사용자 이름을 입력하세요"
              value={formData.username}
              onChange={handleChange}
              maxLength="50"
              required
            />
            <div className={`helper-text ${errors.username ? 'error' : formData.username.length >= 2 && !errors.username ? 'success' : ''}`}>
              {errors.username ? `✗ ${errors.username}` : formData.username.length >= 2 ? '✓ 사용 가능한 이름입니다' : '영문/한글/숫자/_/-/공백 사용 가능 (2-50자)'}
            </div>
          </div>

          {/* Age */}
          <div className="form-group">
            <label className="form-label">나이</label>
            <div className="age-input-wrapper">
              <input
                type="number"
                className={`form-input ${errors.age ? 'invalid' : formData.age ? 'valid' : ''}`}
                name="age"
                placeholder="나이"
                value={formData.age}
                onChange={handleChange}
                min="14"
                max="120"
                required
              />
              <span>세</span>
            </div>
            <div className={`helper-text ${errors.age ? 'error' : formData.age && !errors.age ? 'success' : ''}`}>
              {errors.age ? `✗ ${errors.age}` : formData.age && !errors.age ? '✓ 올바른 나이입니다' : '14세 이상 입력 가능'}
            </div>
          </div>

          {/* Gender */}
          <div className="form-group">
            <label className="form-label">성별</label>
            <div className="radio-group">
              <label className={`radio-option ${formData.gender === 'male' ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name="gender"
                  value="male"
                  checked={formData.gender === 'male'}
                  onChange={handleChange}
                  required
                />
                <span className="radio-label">남성</span>
              </label>
              <label className={`radio-option ${formData.gender === 'female' ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name="gender"
                  value="female"
                  checked={formData.gender === 'female'}
                  onChange={handleChange}
                  required
                />
                <span className="radio-label">여성</span>
              </label>
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={!isFormValid() || isSubmitting}
          >
            {isSubmitting ? '저장 중...' : '저장하기'}
          </button>
        </form>
      </div>
    </div>,
    document.body
  );
};

export default SettingsModal;

