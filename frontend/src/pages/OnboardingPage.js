/**
 * 온보딩 페이지 - 신규 사용자 프로필 입력
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { authAPI } from '../services/api';
import toast from 'react-hot-toast';
import './OnboardingPage.css';

const OnboardingPage = () => {
  const navigate = useNavigate();
  const { user, loadUser } = useAuth();
  const [formData, setFormData] = useState({
    username: '',
    age: '',
    gender: ''
  });
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // 사용자 정보가 로드되면 username을 미리 채움
  useEffect(() => {
    if (user && user.username) {
      setFormData(prev => ({
        ...prev,
        username: user.username
      }));
    }
  }, [user]);

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

      toast.success('프로필 설정이 완료되었습니다!');
      
      // 사용자 정보를 다시 로드하여 profile_completed 상태 업데이트
      await loadUser();
      
      // 홈페이지로 이동
      navigate('/', { replace: true });
    } catch (error) {
      console.error('Profile completion error:', error);
      if (error.response?.data?.detail === 'Username already exists') {
        toast.error('이미 사용 중인 닉네임입니다');
        setErrors(prev => ({
          ...prev,
          username: '이미 사용 중인 닉네임입니다'
        }));
      } else {
        toast.error('프로필 설정에 실패했습니다');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="onboarding-container">
      <div className="onboarding-box">
        <div className="header">
          <div className="logo">SoulPlate</div>
          <div className="subtitle">당신의 음식 취향을 발견하는 여정</div>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Username */}
          <div className="form-group">
            <label className="form-label">
              닉네임
              <span className="character-count">
                {formData.username.length}/50
              </span>
            </label>
            <input
              type="text"
              className={`form-input ${errors.username ? 'invalid' : formData.username.length >= 2 ? 'valid' : ''}`}
              name="username"
              placeholder="닉네임을 입력하세요 (2-50자)"
              value={formData.username}
              onChange={handleChange}
              maxLength="50"
              required
            />
            <div className={`helper-text ${errors.username ? 'error' : formData.username.length >= 2 && !errors.username ? 'success' : ''}`}>
              {errors.username ? `❌ ${errors.username}` : formData.username.length >= 2 ? '✓ 사용 가능한 닉네임입니다' : '💡 2-50자, 영문/한글/숫자/_/-/공백 사용 가능'}
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
              {errors.age ? `❌ ${errors.age}` : formData.age && !errors.age ? '✓ 입력 완료' : '💡 14세 이상만 가입 가능합니다'}
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
            {isSubmitting ? '저장 중...' : '완료하고 시작하기'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default OnboardingPage;

