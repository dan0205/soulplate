import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import toast from 'react-hot-toast';
import { tasteTestAPI } from '../services/api';
import ConfirmModal from '../components/ConfirmModal';
import './TasteTest.css';

function TasteTestPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const testType = location.state?.testType || 'quick';

  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [showExitConfirm, setShowExitConfirm] = useState(false);
  const [showSkipConfirm, setShowSkipConfirm] = useState(false);
  const [pendingExit, setPendingExit] = useState(false);
  const autoProgressTimerRef = useRef(null);

  useEffect(() => {
    loadQuestions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [testType]);

  // 컴포넌트 언마운트 시 타이머 정리
  useEffect(() => {
    return () => {
      if (autoProgressTimerRef.current) {
        clearTimeout(autoProgressTimerRef.current);
      }
    };
  }, []);

  // 뒤로가기 및 페이지 이탈 방지
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (answers.some(a => a !== null)) {
        e.preventDefault();
        e.returnValue = '';
      }
    };

    const handlePopState = (e) => {
      if (answers.some(a => a !== null)) {
        e.preventDefault();
        setPendingExit(true);
        setShowExitConfirm(true);
        window.history.pushState(null, '', window.location.pathname);
      }
    };

    // 현재 페이지를 히스토리 스택에 추가 (뒤로가기 감지용)
    window.history.pushState(null, '', window.location.pathname);
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    window.addEventListener('popstate', handlePopState);
    
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      window.removeEventListener('popstate', handlePopState);
    };
  }, [answers]);

  const loadQuestions = async () => {
    try {
      setLoading(true);
      const response = await tasteTestAPI.getQuestions(testType);
      setQuestions(response.data.questions);
      setAnswers(new Array(response.data.questions.length).fill(null));
      setLoading(false);
    } catch (err) {
      console.error('질문 로딩 실패:', err);
      setError('질문을 불러오는데 실패했습니다.');
      setLoading(false);
    }
  };

  const handleAnswer = (value) => {
    const newAnswers = [...answers];
    newAnswers[currentQuestionIndex] = value;
    setAnswers(newAnswers);

    // 기존 타이머가 있다면 취소
    if (autoProgressTimerRef.current) {
      clearTimeout(autoProgressTimerRef.current);
    }

    // 0.5초 후 자동으로 다음으로 이동
    autoProgressTimerRef.current = setTimeout(() => {
      if (currentQuestionIndex < questions.length - 1) {
        setCurrentQuestionIndex(currentQuestionIndex + 1);
      } else {
        // 마지막 문항이므로 최신 답변 배열을 전달하여 제출
        submitTest(newAnswers);
      }
    }, 500);
  };

  const handleNext = () => {
    if (answers[currentQuestionIndex] === null) {
      toast.dismiss();
      toast.error('답변을 선택해주세요.');
      return;
    }

    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      submitTest();
    }
  };

  const handlePrevious = () => {
    // 자동 진행 타이머 취소
    if (autoProgressTimerRef.current) {
      clearTimeout(autoProgressTimerRef.current);
      autoProgressTimerRef.current = null;
    }

    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const submitTest = async (answersToSubmit = null) => {
    setSubmitting(true);
    const finalAnswers = answersToSubmit || answers;
    
    try {
      const response = await tasteTestAPI.submit({
        test_type: testType,
        answers: finalAnswers
      });
      
      toast.success('취향 테스트가 완료되었습니다! 🎉');
      
      // 0.5초 후 마이페이지로 리다이렉트 (MBTI 상세 페이지로 스크롤)
      setTimeout(() => {
        navigate('/my-profile', { 
          state: { scrollToMbti: true, showResult: true },
          replace: true
        });
      }, 500);
    } catch (err) {
      console.error('테스트 제출 실패:', err);
      toast.dismiss();
      toast.error('테스트 제출에 실패했습니다. 다시 시도해주세요.');
      setSubmitting(false);
    }
  };

  const handleSkip = () => {
    setShowSkipConfirm(true);
  };

  const handleSkipConfirm = () => {
    setShowSkipConfirm(false);
    navigate('/');
  };

  const handleExitConfirm = () => {
    setShowExitConfirm(false);
    setPendingExit(false);
    navigate('/');
  };

  const handleExitCancel = () => {
    setShowExitConfirm(false);
    setPendingExit(false);
  };

  if (loading) {
    return (
      <div className="taste-test-container">
        <div className="loading">질문을 불러오는 중...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="taste-test-container">
        <div className="error">{error}</div>
        <button onClick={() => navigate('/')}>홈으로</button>
      </div>
    );
  }

  if (questions.length === 0) {
    return (
      <div className="taste-test-container">
        <div className="error">질문이 없습니다.</div>
      </div>
    );
  }

  const currentQuestion = questions[currentQuestionIndex];
  const progress = ((currentQuestionIndex + 1) / questions.length) * 100;

  // 섹션 정보 가져오기
  const getSectionInfo = (questionIndex) => {
    if (testType === 'quick') {
      if (questionIndex < 2) return { number: 1, title: '맛의 강도', emoji: '🌶️' };
      if (questionIndex < 4) return { number: 2, title: '분위기 vs 효율', emoji: '✨' };
      if (questionIndex < 6) return { number: 3, title: '비용 기준', emoji: '💰' };
      return { number: 4, title: '식사 인원', emoji: '👥' };
    } else {
      if (questionIndex < 6) return { number: 1, title: '맛의 강도', emoji: '🌶️' };
      if (questionIndex < 12) return { number: 2, title: '분위기 vs 효율', emoji: '✨' };
      if (questionIndex < 18) return { number: 3, title: '비용 기준', emoji: '💰' };
      return { number: 4, title: '식사 인원', emoji: '👥' };
    }
  };

  const currentSection = getSectionInfo(currentQuestionIndex);

  return (
    <div className="taste-test-container">
      <div className="taste-test-header">
        <div className="test-type-badge">
          {testType === 'quick' ? '⚡ 간단 테스트' : '🔍 심화 테스트'}
        </div>
      </div>

      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${progress}%` }}></div>
        <div className="progress-text">
          {currentQuestionIndex + 1} / {questions.length}
        </div>
      </div>

      <div className="question-section">
        <div className="question-number">Q{currentQuestionIndex + 1}</div>
        <h3 className="question-text">{currentQuestion.question}</h3>

        {/* 텍스트 기반 선택지 */}
        <div className="options-list">
          {currentQuestion.options && currentQuestion.options.map((option, index) => (
            <button
              key={index}
              className={`option-button ${answers[currentQuestionIndex] === index + 1 ? 'selected' : ''}`}
              onClick={() => handleAnswer(index + 1)}
            >
              <span className="option-number">{index + 1}</span>
              <span className="option-text">{option}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="navigation-links">
        <a 
          onClick={handlePrevious}
          className={currentQuestionIndex === 0 ? 'disabled' : ''}
        >
          ← 이전
        </a>
        <span className="divider" />
        <a onClick={handleSkip}>
          나중에 하기
        </a>
        <span className="divider" />
        <a onClick={handleNext}>
          {currentQuestionIndex === questions.length - 1 
            ? (submitting ? '제출 중...' : '완료 →') 
            : '다음 →'}
        </a>
      </div>

      {/* 테스트 종료 확인 모달 */}
      <ConfirmModal
        isOpen={showExitConfirm}
        title="테스트를 종료하시겠습니까?"
        message="입력한 답변이 저장되지 않습니다."
        confirmText="종료"
        cancelText="취소"
        variant="danger"
        onConfirm={handleExitConfirm}
        onCancel={handleExitCancel}
      />

      {/* 테스트 건너뛰기 확인 모달 */}
      <ConfirmModal
        isOpen={showSkipConfirm}
        title="테스트를 건너뛰시겠습니까?"
        message="나중에 언제든 다시 할 수 있어요."
        confirmText="건너뛰기"
        cancelText="취소"
        variant="confirm"
        onConfirm={handleSkipConfirm}
        onCancel={() => setShowSkipConfirm(false)}
      />
    </div>
  );
}

export default TasteTestPage;








