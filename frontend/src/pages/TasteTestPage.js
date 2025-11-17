import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { tasteTestAPI } from '../services/api';
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

  useEffect(() => {
    loadQuestions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [testType]);

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
        const confirmLeave = window.confirm('테스트를 종료하시겠습니까? 입력한 답변이 저장되지 않습니다.');
        if (!confirmLeave) {
          window.history.pushState(null, '', window.location.pathname);
        }
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
  };

  const handleNext = () => {
    if (answers[currentQuestionIndex] === null) {
      alert('답변을 선택해주세요.');
      return;
    }

    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      submitTest();
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const submitTest = async () => {
    setSubmitting(true);
    try {
      const response = await tasteTestAPI.submit({
        test_type: testType,
        answers: answers
      });
      
      // 결과 페이지로 이동 (히스토리 스택에서 테스트 페이지 제거)
      navigate('/taste-test/result', { 
        state: { result: response.data, testType },
        replace: true
      });
    } catch (err) {
      console.error('테스트 제출 실패:', err);
      alert('테스트 제출에 실패했습니다. 다시 시도해주세요.');
      setSubmitting(false);
    }
  };

  const handleSkip = () => {
    if (window.confirm('테스트를 건너뛰시겠습니까?')) {
      navigate('/');
    }
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

  return (
    <div className="taste-test-container">
      <div className="taste-test-header">
        <h2>🍽️ 음식 취향 테스트</h2>
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

        <div className="likert-scale">
          {[1, 2, 3, 4, 5].map((value) => (
            <button
              key={value}
              className={`likert-button ${answers[currentQuestionIndex] === value ? 'selected' : ''}`}
              onClick={() => handleAnswer(value)}
            >
              <div className="likert-value">{value}</div>
              <div className="likert-label">{currentQuestion.labels[value - 1]}</div>
            </button>
          ))}
        </div>

        {/* 이모지 버전 (선택적) */}
        <div className="likert-emoji-scale" style={{ display: 'none' }}>
          {['😞', '😕', '😐', '🙂', '😍'].map((emoji, index) => (
            <button
              key={index + 1}
              className={`likert-emoji-button ${answers[currentQuestionIndex] === index + 1 ? 'selected' : ''}`}
              onClick={() => handleAnswer(index + 1)}
            >
              <span className="emoji">{emoji}</span>
              <div className="likert-label">{currentQuestion.labels[index]}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="navigation-buttons">
        <button 
          className="btn-secondary" 
          onClick={handlePrevious}
          disabled={currentQuestionIndex === 0}
        >
          이전
        </button>

        <button 
          className="btn-skip" 
          onClick={handleSkip}
        >
          나중에 하기
        </button>

        <button 
          className="btn-primary" 
          onClick={handleNext}
          disabled={submitting}
        >
          {currentQuestionIndex === questions.length - 1 
            ? (submitting ? '제출 중...' : '완료') 
            : '다음'}
        </button>
      </div>
    </div>
  );
}

export default TasteTestPage;








