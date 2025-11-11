/**
 * 프로그레스 바 컴포넌트
 */

import React from 'react';
import './ProgressBar.css';

const ProgressBar = ({ value, sentiment }) => {
  const percentage = Math.round(value * 100);
  
  return (
    <div className="progress-bar-container">
      <div 
        className={`progress-bar ${sentiment}`}
        style={{ width: `${percentage}%` }}
      >
        <span className="percentage">{percentage}%</span>
      </div>
    </div>
  );
};

export default ProgressBar;

