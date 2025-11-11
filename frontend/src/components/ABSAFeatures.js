/**
 * ABSA 특징 컴포넌트
 * 옵션 2 (상세): 카테고리별 그룹 + 프로그레스 바
 * 옵션 4 (간결): 상위 특징만 태그로 표시
 */

import React from 'react';
import ProgressBar from './ProgressBar';
import './ABSAFeatures.css';

// 카테고리 정의
const ASPECT_CATEGORIES = {
  '음식 관련': ['맛', '짠맛', '매운맛', '단맛', '느끼함', '담백함', '고소함', '품질/신선도', '양'],
  '서비스': ['서비스', '대기'],
  '가격/가치': ['가격'],
  '분위기/시설': ['분위기', '쾌적함/청결도', '소음', '공간', '주차']
};

// 간결한 버전 (홈페이지용)
export const ABSAFeaturesCompact = ({ topFeatures }) => {
  if (!topFeatures || topFeatures.length === 0) return null;

  return (
    <div className="absa-features-compact">
      {topFeatures.map((feature, index) => (
        <span key={index} className={`feature-tag ${feature.sentiment}`}>
          {feature.aspect}({Math.round(feature.score * 100)}%)
        </span>
      ))}
    </div>
  );
};

// 상세 버전 (디테일 페이지용)
export const ABSAFeaturesDetailed = ({ absaFeatures, topFeatures }) => {
  if (!absaFeatures && !topFeatures) return null;

  // absaFeatures JSON을 카테고리별로 그룹화
  const groupedFeatures = {};
  
  if (absaFeatures) {
    Object.entries(absaFeatures).forEach(([key, score]) => {
      const parts = key.split('_');
      if (parts.length >= 2) {
        const sentiment = parts[parts.length - 1];
        const aspect = parts.slice(0, -1).join('_');
        
        // 카테고리 찾기
        let category = '기타';
        for (const [cat, aspects] of Object.entries(ASPECT_CATEGORIES)) {
          if (aspects.includes(aspect)) {
            category = cat;
            break;
          }
        }
        
        if (!groupedFeatures[category]) {
          groupedFeatures[category] = [];
        }
        
        // 주요 sentiment만 표시 (긍정/부정)
        if (sentiment === '긍정' || sentiment === '부정') {
          groupedFeatures[category].push({ aspect, sentiment, score });
        }
      }
    });
  }

  // 카테고리 아이콘
  const categoryIcons = {
    '음식 관련': '🍽️',
    '서비스': '🙋',
    '가격/가치': '💰',
    '분위기/시설': '🏠'
  };

  return (
    <div className="absa-features-detailed">
      <h3>📍 이 가게의 특징 (리뷰 분석)</h3>
      
      {Object.entries(groupedFeatures).map(([category, features]) => (
        <div key={category} className="feature-category">
          <h4>
            <span className="category-icon">{categoryIcons[category] || '📊'}</span>
            {category}
          </h4>
          <div className="features-list">
            {features
              .sort((a, b) => b.score - a.score)
              .slice(0, 5)  // 상위 5개만
              .map((feature, index) => (
                <div key={index} className="feature-item">
                  <div className="feature-header">
                    <span className="aspect-name">{feature.aspect}</span>
                    <span className={`sentiment-badge ${feature.sentiment}`}>
                      {feature.sentiment}
                    </span>
                  </div>
                  <ProgressBar value={feature.score} sentiment={feature.sentiment} />
                </div>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default { ABSAFeaturesCompact, ABSAFeaturesDetailed };

