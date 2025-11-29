import React from 'react';
import './Map.css';

// 카테고리 데이터 정의
const CATEGORIES = [
  { id: 'korean', label: '한식', emoji: '🍖' },
  { id: 'western', label: '양식', emoji: '🍝' },
  { id: 'japanese', label: '일식', emoji: '🍣' },
  { id: 'asian', label: '아시안', emoji: '🍜' },
  { id: 'chinese', label: '중식', emoji: '🥟' },
];

const CategoryFilter = ({ selectedCategory, onCategoryChange }) => {
  const handleClick = (categoryId) => {
    // 이미 선택된 카테고리를 다시 클릭하면 선택 해제 (전체 보기)
    if (selectedCategory === categoryId) {
      onCategoryChange(null);
    } else {
      onCategoryChange(categoryId);
    }
  };

  return (
    <div className="category-filter-container">
      <div className="category-filter-scroll">
        {CATEGORIES.map((category) => (
          <button
            key={category.id}
            className={`category-chip ${selectedCategory === category.id ? 'active' : ''}`}
            onClick={() => handleClick(category.id)}
          >
            <span className="category-emoji">{category.emoji}</span>
            <span className="category-label">{category.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default CategoryFilter;

