import React, { useState, useEffect, useRef } from 'react';
import './Map.css';

const FloatingSearchBar = ({ onSearch, placeholder, defaultValue = '' }) => {
  const [value, setValue] = useState(defaultValue);
  const debounceTimerRef = useRef(null);

  // 디바운싱: 300ms 후 검색 실행
  useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      if (onSearch) {
        onSearch(value);
      }
    }, 300);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [value, onSearch]);

  // 외부에서 defaultValue 변경 시 동기화
  useEffect(() => {
    setValue(defaultValue);
  }, [defaultValue]);

  const handleChange = (e) => {
    setValue(e.target.value);
  };

  const handleClear = () => {
    setValue('');
  };

  return (
    <div className="floating-search-bar">
      <input
        type="text"
        value={value}
        onChange={handleChange}
        placeholder={placeholder}
        className="search-input"
      />
      {value && (
        <button 
          className="clear-search-btn" 
          onClick={handleClear}
          aria-label="검색어 지우기"
        >
          ✕
        </button>
      )}
    </div>
  );
};

export default FloatingSearchBar;

