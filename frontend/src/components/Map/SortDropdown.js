import React, { useState, useEffect, useRef } from 'react';
import './Map.css';

const SortDropdown = ({ value, onChange, options }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // 현재 선택된 옵션 찾기
  const selectedOption = options.find(opt => opt.value === value) || options[0];

  // 외부 클릭 시 드롭다운 닫기
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('touchstart', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleClickOutside);
    };
  }, [isOpen]);

  const handleToggle = () => {
    setIsOpen(!isOpen);
  };

  const handleSelect = (optionValue) => {
    onChange(optionValue);
    setIsOpen(false);
  };

  return (
    <div className="sort-dropdown" ref={dropdownRef}>
      <button 
        className="sort-dropdown-button" 
        onClick={handleToggle}
        aria-haspopup="true"
        aria-expanded={isOpen}
      >
        <span>{selectedOption.label}</span>
        <span>{isOpen ? '▲' : '▼'}</span>
      </button>
      
      {isOpen && (
        <div className="sort-dropdown-menu" role="menu">
          {options.map((option) => (
            <div
              key={option.value}
              className={`sort-dropdown-item ${option.value === value ? 'active' : ''}`}
              onClick={() => handleSelect(option.value)}
              role="menuitem"
            >
              <span>{option.label}</span>
              {option.value === value && <span>✓</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SortDropdown;

