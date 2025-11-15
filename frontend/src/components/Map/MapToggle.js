import React from 'react';
import './Map.css';

const MapToggle = ({ viewMode, onToggle }) => {
  return (
    <div className="map-toggle-container">
      <button
        className={`toggle-btn ${viewMode === 'map' ? 'active' : ''}`}
        onClick={() => onToggle('map')}
      >
        🗺️ 지도
      </button>
      <button
        className={`toggle-btn ${viewMode === 'list' ? 'active' : ''}`}
        onClick={() => onToggle('list')}
      >
        📋 리스트
      </button>
    </div>
  );
};

export default MapToggle;

