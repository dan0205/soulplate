import React from 'react';
import EmptyState from '../EmptyState';

const PhotoTab = () => {
  return (
    <div className="photo-tab">
      <EmptyState
        icon="📸"
        message="사진이 없어요. 음식, 매장 사진을 공유해주세요!"
      />
    </div>
  );
};

export default PhotoTab;

