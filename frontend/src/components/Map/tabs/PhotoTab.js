import React from 'react';
import EmptyState from '../EmptyState';

const PhotoTab = () => {
  const handlePhotoUpload = () => {
    alert('준비 중입니다');
  };

  return (
    <div className="photo-tab">
      <EmptyState
        icon="📸"
        message="사진이 없어요. 음식, 매장 사진을 공유해주세요!"
        action={
          <button className="btn-upload" onClick={handlePhotoUpload}>
            + 사진 업로드
          </button>
        }
      />
    </div>
  );
};

export default PhotoTab;

