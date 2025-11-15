import React from 'react';
import EmptyState from '../EmptyState';

const MenuTab = () => {
  const handleMenuUpload = () => {
    alert('준비 중입니다');
  };

  return (
    <div className="menu-tab">
      <EmptyState
        icon="📋"
        message="준비 중입니다. 곧 업데이트 예정입니다."
        action={
          <button className="btn-upload" onClick={handleMenuUpload}>
            + 메뉴 추가
          </button>
        }
      />
    </div>
  );
};

export default MenuTab;

