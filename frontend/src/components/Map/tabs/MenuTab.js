import React from 'react';
import EmptyState from '../EmptyState';

const MenuTab = () => {
  return (
    <div className="menu-tab">
      <EmptyState
        icon="📋"
        message="준비 중입니다. 곧 업데이트 예정입니다."
      />
    </div>
  );
};

export default MenuTab;

