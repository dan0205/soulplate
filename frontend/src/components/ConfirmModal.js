import React, { useEffect } from 'react';
import ReactDOM from 'react-dom';
import './ConfirmModal.css';

function ConfirmModal({
  isOpen,
  title,
  message,
  confirmText = '확인',
  cancelText = '취소',
  variant = 'confirm',
  onConfirm,
  onCancel,
}) {
  // ESC 키로 모달 닫기
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        onCancel();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, onCancel]);

  if (!isOpen) return null;

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onCancel();
    }
  };

  return ReactDOM.createPortal(
    <div className="confirm-modal-overlay" onClick={handleOverlayClick}>
      <div className="confirm-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-modal-header">
          <h2>{title}</h2>
          <p>{message}</p>
        </div>
        <div className="confirm-modal-buttons">
          <button
            className="confirm-modal-btn confirm-modal-btn-cancel"
            onClick={onCancel}
          >
            {cancelText}
          </button>
          <button
            className={`confirm-modal-btn ${
              variant === 'danger'
                ? 'confirm-modal-btn-danger'
                : 'confirm-modal-btn-confirm'
            }`}
            onClick={onConfirm}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}

export default ConfirmModal;

