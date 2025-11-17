"""
기존 사용자들의 예측값 초기화 스크립트
"""

import sys
import os
import asyncio

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy.orm import Session
from backend_web.database import SessionLocal
from backend_web import models
from backend_web.prediction_cache import calculate_and_store_predictions


async def initialize_all_predictions():
    """모든 사용자의 예측값 초기화"""
    print("=" * 60)
    print("기존 사용자 예측값 초기화 시작")
    print("=" * 60)
    
    db: Session = SessionLocal()
    
    try:
        # 모든 사용자 조회
        users = db.query(models.User).all()
        total_users = len(users)
        
        print(f"\n총 {total_users}명의 사용자 발견")
        print(f"예상 소요 시간: 약 {total_users * 11}초 (~{total_users * 11 / 60:.1f}분)")
        print("\n계속하시겠습니까? (y/n): ", end='')
        
        response = input().strip().lower()
        if response != 'y':
            print("취소되었습니다.")
            return
        
        print("\n처리 시작...\n")
        
        # 각 사용자에 대해 예측 계산
        for idx, user in enumerate(users, 1):
            print(f"[{idx}/{total_users}] 사용자 {user.id} ({user.username}) 처리 중...")
            
            try:
                await calculate_and_store_predictions(user.id, db)
                print(f"  ✓ 완료")
            except Exception as e:
                print(f"  ✗ 오류: {e}")
        
        print("\n" + "=" * 60)
        print("✓ 모든 사용자의 예측값 초기화 완료!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
    
    finally:
        db.close()


async def initialize_single_user(user_id: int):
    """특정 사용자의 예측값 초기화"""
    print("=" * 60)
    print(f"사용자 {user_id}의 예측값 초기화")
    print("=" * 60)
    
    db: Session = SessionLocal()
    
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            print(f"\n✗ 사용자 {user_id}를 찾을 수 없습니다.")
            return
        
        print(f"\n사용자: {user.username}")
        print("처리 시작...\n")
        
        await calculate_and_store_predictions(user.id, db)
        
        print("\n" + "=" * 60)
        print("✓ 예측값 초기화 완료!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
    
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 특정 사용자 ID가 제공된 경우
        try:
            user_id = int(sys.argv[1])
            asyncio.run(initialize_single_user(user_id))
        except ValueError:
            print("오류: 사용자 ID는 숫자여야 합니다.")
            print("사용법: python scripts/initialize_predictions.py [user_id]")
    else:
        # 모든 사용자 초기화
        asyncio.run(initialize_all_predictions())







