"""
데이터베이스 초기화 스크립트
테이블 생성 및 샘플 데이터 삽입
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path

from backend_web.database import engine, SessionLocal
from backend_web import models
from backend_web.auth import get_password_hash

# 경로
DATA_DIR = Path("data/processed")

def create_tables():
    """테이블 생성"""
    print("Creating database tables...")
    models.Base.metadata.create_all(bind=engine)
    print("Tables created!")

def seed_businesses():
    """비즈니스 데이터 시드"""
    print("\nSeeding businesses...")
    
    db = SessionLocal()
    try:
        # 이미 데이터가 있는지 확인
        existing_count = db.query(models.Business).count()
        if existing_count > 0:
            print(f"Businesses already seeded ({existing_count} records)")
            return
        
        # CSV에서 로드
        businesses_df = pd.read_csv(DATA_DIR / "businesses.csv")
        
        # 데이터베이스에 삽입
        for _, row in businesses_df.iterrows():
            business = models.Business(
                business_id=row['business_id'],
                name=row['name'],
                categories=row.get('categories_str', ''),
                stars=float(row['stars']) if pd.notna(row['stars']) else None,
                review_count=int(row['review_count']) if pd.notna(row['review_count']) else 0,
                address=row.get('address', ''),
                city=row.get('city', ''),
                state=row.get('state', ''),
                latitude=float(row['latitude']) if pd.notna(row['latitude']) else None,
                longitude=float(row['longitude']) if pd.notna(row['longitude']) else None,
                is_open=bool(row['is_open']) if pd.notna(row['is_open']) else True
            )
            db.add(business)
        
        db.commit()
        print(f"Seeded {len(businesses_df)} businesses")
        
    except Exception as e:
        print(f"Error seeding businesses: {e}")
        db.rollback()
    finally:
        db.close()

def seed_test_users():
    """테스트 사용자 생성"""
    print("\nCreating test users...")
    
    db = SessionLocal()
    try:
        # 이미 데이터가 있는지 확인
        existing_count = db.query(models.User).count()
        if existing_count > 0:
            print(f"Users already exist ({existing_count} records)")
            return
        
        # 테스트 사용자 생성
        test_users = [
            {
                "username": "testuser",
                "email": "test@example.com",
                "password": "test123",
                "age": 30,
                "gender": "M"
            },
            {
                "username": "alice",
                "email": "alice@example.com",
                "password": "alice123",
                "age": 25,
                "gender": "F"
            },
            {
                "username": "bob",
                "email": "bob@example.com",
                "password": "bob123",
                "age": 35,
                "gender": "M"
            }
        ]
        
        for user_data in test_users:
            # Truncate password to 72 bytes for bcrypt
            password = user_data["password"]
            if len(password.encode('utf-8')) > 72:
                password = password[:72]
            
            user = models.User(
                username=user_data["username"],
                email=user_data["email"],
                hashed_password=get_password_hash(password),
                age=user_data["age"],
                gender=user_data["gender"]
            )
            db.add(user)
        
        db.commit()
        print(f"Created {len(test_users)} test users")
        print("\nTest Users:")
        for user_data in test_users:
            print(f"  - {user_data['username']} / {user_data['password']}")
        
    except Exception as e:
        print(f"Error creating test users: {e}")
        db.rollback()
    finally:
        db.close()

def main():
    """메인 함수"""
    print("=" * 60)
    print("Database Initialization")
    print("=" * 60)
    
    create_tables()
    seed_businesses()
    seed_test_users()
    
    print("\n" + "=" * 60)
    print("Database initialized successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

