"""
SQLAlchemy 데이터베이스 모델
"""
# 이 방식을 ORM 이라고 부른다 
# 이 코드를 실행하면, SQLAlchemy가 이 파이썬 클래스 정의를 읽어서
# 실제 SQL 데이터베이스에 각 테이블을 생성해준다 

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from backend_web.database import Base

# Base = 기본 클래스. 이 클래스는 데이터베이스 테이블과 연결됨을 인식하게 해줌 
# __tablename__ = 디비에 생성될 실제 테이블의 이름 
# Column = 테이블의 열을 정의 

class User(Base):
    """사용자 모델"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    # 사용자 정보를 저장하는 users 테이블 
    # id = 사용자의 고유 번호
    # username, email = 로그인 및 식별에서 사용되며, 중복될 수 없다
    # hashed... = get_password_hash()로 암호화된 비밀번호가 저장된다
    # created_at = 사용자 계정이 생성된 시간을 자동으로 기록되도록 한다
    
    reviews = relationship("Review", back_populates="user")
    # 한 명의 유저는 여러 개의 리뷰를 가질 수 있다 
    # my_user.reviews 처럼 해당 사용자가 작성한 모든 리뷰 목록에 바로 접근할 수 있다


class Business(Base):
    """비즈니스(가게) 모델"""
    __tablename__ = "businesses"
    
    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    categories = Column(String, nullable=True)
    stars = Column(Float, nullable=True)
    review_count = Column(Integer, default=0)
    address = Column(String, nullable=True)
    city = Column(String, nullable=True)
    state = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    is_open = Column(Boolean, default=True)
    # 가게 정보를 저장하는 테이블
    # id = 가게의 고유 번호
    # business_id = ㅇ외부에서 가져온 원본 ID
    # stars, review_count = 가게의 평균 별점과 리뷰 수
    
    # Relationships
    reviews = relationship("Review", back_populates="business")
    # 하나의 비즈니스는 여러 개의 리뷰를 가질 수 있다 
    # my_business.reviews 코드로 해당 가게의 모든 리뷰에 접근 가능 

class Review(Base):
    """리뷰 모델"""
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False)
    stars = Column(Integer, nullable=False)  # 1-5
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    # 리뷰 정보를 저장하는 테이블
    # user_id, business_id = 외래 키로 다른 테이블을 연결한다
    
    # Relationships
    user = relationship("User", back_populates="reviews")
    business = relationship("Business", back_populates="reviews")
    # 하나의 리뷰는 한 명의 유저에 속한다
    # my_review.user 코드로 user 객체 정보에 바로 접근 
    # 하나의 리뷰는 하나의 비즈니스에 속한다 
    # my_review.business 코드로 business 객체 정보에 바로 접근 
    # Foreign key 를 갖고 있는 클래스가 Many 쪽이다 

