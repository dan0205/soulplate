"""
SQLAlchemy 데이터베이스 모델
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from backend_web.database import Base

class User(Base):
    """사용자 모델"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    reviews = relationship("Review", back_populates="user")

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
    
    # Relationships
    reviews = relationship("Review", back_populates="business")

class Review(Base):
    """리뷰 모델"""
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False)
    stars = Column(Integer, nullable=False)  # 1-5
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="reviews")
    business = relationship("Business", back_populates="reviews")

