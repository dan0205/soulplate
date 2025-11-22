"""
SQLAlchemy 데이터베이스 모델 (PostgreSQL + ABSA 통합)
"""
# 이 방식을 ORM 이라고 부른다 
# 이 코드를 실행하면, SQLAlchemy가 이 파이썬 클래스 정의를 읽어서
# 실제 SQL 데이터베이스에 각 테이블을 생성해준다 

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, JSON, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime, timezone

from .database import Base

# Base = 기본 클래스. 이 클래스는 데이터베이스 테이블과 연결됨을 인식하게 해줌 
# __tablename__ = 디비에 생성될 실제 테이블의 이름 
# Column = 테이블의 열을 정의 

class User(Base):
    """사용자 모델 (Yelp 데이터 + 신규 회원 통합)"""
    __tablename__ = "users"
    
    # 기본 인증 정보
    id = Column(Integer, primary_key=True, index=True)
    yelp_user_id = Column(String, unique=True, index=True, nullable=True)  # Yelp 데이터 매칭용
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    
    # Yelp 통계 데이터
    review_count = Column(Integer, default=0)
    useful = Column(Integer, default=0)  # useful + funny + cool 통합
    compliment = Column(Integer, default=0)  # 11개 compliment 통합
    fans = Column(Integer, default=0)
    average_stars = Column(Float, default=0.0)
    yelping_since_days = Column(Integer, default=0)  # 가입 경과일
    
    # ABSA 피처 (JSON: 51개 aspect-sentiment 평균값)
    absa_features = Column(JSONB, nullable=True)
    
    # 텍스트 임베딩 (JSON: 100차원 평균 벡터)
    text_embedding = Column(JSONB, nullable=True)
    
    # 취향 테스트 관련 필드
    taste_test_mbti_type = Column(String(4), nullable=True, index=True)  # 'SAPA', 'MOCA' 등
    taste_test_completed = Column(Boolean, default=False, nullable=False)
    taste_test_type = Column(String, nullable=True)  # 'quick' or 'deep'
    taste_test_axis_scores = Column(JSONB, nullable=True)  # 각 축의 확률 점수
    # {"flavor_intensity": {"S": 73, "M": 27}, "dining_environment": {"A": 82, "O": 18}, ...}
    
    # 인구통계 정보 (미래 사용을 위해 저장, 현재 모델 학습에는 미사용)
    age = Column(Integer, nullable=True)  # 나이
    gender = Column(String(10), nullable=True)  # 'M', 'F', 'Other', None
    
    # Relationships
    reviews = relationship("Review", back_populates="user")
    # 한 명의 유저는 여러 개의 리뷰를 가질 수 있다 
    # my_user.reviews 처럼 해당 사용자가 작성한 모든 리뷰 목록에 바로 접근할 수 있다


class Business(Base):
    """비즈니스(가게) 모델 (Yelp 데이터 + ABSA)"""
    __tablename__ = "businesses"
    
    # 기본 정보
    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(String, unique=True, index=True, nullable=False)  # Yelp ID
    name = Column(String, nullable=False)
    categories = Column(String, nullable=True)
    
    # 통계 정보
    stars = Column(Float, nullable=True)
    review_count = Column(Integer, default=0)
    
    # 위치 정보
    address = Column(String, nullable=True)
    city = Column(String, nullable=True)
    state = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # ABSA 피처 (JSON: 51개 aspect-sentiment 평균값)
    absa_features = Column(JSONB, nullable=True)
    
    # 텍스트 임베딩 (JSON: 100차원 평균 벡터)
    text_embedding = Column(JSONB, nullable=True)
    
    # Relationships
    reviews = relationship("Review", back_populates="business")
    # 하나의 비즈니스는 여러 개의 리뷰를 가질 수 있다 
    # my_business.reviews 코드로 해당 가게의 모든 리뷰에 접근 가능 

class Review(Base):
    """리뷰 모델 (Yelp 데이터 + ABSA + 취향 테스트)"""
    __tablename__ = "reviews"
    
    # 기본 정보
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=True)  # 취향 테스트는 null
    stars = Column(Float, nullable=True)  # 1-5 (취향 테스트는 null, 답글도 null)
    text = Column(Text, nullable=False)
    date = Column(DateTime, nullable=True)  # Yelp 리뷰 원본 날짜
    created_at = Column(DateTime, default=datetime.now(timezone.utc))  # DB 삽입 시간
    
    # ABSA 피처 (JSON: 51개 aspect-sentiment 값)
    absa_features = Column(JSONB, nullable=True)
    
    # 유용성 점수
    useful = Column(Integer, default=0, nullable=False)
    
    # 취향 테스트 관련 필드
    is_taste_test = Column(Boolean, default=False, nullable=False)  # 취향 테스트 여부
    taste_test_type = Column(String, nullable=True)  # 'quick' or 'deep'
    taste_test_weight = Column(Float, default=1.0, nullable=False)  # 가중치 (0.7 or 1.0)
    
    # 답글 기능
    parent_review_id = Column(Integer, ForeignKey("reviews.id"), nullable=True, index=True)  # 답글인 경우 부모 리뷰 ID
    
    # Relationships
    user = relationship("User", back_populates="reviews")
    business = relationship("Business", back_populates="reviews")
    parent_review = relationship("Review", remote_side=[id], backref="replies")
    # 하나의 리뷰는 한 명의 유저에 속한다
    # my_review.user 코드로 user 객체 정보에 바로 접근 
    # 하나의 리뷰는 하나의 비즈니스에 속한다 
    # my_review.business 코드로 business 객체 정보에 바로 접근 
    # Foreign key 를 갖고 있는 클래스가 Many 쪽이다 


class UserBusinessPrediction(Base):
    """사용자-음식점 예측 점수 캐시"""
    __tablename__ = "user_business_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False)
    deepfm_score = Column(Float, nullable=False)
    multitower_score = Column(Float, nullable=False)
    is_stale = Column(Boolean, default=False, nullable=False)
    calculated_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User")
    business = relationship("Business")
    
    # Unique constraint and indexes
    __table_args__ = (
        UniqueConstraint('user_id', 'business_id', name='uq_user_business'),
        Index('idx_user_scores', 'user_id', 'deepfm_score'),
        Index('idx_user_multitower', 'user_id', 'multitower_score'),
        Index('idx_stale_predictions', 'is_stale', 'user_id'),
    )

