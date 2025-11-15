"""
한국 레스토랑 데이터 추가 스크립트
CSV 파일에서 레스토랑 정보를 읽어 DB에 삽입합니다.

사용법:
    python scripts/add_korean_restaurants.py restaurants.csv

CSV 형식:
    name,address,category,phone,latitude,longitude
    아주반점,경기도 수원시 영통구 월드컵로 206,중식,031-123-4567,37.2809,127.0445
"""

import sys
import csv
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))

from database import SessionLocal
import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_csv(csv_path):
    """CSV 파일 파싱"""
    restaurants = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            restaurants.append({
                'name': row['name'],
                'address': row['address'],
                'categories': row.get('category', '레스토랑'),
                'phone': row.get('phone', ''),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
            })
    
    return restaurants


def extract_city_state(address):
    """주소에서 시/도 추출"""
    # 간단한 파싱 (더 정교하게 개선 가능)
    parts = address.split()
    
    if len(parts) >= 2:
        city = parts[1]  # 예: 수원시
        state = parts[0]  # 예: 경기도
    elif len(parts) == 1:
        city = parts[0]
        state = "Korea"
    else:
        city = "Unknown"
        state = "Korea"
    
    return city, state


def add_restaurants_to_db(restaurants):
    """레스토랑을 DB에 추가"""
    db = SessionLocal()
    added_count = 0
    skipped_count = 0
    
    try:
        for rest in restaurants:
            # 중복 확인 (이름 + 주소로)
            existing = db.query(models.Business).filter(
                models.Business.name == rest['name'],
                models.Business.address == rest['address']
            ).first()
            
            if existing:
                logger.info(f"이미 존재: {rest['name']}")
                skipped_count += 1
                continue
            
            # 시/도 추출
            city, state = extract_city_state(rest['address'])
            
            # business_id 생성 (이름 기반)
            business_id = f"KR_{rest['name'].replace(' ', '_')}_{added_count}"
            
            # DB 객체 생성
            db_business = models.Business(
                business_id=business_id,
                name=rest['name'],
                address=rest['address'],
                city=city,
                state=state,
                latitude=rest['latitude'],
                longitude=rest['longitude'],
                stars=0.0,  # 초기값
                review_count=0,
                categories=rest['categories'],
                absa_features={}  # 빈 ABSA 특징
            )
            
            db.add(db_business)
            added_count += 1
            logger.info(f"추가: {rest['name']} (위치: {rest['latitude']}, {rest['longitude']})")
        
        # 커밋
        db.commit()
        logger.info(f"완료: {added_count}개 추가, {skipped_count}개 스킵")
        
    except Exception as e:
        logger.error(f"에러 발생: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)
    
    logger.info(f"CSV 파일 읽는 중: {csv_path}")
    restaurants = parse_csv(csv_path)
    logger.info(f"{len(restaurants)}개 레스토랑 발견")
    
    logger.info("DB에 추가 중...")
    add_restaurants_to_db(restaurants)
    
    logger.info("모든 작업 완료!")


if __name__ == "__main__":
    main()

