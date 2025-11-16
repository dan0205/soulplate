"""
클라우드 Railway PostgreSQL에 음식점 추가 스크립트
"""

import sys
import csv
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))
sys.path.insert(0, str(project_root / "scripts"))

import models
import logging
from geocoding_utils import get_coordinates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_csv(csv_path):
    """CSV 파일 파싱 및 자동 geocoding"""
    restaurants = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            name = row['name'].strip()
            address = row['address'].strip()
            categories = row.get('category', '레스토랑').strip()
            phone = row.get('phone', '').strip()
            
            # latitude, longitude 처리
            lat_str = row.get('latitude', '').strip()
            lng_str = row.get('longitude', '').strip()
            
            # 좌표가 비어있으면 자동 geocoding
            if not lat_str or not lng_str:
                logger.info(f"[{idx}] 좌표 없음. Kakao API로 변환 시도: {name} ({address})")
                lat, lng = get_coordinates(address)
                
                if lat is None or lng is None:
                    logger.warning(f"[{idx}] ❌ Geocoding 실패. 건너뜀: {name}")
                    continue
            else:
                # 좌표가 있으면 그대로 사용
                try:
                    lat = float(lat_str)
                    lng = float(lng_str)
                    logger.info(f"[{idx}] 좌표 사용: {name} ({lat}, {lng})")
                except ValueError:
                    logger.error(f"[{idx}] ❌ 좌표 형식 오류. 건너뜀: {name}")
                    continue
            
            restaurants.append({
                'name': name,
                'address': address,
                'categories': categories,
                'phone': phone,
                'latitude': lat,
                'longitude': lng,
            })
    
    return restaurants


def extract_city_state(address):
    """주소에서 시/도 추출"""
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


def add_restaurants_to_cloud(restaurants, database_url):
    """레스토랑을 클라우드 DB에 추가"""
    # Railway DB 연결
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    db = Session()
    
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
            
            # business_id 생성 (이름 기반 + 타임스탬프)
            import time
            timestamp = int(time.time() * 1000)
            business_id = f"KR_{rest['name'].replace(' ', '_')}_{timestamp}"
            
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
            logger.info(f"✅ 추가: {rest['name']} (위치: {rest['latitude']}, {rest['longitude']})")
        
        # 커밋
        db.commit()
        logger.info(f"\n🎉 완료: {added_count}개 추가, {skipped_count}개 스킵")
        
    except Exception as e:
        logger.error(f"❌ 에러 발생: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def main():
    # 환경변수에서 설정 가져오기
    database_url = os.getenv("RAILWAY_DATABASE_URL")
    kakao_api_key = os.getenv("KAKAO_REST_API_KEY")
    
    if not database_url:
        logger.error("❌ RAILWAY_DATABASE_URL 환경변수가 설정되지 않았습니다.")
        sys.exit(1)
    
    if not kakao_api_key:
        logger.error("❌ KAKAO_REST_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        logger.error("❌ CSV 파일 경로를 제공해주세요.")
        logger.error("사용법: python scripts/add_restaurant_to_cloud.py data/restaurants_to_add.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("🚀 클라우드 Railway DB에 음식점 추가")
    logger.info("=" * 60)
    logger.info(f"📁 CSV 파일: {csv_path}")
    logger.info(f"🗄️  DB: {database_url.split('@')[1].split('/')[0]}")  # 호스트만 표시
    logger.info("=" * 60)
    
    logger.info(f"\n📖 CSV 파일 읽는 중: {csv_path}")
    restaurants = parse_csv(csv_path)
    logger.info(f"✅ {len(restaurants)}개 레스토랑 발견\n")
    
    if not restaurants:
        logger.error("❌ 추가할 레스토랑이 없습니다.")
        sys.exit(1)
    
    logger.info("🔄 DB에 추가 중...")
    add_restaurants_to_cloud(restaurants, database_url)
    
    logger.info("\n✅ 모든 작업 완료!")


if __name__ == "__main__":
    main()

