"""
Yelp 레스토랑 10개의 좌표를 경기도 안양시로 변경하는 스크립트
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend_web"))

# .env 파일 로드
env_path = project_root / "backend_web" / ".env"
load_dotenv(env_path)

from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Railway DATABASE_URL 사용
    database_url = os.getenv("RAILWAY_DATABASE_URL")
    
    if not database_url:
        logger.error("RAILWAY_DATABASE_URL이 설정되지 않았습니다!")
        sys.exit(1)
    
    logger.info(f"Railway DB에 연결 중...")
    engine = create_engine(database_url)
    
    with engine.connect() as conn:
        # Step 1: 수정할 레스토랑 확인
        logger.info("\n=== Step 1: 수정할 레스토랑 10개 확인 ===")
        result = conn.execute(text("""
            SELECT id, business_id, name, latitude, longitude, review_count, city, state
            FROM businesses 
            WHERE latitude IS NOT NULL 
              AND longitude IS NOT NULL
              AND review_count > 100
            ORDER BY review_count DESC
            LIMIT 10
        """))
        
        restaurants = result.fetchall()
        logger.info(f"\n수정할 레스토랑 {len(restaurants)}개:")
        for r in restaurants:
            logger.info(f"  ID {r[0]}: {r[2]} (리뷰: {r[5]}) - 현재 위치: {r[3]:.4f}, {r[4]:.4f}")
        
        if not restaurants:
            logger.error("수정할 레스토랑이 없습니다!")
            return
        
        # Step 2: 좌표를 안양시로 변경
        logger.info("\n=== Step 2: 좌표를 안양시(37.39, 126.95)로 변경 ===")
        
        result = conn.execute(text("""
            UPDATE businesses 
            SET 
              latitude = 37.39 + (RANDOM() * 0.02 - 0.01),
              longitude = 126.95 + (RANDOM() * 0.02 - 0.01),
              city = '안양시',
              state = '경기도'
            WHERE id IN (
              SELECT id FROM businesses 
              WHERE latitude IS NOT NULL 
                AND longitude IS NOT NULL
                AND review_count > 100
              ORDER BY review_count DESC 
              LIMIT 10
            )
        """))
        
        conn.commit()
        logger.info(f"✅ {result.rowcount}개 레스토랑 좌표 업데이트 완료!")
        
        # Step 3: 변경 확인
        logger.info("\n=== Step 3: 변경 확인 ===")
        result = conn.execute(text("""
            SELECT id, name, latitude, longitude, city, state, review_count
            FROM businesses 
            WHERE city = '안양시'
            ORDER BY review_count DESC
        """))
        
        updated_restaurants = result.fetchall()
        logger.info(f"\n안양시 레스토랑 {len(updated_restaurants)}개:")
        for r in updated_restaurants:
            logger.info(f"  {r[1]}: ({r[2]:.6f}, {r[3]:.6f}) - 리뷰 {r[6]}개")
        
        logger.info("\n✅ 모든 작업 완료!")
        logger.info("Vercel 사이트를 새로고침하고 지도에서 안양시 위치를 확인하세요.")
        logger.info("좌표: 37.39, 126.95")


if __name__ == "__main__":
    main()

