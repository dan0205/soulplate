"""
데이터베이스 설정
"""

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import time
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)
# 데이터베이스 접속 정보를 .env 라는 별도 파일에서 안전하게 불러오기 위해 사용된다 
# .env 파일 내 변수를 환경 변수로 로드한다 

# PostgreSQL 연결 문자열
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://two_tower_user:twotower2024@localhost:5432/two_tower_db"
)
# DATABASE_URL 이라는 이름의 환경 변수 값을 가져온다
# 환경 변수가 없으면 기본값을 사용한다 

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    pool_pre_ping=True,
    # SQLAlchemy가 커넥션 풀에서 커넥션을 꺼내 API에 전달하기 직전에 ping을 보내
    # 해당 커넥션이 현재 유효한지 미리 확인한다다
    pool_size=5,
    # 기본적으로 5개의 데이터베이스 커넥션을 미리 만들어 풀을 유지한다
    max_overflow=10
    # 5개가 모두 사용 중일때, 추가로 10개의 커넥션을 더 만들수있다 
)
# postgre는 여러 사용자가 동시에 접속하기 때문에, 커넥션 풀 관리가 매우 중요하다


# DB 쿼리 성능 모니터링 - 쿼리 실행 전
@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """쿼리 실행 시작 시간 기록"""
    conn.info.setdefault('query_start_time', []).append(time.time())
    conn.info.setdefault('query_count', 0)
    conn.info['query_count'] += 1


# DB 쿼리 성능 모니터링 - 쿼리 실행 후
@event.listens_for(engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """쿼리 실행 시간 측정 및 로깅"""
    total = time.time() - conn.info['query_start_time'].pop(-1)
    
    # 100ms 이상 걸린 쿼리만 로깅 (슬로우 쿼리)
    if total > 0.1:
        # 쿼리 문자열을 200자로 제한 (너무 길면 로그가 복잡함)
        query_preview = statement[:200].replace('\n', ' ')
        logger.warning(f"🐌 SLOW QUERY ({total:.3f}s): {query_preview}")
        
        # 파라미터도 출력 (디버깅용)
        if parameters:
            logger.warning(f"   Parameters: {parameters}")
    
    # 모든 쿼리 실행 시간 로깅 (개발 환경용)
    # 프로덕션에서는 주석 처리하거나 환경변수로 제어
    elif os.getenv("DEBUG_SQL", "false").lower() == "true":
        query_preview = statement[:100].replace('\n', ' ')
        logger.debug(f"⚡ Query ({total:.3f}s): {query_preview}")


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# engine에 연결된 세션을 만드는 공장이다 
Base = declarative_base()
# models.py의 user, review 등이 상속받는 기본 클래스이다 

def get_db():
    """데이터베이스 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
    # FastAPI의 Depends()를 통해 API요청이 올때마다 SessionLocal()로 새 세션을 생성한다
    # 데이터베이스 세션 = 작업 장바구니 or 임시 작업 공간
    # 사용자가 자기만의 세션에서 작업을 하고, 커밋을 통해 디비를 업데이트하는 용도 

