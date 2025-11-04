# PostgreSQL 전환 가이드

## 1. PostgreSQL 설치

### Windows
```powershell
# PostgreSQL 다운로드 및 설치
# https://www.postgresql.org/download/windows/
# 또는 Chocolatey 사용
choco install postgresql
```

### Mac
```bash
brew install postgresql
brew services start postgresql
```

### Linux (Ubuntu)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

## 2. 데이터베이스 생성

```bash
# PostgreSQL 접속
sudo -u postgres psql

# 데이터베이스 및 사용자 생성
CREATE DATABASE two_tower_db;
CREATE USER two_tower_user WITH PASSWORD 'your_strong_password';
GRANT ALL PRIVILEGES ON DATABASE two_tower_db TO two_tower_user;

# PostgreSQL 15 이상인 경우 추가 권한 부여 필요
\c two_tower_db
GRANT ALL ON SCHEMA public TO two_tower_user;

# 확인
\l  # 데이터베이스 목록
\q  # 종료
```

## 3. 코드 변경

### 3.1 requirements.txt 업데이트

**backend_web/requirements.txt**에 추가:
```txt
psycopg2-binary==2.9.9  # PostgreSQL 드라이버
```

또는 컴파일 버전:
```txt
psycopg2==2.9.9
```

### 3.2 database.py 수정

**변경 전 (backend_web/database.py):**
```python
SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}  # SQLite only
)
```

**변경 후:**
```python
import os

# PostgreSQL 연결 문자열
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://two_tower_user:your_strong_password@localhost:5432/two_tower_db"
)

# SQLite와 달리 connect_args 불필요
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # 연결 체크
    pool_size=5,  # 연결 풀 크기
    max_overflow=10  # 최대 추가 연결
)
```

### 3.3 환경 변수 설정

**.env 파일 생성** (backend_web/.env):
```env
DATABASE_URL=postgresql://two_tower_user:your_strong_password@localhost:5432/two_tower_db
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
MODEL_API_URL=http://localhost:8001
```

**database.py에서 .env 로드:**
```python
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 로드

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
```

### 3.4 requirements.txt에 python-dotenv 추가

```txt
python-dotenv==1.0.0
```

## 4. 모델 수정 (필요시)

### Boolean 타입 처리

PostgreSQL은 Boolean을 네이티브로 지원하지만, SQLite는 INTEGER로 저장합니다.

**models.py**는 수정 불필요 (SQLAlchemy가 자동 처리):
```python
is_open = Column(Boolean, default=True)  # 그대로 유지
```

### DateTime 타입

PostgreSQL에서 timezone 사용 시:
```python
from sqlalchemy import Column, DateTime
from sqlalchemy.sql import func

created_at = Column(DateTime(timezone=True), server_default=func.now())
```

## 5. 데이터 마이그레이션

### 방법 1: init_db.py 재실행 (권장)

```bash
cd backend_web
source venv/bin/activate  # Windows: venv\Scripts\activate
python ../scripts/init_db.py
```

### 방법 2: 기존 SQLite 데이터 이전

**데이터 덤프 스크립트 (scripts/migrate_sqlite_to_postgres.py):**
```python
"""
SQLite에서 PostgreSQL로 데이터 마이그레이션
"""

import sqlite3
import psycopg2
from psycopg2.extras import execute_values

# SQLite 연결
sqlite_conn = sqlite3.connect('app.db')
sqlite_cursor = sqlite_conn.cursor()

# PostgreSQL 연결
pg_conn = psycopg2.connect(
    "postgresql://two_tower_user:your_strong_password@localhost:5432/two_tower_db"
)
pg_cursor = pg_conn.cursor()

print("마이그레이션 시작...")

# Users 마이그레이션
print("Users 테이블 마이그레이션...")
sqlite_cursor.execute("SELECT * FROM users")
users = sqlite_cursor.fetchall()
if users:
    execute_values(
        pg_cursor,
        "INSERT INTO users VALUES %s",
        users
    )
print(f"  ✓ {len(users)} users 마이그레이션 완료")

# Businesses 마이그레이션
print("Businesses 테이블 마이그레이션...")
sqlite_cursor.execute("SELECT * FROM businesses")
businesses = sqlite_cursor.fetchall()
if businesses:
    execute_values(
        pg_cursor,
        "INSERT INTO businesses VALUES %s",
        businesses
    )
print(f"  ✓ {len(businesses)} businesses 마이그레이션 완료")

# Reviews 마이그레이션
print("Reviews 테이블 마이그레이션...")
sqlite_cursor.execute("SELECT * FROM reviews")
reviews = sqlite_cursor.fetchall()
if reviews:
    execute_values(
        pg_cursor,
        "INSERT INTO reviews VALUES %s",
        reviews
    )
print(f"  ✓ {len(reviews)} reviews 마이그레이션 완료")

# Sequence 재설정 (Auto-increment)
print("\nSequence 재설정...")
pg_cursor.execute("SELECT setval('users_id_seq', (SELECT MAX(id) FROM users));")
pg_cursor.execute("SELECT setval('businesses_id_seq', (SELECT MAX(id) FROM businesses));")
pg_cursor.execute("SELECT setval('reviews_id_seq', (SELECT MAX(id) FROM reviews));")

pg_conn.commit()
sqlite_conn.close()
pg_conn.close()

print("\n마이그레이션 완료!")
```

## 6. 패키지 설치

```bash
cd backend_web
pip install psycopg2-binary python-dotenv
```

## 7. 테이블 생성 및 확인

```bash
# Python 콘솔에서
python

>>> from backend_web.database import engine
>>> from backend_web import models
>>> models.Base.metadata.create_all(bind=engine)
>>> print("테이블 생성 완료!")
```

또는:
```bash
python scripts/init_db.py
```

## 8. 연결 테스트

```python
# test_postgres_connection.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://two_tower_user:your_strong_password@localhost:5432/two_tower_db"

try:
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    print("✓ PostgreSQL 연결 성공!")
    
    # 테스트 쿼리
    result = connection.execute("SELECT version();")
    version = result.fetchone()
    print(f"PostgreSQL 버전: {version[0]}")
    
    connection.close()
except Exception as e:
    print(f"✗ 연결 실패: {e}")
```

## 9. 주요 차이점

### SQLite vs PostgreSQL

| 항목 | SQLite | PostgreSQL |
|------|--------|------------|
| 연결 문자열 | `sqlite:///./app.db` | `postgresql://user:pass@host:port/db` |
| 드라이버 | 내장 | `psycopg2` 필요 |
| 동시성 | 제한적 (단일 쓰기) | 높음 (다중 쓰기) |
| 성능 | 소규모 적합 | 대규모 적합 |
| 타입 시스템 | 유연함 | 엄격함 |
| Boolean | INTEGER (0/1) | 네이티브 BOOLEAN |
| Date/Time | TEXT | 네이티브 TIMESTAMP |

## 10. 트러블슈팅

### 문제 1: psycopg2 설치 오류
```bash
# 해결: binary 버전 사용
pip install psycopg2-binary
```

### 문제 2: 연결 거부 (Connection refused)
```bash
# PostgreSQL 서비스 확인
sudo systemctl status postgresql  # Linux
brew services list  # Mac

# 방화벽 확인
sudo ufw allow 5432  # Linux
```

### 문제 3: 권한 오류
```sql
-- PostgreSQL에서
GRANT ALL PRIVILEGES ON DATABASE two_tower_db TO two_tower_user;
GRANT ALL ON SCHEMA public TO two_tower_user;
```

### 문제 4: pg_hba.conf 설정
```bash
# /etc/postgresql/15/main/pg_hba.conf 수정
# 로컬 연결 허용
local   all   all   md5
host    all   all   127.0.0.1/32   md5

# PostgreSQL 재시작
sudo systemctl restart postgresql
```

## 11. 성능 최적화

### 인덱스 생성
```sql
-- 자주 검색되는 컬럼에 인덱스
CREATE INDEX idx_business_city ON businesses(city);
CREATE INDEX idx_business_stars ON businesses(stars);
CREATE INDEX idx_review_user_id ON reviews(user_id);
CREATE INDEX idx_review_business_id ON reviews(business_id);
CREATE INDEX idx_review_created_at ON reviews(created_at);
```

### 연결 풀 설정
```python
# database.py
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=10,  # 기본 연결 수
    max_overflow=20,  # 최대 추가 연결
    pool_pre_ping=True,  # 연결 체크
    pool_recycle=3600  # 1시간마다 연결 재생성
)
```

## 12. 백업 및 복원

### 백업
```bash
pg_dump -U two_tower_user -d two_tower_db > backup.sql
```

### 복원
```bash
psql -U two_tower_user -d two_tower_db < backup.sql
```

## 완료 체크리스트

- [ ] PostgreSQL 설치 및 실행
- [ ] 데이터베이스 및 사용자 생성
- [ ] requirements.txt에 psycopg2 추가
- [ ] database.py 연결 문자열 변경
- [ ] .env 파일 생성 및 설정
- [ ] python-dotenv 설치
- [ ] 테이블 생성 (init_db.py)
- [ ] 연결 테스트
- [ ] 애플리케이션 실행 확인
- [ ] 데이터 마이그레이션 (필요시)

---

이제 PostgreSQL을 사용할 준비가 완료되었습니다! 🎉

