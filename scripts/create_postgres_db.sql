-- PostgreSQL 데이터베이스 생성 스크립트
-- Windows에서 실행: psql -U postgres
-- 비밀번호: 0205

-- 데이터베이스 생성
CREATE DATABASE two_tower_db;

-- 사용자 생성 (비밀번호: twotower2024)
CREATE USER two_tower_user WITH PASSWORD 'twotower2024';

-- 권한 부여
GRANT ALL PRIVILEGES ON DATABASE two_tower_db TO two_tower_user;

-- two_tower_db에 연결
\c two_tower_db

-- PostgreSQL 15 이상인 경우 추가 권한 부여
GRANT ALL ON SCHEMA public TO two_tower_user;

-- 확인
\l

