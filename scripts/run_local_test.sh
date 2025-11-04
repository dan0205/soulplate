#!/bin/bash

# 로컬 환경에서 전체 시스템 테스트 스크립트

echo "========================================="
echo "Two-Tower System Local Test"
echo "========================================="

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Model API 시작
echo -e "${YELLOW}[1/4] Starting Model API...${NC}"
cd "$(dirname "$0")/.."
python -m uvicorn backend_model.main:app --host 0.0.0.0 --port 8001 > /dev/null 2>&1 &
MODEL_API_PID=$!
echo -e "${GREEN}✓ Model API started (PID: $MODEL_API_PID)${NC}"

# 2. Web API 시작
echo -e "${YELLOW}[2/4] Starting Web API...${NC}"
python -m uvicorn backend_web.main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
WEB_API_PID=$!
echo -e "${GREEN}✓ Web API started (PID: $WEB_API_PID)${NC}"

# 3. Frontend 시작
echo -e "${YELLOW}[3/4] Starting Frontend...${NC}"
cd frontend
npm start > /dev/null 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"

# 4. 서버 시작 대기
echo -e "${YELLOW}[4/4] Waiting for servers to be ready...${NC}"
sleep 15

# Health check
echo ""
echo "========================================="
echo "Health Check"
echo "========================================="

# Model API
if curl -s http://localhost:8001/health > /dev/null; then
    echo -e "${GREEN}✓ Model API is healthy${NC}"
else
    echo -e "${RED}✗ Model API is not responding${NC}"
fi

# Web API
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo -e "${GREEN}✓ Web API is healthy${NC}"
else
    echo -e "${RED}✗ Web API is not responding${NC}"
fi

# Frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✓ Frontend is healthy${NC}"
else
    echo -e "${RED}✗ Frontend is not responding${NC}"
fi

echo ""
echo "========================================="
echo "System is running!"
echo "========================================="
echo ""
echo "Access points:"
echo "  - Frontend:  http://localhost:3000"
echo "  - Web API:   http://localhost:8000/docs"
echo "  - Model API: http://localhost:8001/docs"
echo ""
echo "Demo accounts:"
echo "  - testuser / test123"
echo "  - alice / alice123"
echo "  - bob / bob123"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Ctrl+C 핸들러
trap "echo ''; echo 'Stopping services...'; kill $MODEL_API_PID $WEB_API_PID $FRONTEND_PID 2>/dev/null; echo 'All services stopped'; exit" INT

# 백그라운드 프로세스 대기
wait

