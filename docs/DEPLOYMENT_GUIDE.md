# 🚀 배포 가이드 (Phase 5)

이 문서는 Two-Tower 추천 시스템을 실제 서버에 배포하는 방법을 안내합니다.

## 사전 준비

### 필요한 것들
- 클라우드 VM (AWS EC2, GCP Compute Engine 등)
- OS: Ubuntu 22.04 LTS
- 최소 사양: 2 CPU, 4GB RAM, 20GB Storage
- 도메인 (선택사항)

## Step 1: 서버 준비

```bash
# SSH로 서버 접속
ssh user@your-server-ip

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y nginx python3-pip python3-venv nodejs npm git
```

## Step 2: 방화벽 설정

```bash
# 필요한 포트 열기
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS (SSL 사용 시)
sudo ufw enable
```

## Step 3: 코드 배포

```bash
# 작업 디렉터리 생성
mkdir -p /var/www/two-tower
cd /var/www/two-tower

# Git 클론
git clone <your-repo-url> .

# 또는 파일 업로드
# scp -r ./demo/* user@server:/var/www/two-tower/
```

## Step 4: Backend Model API 설정

```bash
cd /var/www/two-tower/backend_model

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 모델 파일 복사 (로컬에서 학습한 모델)
# scp -r ./models/* user@server:/var/www/two-tower/models/
```

## Step 5: Backend Web API 설정

```bash
cd /var/www/two-tower/backend_web

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 데이터베이스 초기화
python /var/www/two-tower/scripts/init_db.py
```

## Step 6: Frontend 빌드

```bash
cd /var/www/two-tower/frontend

# Node 패키지 설치
npm install

# 프로덕션 빌드
npm run build
```

## Step 7: Systemd 서비스 설정

### Backend Model API 서비스

```bash
# /etc/systemd/system/model-api.service
sudo nano /etc/systemd/system/model-api.service
```

내용:
```ini
[Unit]
Description=Two-Tower Model API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/two-tower
Environment="PATH=/var/www/two-tower/backend_model/venv/bin"
ExecStart=/var/www/two-tower/backend_model/venv/bin/gunicorn -k uvicorn.workers.UvicornWorker backend_model.main:app --bind 0.0.0.0:8001 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
```

### Backend Web API 서비스

```bash
# /etc/systemd/system/web-api.service
sudo nano /etc/systemd/system/web-api.service
```

내용:
```ini
[Unit]
Description=Two-Tower Web API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/two-tower
Environment="PATH=/var/www/two-tower/backend_web/venv/bin"
ExecStart=/var/www/two-tower/backend_web/venv/bin/gunicorn -k uvicorn.workers.UvicornWorker backend_web.main:app --bind 0.0.0.0:8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

### 서비스 시작

```bash
# 서비스 등록 및 시작
sudo systemctl daemon-reload
sudo systemctl enable model-api web-api
sudo systemctl start model-api web-api

# 상태 확인
sudo systemctl status model-api
sudo systemctl status web-api
```

## Step 8: Nginx 설정

```bash
sudo nano /etc/nginx/sites-available/two-tower
```

내용:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # 또는 서버 IP

    # Frontend 정적 파일
    location / {
        root /var/www/two-tower/frontend/build;
        try_files $uri /index.html;
    }

    # Web API 프록시
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # 로그
    access_log /var/log/nginx/two-tower-access.log;
    error_log /var/log/nginx/two-tower-error.log;
}
```

```bash
# Nginx 설정 활성화
sudo ln -s /etc/nginx/sites-available/two-tower /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default  # 기본 사이트 비활성화

# Nginx 테스트 및 재시작
sudo nginx -t
sudo systemctl restart nginx
```

## Step 9: SSL 인증서 설정 (선택사항)

```bash
# Certbot 설치
sudo apt install -y certbot python3-certbot-nginx

# SSL 인증서 발급
sudo certbot --nginx -d your-domain.com

# 자동 갱신 테스트
sudo certbot renew --dry-run
```

## Step 10: 최종 테스트

```bash
# 브라우저에서 접속
# http://your-server-ip
# 또는
# https://your-domain.com
```

## 모니터링

### 로그 확인

```bash
# Model API 로그
sudo journalctl -u model-api -f

# Web API 로그
sudo journalctl -u web-api -f

# Nginx 로그
sudo tail -f /var/log/nginx/two-tower-error.log
sudo tail -f /var/log/nginx/two-tower-access.log
```

### 서비스 재시작

```bash
# 서비스 재시작
sudo systemctl restart model-api
sudo systemctl restart web-api
sudo systemctl restart nginx
```

## 트러블슈팅

### 서비스가 시작되지 않을 때
```bash
# 로그 확인
sudo journalctl -u model-api -n 50
sudo journalctl -u web-api -n 50

# 권한 확인
sudo chown -R www-data:www-data /var/www/two-tower
```

### 데이터베이스 오류
```bash
# 데이터베이스 재초기화
cd /var/www/two-tower
source backend_web/venv/bin/activate
python scripts/init_db.py
```

### Frontend가 표시되지 않을 때
```bash
# Frontend 재빌드
cd /var/www/two-tower/frontend
npm run build

# Nginx 재시작
sudo systemctl restart nginx
```

## 보안 강화

1. **환경 변수 설정**: `.env` 파일에 비밀키 저장
2. **방화벽 설정**: 필요한 포트만 열기
3. **정기 업데이트**: `sudo apt update && sudo apt upgrade -y`
4. **백업**: 정기적인 데이터베이스 및 모델 백업

## 성능 최적화

1. **Gunicorn Workers**: CPU 코어 수에 맞게 조정
2. **Nginx 캐싱**: 정적 파일 캐싱 설정
3. **데이터베이스**: PostgreSQL로 전환 고려
4. **Redis**: 추천 결과 캐싱

---

배포 완료! 🎉

