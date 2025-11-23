"""
OAuth 설정 및 클라이언트
"""

import os
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Starlette Config
config = Config('.env')

# OAuth 클라이언트 생성
oauth = OAuth()

# Google OAuth 등록
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'prompt': 'select_account',  # 항상 계정 선택 화면 표시
    }
)

