"""
CORS 설정 업데이트 스크립트

배포 완료 후 실제 URL로 CORS 설정을 업데이트합니다.

Usage:
    python scripts/update_cors.py <frontend_url> <web_backend_url>

Example:
    python scripts/update_cors.py https://soulplate.vercel.app https://restaurant-web-api-xxx.koyeb.app
"""

import sys
import os
import re

def update_cors_settings(frontend_url, web_backend_url):
    """
    백엔드 CORS 설정을 실제 배포 URL로 업데이트
    """
    
    # backend_web/main.py 업데이트
    web_main_path = "backend_web/main.py"
    
    print(f"🔄 {web_main_path} 업데이트 중...")
    
    with open(web_main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # CORS 설정 찾기 및 업데이트
    cors_pattern = r'(app\.add_middleware\s*\(\s*CORSMiddleware,\s*allow_origins\s*=\s*\[)([^\]]*?)(\])'
    
    new_origins = f'''
        "http://localhost:3000",  # 로컬 개발
        "{frontend_url}",  # 프로덕션
    '''
    
    updated_content = re.sub(
        cors_pattern,
        rf'\g<1>{new_origins}\g<3>',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    with open(web_main_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"✅ {web_main_path} 업데이트 완료")
    
    # backend_model/main.py 업데이트
    model_main_path = "backend_model/main.py"
    
    print(f"🔄 {model_main_path} 업데이트 중...")
    
    with open(model_main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_origins_model = f'''
        "http://localhost:8000",  # 로컬 Web Backend
        "{web_backend_url}",  # 프로덕션
    '''
    
    updated_content = re.sub(
        cors_pattern,
        rf'\g<1>{new_origins_model}\g<3>',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    with open(model_main_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"✅ {model_main_path} 업데이트 완료")
    print()
    print("📝 이제 변경사항을 커밋하고 푸시하세요:")
    print("   git add backend_web/main.py backend_model/main.py")
    print("   git commit -m 'Update CORS with production URLs'")
    print("   git push origin master")
    print()
    print("⏱️  Koyeb와 Vercel이 자동으로 재배포합니다 (2-3분 소요)")

def main():
    if len(sys.argv) < 3:
        print("❌ 사용법: python scripts/update_cors.py <frontend_url> <web_backend_url>")
        print()
        print("예시:")
        print("   python scripts/update_cors.py \\")
        print("       https://soulplate.vercel.app \\")
        print("       https://restaurant-web-api-xxx.koyeb.app")
        sys.exit(1)
    
    frontend_url = sys.argv[1].rstrip('/')
    web_backend_url = sys.argv[2].rstrip('/')
    
    print("🌐 CORS 설정 업데이트")
    print(f"   프론트엔드: {frontend_url}")
    print(f"   Web Backend: {web_backend_url}")
    print()
    
    update_cors_settings(frontend_url, web_backend_url)

if __name__ == "__main__":
    main()

