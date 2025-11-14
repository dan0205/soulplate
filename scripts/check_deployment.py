"""
배포 상태 체크 스크립트

배포된 서비스들의 상태를 확인합니다.

Usage:
    python scripts/check_deployment.py

환경 변수로 URL 설정:
    export FRONTEND_URL="https://soulplate.vercel.app"
    export WEB_BACKEND_URL="https://restaurant-web-api-xxx.koyeb.app"
    export MODEL_BACKEND_URL="https://restaurant-model-api-xxx.koyeb.app"
"""

import os
import sys

def check_service(name, url):
    """서비스 상태 확인"""
    try:
        import requests
    except ImportError:
        print("❌ requests 패키지가 설치되지 않았습니다.")
        print("설치 명령어: pip install requests")
        sys.exit(1)
    
    print(f"🔍 {name} 체크 중...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"   ✅ 정상 (200 OK)")
            return True
        else:
            print(f"   ⚠️  응답 코드: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"   ❌ 타임아웃 (10초)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ 연결 실패")
        return False
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 배포 상태 체크")
    print("=" * 60)
    print()
    
    # 환경 변수에서 URL 읽기
    frontend_url = os.getenv("FRONTEND_URL")
    web_backend_url = os.getenv("WEB_BACKEND_URL")
    model_backend_url = os.getenv("MODEL_BACKEND_URL")
    
    if not all([frontend_url, web_backend_url, model_backend_url]):
        print("⚠️  환경 변수가 설정되지 않았습니다.")
        print()
        print("다음 환경 변수를 설정하세요:")
        print("   export FRONTEND_URL=\"https://soulplate.vercel.app\"")
        print("   export WEB_BACKEND_URL=\"https://restaurant-web-api-xxx.koyeb.app\"")
        print("   export MODEL_BACKEND_URL=\"https://restaurant-model-api-xxx.koyeb.app\"")
        print()
        
        # 대화형으로 입력받기
        frontend_url = input("프론트엔드 URL: ").strip()
        web_backend_url = input("Web Backend URL: ").strip()
        model_backend_url = input("Model Backend URL: ").strip()
        print()
    
    # 각 서비스 체크
    results = {}
    
    results['frontend'] = check_service(
        "프론트엔드 (Vercel)",
        frontend_url
    )
    print()
    
    results['web_backend'] = check_service(
        "Web Backend (Koyeb)",
        f"{web_backend_url}/health" if "/health" not in web_backend_url else web_backend_url
    )
    print()
    
    results['model_backend'] = check_service(
        "Model Backend (Koyeb)",
        f"{model_backend_url}/health" if "/health" not in model_backend_url else model_backend_url
    )
    print()
    
    # 결과 요약
    print("=" * 60)
    print("📊 체크 결과 요약")
    print("=" * 60)
    
    all_ok = all(results.values())
    
    for service, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {service}: {'정상' if status else '오류'}")
    
    print()
    
    if all_ok:
        print("🎉 모든 서비스가 정상 작동 중입니다!")
        print()
        print("다음 단계:")
        print("1. 프론트엔드에서 회원가입 테스트")
        print("2. 비즈니스 목록 조회 테스트")
        print("3. AI 추천 기능 테스트")
        print("4. QR 코드 생성:")
        print(f"   python scripts/generate_qr.py {frontend_url}")
    else:
        print("⚠️  일부 서비스에 문제가 있습니다.")
        print()
        print("문제 해결:")
        print("1. Koyeb/Vercel 대시보드에서 로그 확인")
        print("2. 환경 변수 설정 확인")
        print("3. CORS 설정 확인")
        print("4. 빌드 로그 확인")

if __name__ == "__main__":
    main()

