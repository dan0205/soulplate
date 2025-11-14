"""
QR 코드 생성 스크립트

Usage:
    python scripts/generate_qr.py <URL>

Example:
    python scripts/generate_qr.py https://soulplate.vercel.app
"""

import sys
import os

def generate_qr_code(url, output_path="restaurant_qr_code.png"):
    """
    QR 코드 생성
    
    Args:
        url: QR 코드에 포함할 URL
        output_path: 출력 파일 경로
    """
    try:
        import qrcode
    except ImportError:
        print("❌ qrcode 패키지가 설치되지 않았습니다.")
        print("설치 명령어: pip install qrcode[pil]")
        sys.exit(1)
    
    print(f"🔄 QR 코드 생성 중...")
    print(f"📍 URL: {url}")
    
    # QR 코드 생성
    qr = qrcode.QRCode(
        version=1,  # 크기 (1-40)
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,  # 픽셀 크기
        border=4,  # 테두리 크기
    )
    
    qr.add_data(url)
    qr.make(fit=True)
    
    # 이미지 생성
    img = qr.make_image(fill_color="black", back_color="white")
    
    # 파일 저장
    img.save(output_path)
    
    print(f"✅ QR 코드가 생성되었습니다: {output_path}")
    print(f"📏 이미지 크기: {img.size[0]}x{img.size[1]} pixels")
    print(f"📱 스마트폰으로 스캔하여 접속하세요!")
    
    # 절대 경로 출력
    abs_path = os.path.abspath(output_path)
    print(f"📂 절대 경로: {abs_path}")

def main():
    if len(sys.argv) < 2:
        print("❌ 사용법: python scripts/generate_qr.py <URL>")
        print("예시: python scripts/generate_qr.py https://soulplate.vercel.app")
        sys.exit(1)
    
    url = sys.argv[1]
    
    # URL 유효성 검사
    if not url.startswith(("http://", "https://")):
        print("⚠️  경고: URL이 http:// 또는 https://로 시작하지 않습니다.")
        response = input("계속하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("취소되었습니다.")
            sys.exit(0)
    
    # 출력 파일명 (선택사항)
    output_path = sys.argv[2] if len(sys.argv) > 2 else "restaurant_qr_code.png"
    
    generate_qr_code(url, output_path)

if __name__ == "__main__":
    main()

