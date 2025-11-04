"""
Web API import 테스트
"""

try:
    print("Testing imports...")
    from backend_web.main import app
    from backend_web import models, schemas, auth
    print("[SUCCESS] All imports successful!")
    
    print("\nWeb API is ready!")
    print("To run:")
    print("  python -m uvicorn backend_web.main:app --host 0.0.0.0 --port 8000")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

