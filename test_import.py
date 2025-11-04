"""
간단한 import 테스트
"""

try:
    print("Testing imports...")
    from backend_model.main import app
    from backend_model.model_loader import get_model_loader
    from backend_model.schemas import RecommendRequest, RecommendResponse
    print("[SUCCESS] All imports successful!")
    
    print("\nTesting model loader...")
    loader = get_model_loader()
    print(f"Loader created: {loader}")
    print("[SUCCESS] Model loader initialized!")
    
    print("\nModel API is ready!")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

