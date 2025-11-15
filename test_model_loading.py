"""모델 로딩 테스트"""
import sys
sys.path.append('.')

try:
    print("1. Importing model_loader...")
    from backend_model.model_loader import get_model_loader
    
    print("2. Getting model loader instance...")
    loader = get_model_loader()
    
    print("3. Loading all resources...")
    loader.load_all()
    
    print("\n[SUCCESS] All models loaded successfully!")
    print(f"  - User tower: {loader.user_tower is not None}")
    print(f"  - FAISS index: {loader.faiss_index is not None}")
    print(f"  - Num items: {loader.faiss_index.ntotal if loader.faiss_index else 0}")
    
except Exception as e:
    print(f"\n[ERROR] Failed to load models: {e}")
    import traceback
    traceback.print_exc()







