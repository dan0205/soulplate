"""
Model API 테스트 스크립트
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_health():
    """Health check 테스트"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_recommend():
    """Recommend 엔드포인트 테스트"""
    print("\nTesting /recommend endpoint...")
    
    # 샘플 데이터
    data = {
        "user_id": "test_user_123",
        "user_features": {
            "age": 30.0,
            "review_count": 50.0,
            "useful": 20.0,
            "average_stars": 4.5
        },
        "top_k": 5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=data,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Model info 엔드포인트 테스트"""
    print("\nTesting /model/info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Model API Test")
    print("=" * 60)
    print(f"Target: {BASE_URL}")
    print("\nWaiting for server to start...")
    time.sleep(2)
    
    # Tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Recommendation", test_recommend()))
    results.append(("Model Info", test_model_info()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {passed}/{total} passed")

