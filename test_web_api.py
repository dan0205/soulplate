"""
Web API 테스트 스크립트
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# 전역 토큰 저장
access_token = None

def test_register():
    """회원가입 테스트"""
    print("\n[1] Testing /api/auth/register...")
    data = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "newpass123",
        "age": 28,
        "gender": "F"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/register", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code in [200, 201]
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_login():
    """로그인 테스트"""
    global access_token
    print("\n[2] Testing /api/auth/login...")
    data = {
        "username": "testuser",
        "password": "test123"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            data=data  # OAuth2PasswordRequestForm uses form data
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            access_token = result["access_token"]
            print(f"Token saved: {access_token[:20]}...")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_me():
    """현재 사용자 정보 테스트"""
    print("\n[3] Testing /api/auth/me...")
    
    if not access_token:
        print("No token available")
        return False
    
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_businesses():
    """비즈니스 목록 테스트"""
    print("\n[4] Testing /api/businesses...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/businesses?limit=5")
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Retrieved {len(result)} businesses")
        if result:
            print(f"First business: {result[0]['name']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_recommendations():
    """추천 테스트"""
    print("\n[5] Testing /api/recommendations...")
    
    if not access_token:
        print("No token available")
        return False
    
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(
            f"{BASE_URL}/api/recommendations?top_k=5",
            headers=headers,
            timeout=15
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print(f"Received {len(result.get('recommendations', []))} recommendations")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Web API Test")
    print("=" * 70)
    print(f"Target: {BASE_URL}")
    print("\nMake sure both Model API (8001) and Web API (8000) are running!")
    
    import time
    print("\nWaiting 2 seconds for server...")
    time.sleep(2)
    
    # Tests
    results = []
    results.append(("Register", test_register()))
    results.append(("Login", test_login()))
    results.append(("Get Current User", test_me()))
    results.append(("List Businesses", test_businesses()))
    results.append(("Get Recommendations", test_recommendations()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {passed}/{total} passed")

