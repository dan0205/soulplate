"""API 테스트"""
import requests
import time

print("Waiting for server to start...")
time.sleep(5)

try:
    print("\n[TEST] GET /api/businesses")
    r = requests.get('http://localhost:8000/api/businesses?skip=0&limit=5')
    print(f"  Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        print(f"  Businesses: {len(data)}")
        if len(data) > 0:
            print(f"  First business: {data[0]['name']}")
            print(f"  Has top_features: {'top_features' in data[0]}")
            if 'top_features' in data[0]:
                print(f"  Top features: {data[0]['top_features']}")
        print("\n[SUCCESS] API is working!")
    else:
        print(f"  Error: {r.text}")
        
except Exception as e:
    print(f"\n[ERROR] {e}")









