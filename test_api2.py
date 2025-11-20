"""API 테스트 2"""
import requests
import time

print("Waiting 10 seconds for server...")
time.sleep(10)

try:
    print("\n[TEST] GET /api/businesses")
    r = requests.get('http://localhost:8000/api/businesses?skip=0&limit=5')
    print(f"  Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        print(f"  Businesses: {len(data)}")
        if len(data) > 0:
            business = data[0]
            print(f"\n  First business details:")
            print(f"    Name: {business['name']}")
            print(f"    Stars: {business['stars']}")
            print(f"    Review count: {business['review_count']}")
            print(f"    Has top_features: {'top_features' in business}")
            if 'top_features' in business:
                print(f"    Top features: {business['top_features'][:3]}")
        print("\n[SUCCESS] API is working!")
    else:
        print(f"  Error: {r.text}")
        
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

















