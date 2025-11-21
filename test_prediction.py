"""
AI 예측 테스트 스크립트
"""
import requests
import json

# 테스트 데이터 (실제 DB에 있는 user/business와 유사하게)
test_request = {
    "user_data": {
        "review_count": 10,
        "useful": 5,
        "compliment": 2,
        "fans": 1,
        "average_stars": 4.2,
        "yelping_since_days": 1000,
        "absa_features": {
            "맛_긍정": 0.85,
            "맛_부정": 0.05,
            "맛_중립": 0.10,
            # ... 나머지 ABSA 피처들
        }
    },
    "business_data": {
        "stars": 4.5,
        "review_count": 100,
        "latitude": 37.5,
        "longitude": -122.4,
        "absa_features": {
            "맛_긍정": 0.90,
            "맛_부정": 0.03,
            "맛_중립": 0.07,
            # ... 나머지 ABSA 피처들
        }
    }
}

print("=" * 80)
print("AI 예측 테스트")
print("=" * 80)

try:
    # backend_model API 호출
    response = requests.post(
        "http://localhost:8001/predict_rating",
        json=test_request,
        timeout=10.0
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResponse:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)




















