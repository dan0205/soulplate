"""
예측 서비스 테스트
"""
import sys
sys.path.append('.')

from backend_web.database import SessionLocal
from backend_web import models
from backend_model.prediction_service import get_prediction_service

print("=" * 80)
print("예측 서비스 테스트")
print("=" * 80)

# DB에서 샘플 데이터 가져오기
print("\n[1/3] 샘플 데이터 로딩 중...")
db = SessionLocal()

# Yelp 사용자 1명 (yelp_user_id가 있는 사용자)
user = db.query(models.User).filter(models.User.yelp_user_id.isnot(None)).first()
if not user:
    print("[ERROR] Yelp 사용자를 찾을 수 없습니다.")
    sys.exit(1)

print(f"  User: {user.username} (ID: {user.id})")
print(f"    review_count: {user.review_count}")
print(f"    average_stars: {user.average_stars}")
print(f"    ABSA features: {len(user.absa_features) if user.absa_features else 0}개")

# 비즈니스 1개
business = db.query(models.Business).first()
if not business:
    print("[ERROR] 비즈니스를 찾을 수 없습니다.")
    sys.exit(1)

print(f"\n  Business: {business.name} (ID: {business.business_id})")
print(f"    stars: {business.stars}")
print(f"    review_count: {business.review_count}")
print(f"    ABSA features: {len(business.absa_features) if business.absa_features else 0}개")

# 예측 서비스 초기화
print("\n[2/3] 예측 서비스 로딩 중...")
pred_service = get_prediction_service()

# 예측 수행
print("\n[3/3] 예측 수행 중...")
try:
    result = pred_service.predict_rating(
        user_data={
            "review_count": user.review_count,
            "useful": user.useful,
            "compliment": user.compliment,
            "fans": user.fans,
            "average_stars": user.average_stars,
            "yelping_since_days": user.yelping_since_days,
            "absa_features": user.absa_features or {}
        },
        business_data={
            "stars": business.stars,
            "review_count": business.review_count,
            "latitude": business.latitude,
            "longitude": business.longitude,
            "absa_features": business.absa_features or {}
        }
    )
    
    print("\n" + "=" * 80)
    print("예측 결과")
    print("=" * 80)
    print(f"  DeepFM 예측:      {result['deepfm_rating']}")
    print(f"  Multi-Tower 예측: {result['multitower_rating']}")
    print(f"  앙상블 예측:      {result['ensemble_rating']}")
    print(f"  신뢰도:           {result['confidence']:.2%}")
    
    print("\n[SUCCESS] 예측 완료!")
    
    # 예측값이 합리적인지 확인
    if result['deepfm_rating'] < 1.0 or result['deepfm_rating'] > 5.0:
        print(f"[WARNING] DeepFM 예측값이 범위를 벗어났습니다: {result['deepfm_rating']}")
    
    if result['multitower_rating'] and (result['multitower_rating'] < 1.0 or result['multitower_rating'] > 5.0):
        print(f"[WARNING] Multi-Tower 예측값이 범위를 벗어났습니다: {result['multitower_rating']}")
    
    if result['deepfm_rating'] == 1.0:
        print(f"[WARNING] DeepFM이 1.0만 예측하고 있습니다. 모델 문제일 수 있습니다.")
    
except Exception as e:
    print(f"\n[ERROR] 예측 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    db.close()

print("\n" + "=" * 80)

