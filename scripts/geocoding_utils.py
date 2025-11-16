"""
Kakao Local API를 사용한 주소 -> 위도/경도 변환 유틸리티

사용법:
    from geocoding_utils import get_coordinates
    
    lat, lng = get_coordinates("서울특별시 강남구 테헤란로 123")
    if lat and lng:
        print(f"위도: {lat}, 경도: {lng}")
    else:
        print("주소 변환 실패")
"""

import os
import time
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_coordinates(address: str) -> tuple:
    """
    Kakao Local API를 사용하여 주소를 위도/경도로 변환
    
    Args:
        address: 변환할 주소 (예: "서울특별시 강남구 테헤란로 123")
    
    Returns:
        (latitude, longitude) 튜플, 실패 시 (None, None)
    """
    # 환경변수에서 API 키 가져오기
    api_key = os.getenv("KAKAO_REST_API_KEY")
    
    if not api_key:
        logger.error("KAKAO_REST_API_KEY 환경변수가 설정되지 않았습니다.")
        logger.error("다음 링크에서 API 키를 발급받으세요: https://developers.kakao.com/")
        return None, None
    
    # Kakao Local API 엔드포인트
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    
    # 헤더 설정
    headers = {
        "Authorization": f"KakaoAK {api_key}"
    }
    
    # 쿼리 파라미터
    params = {
        "query": address
    }
    
    try:
        # API 요청
        logger.info(f"주소 변환 시도: {address}")
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        # 요청 제한 대응 (0.1초 딜레이)
        time.sleep(0.1)
        
        # 응답 확인
        if response.status_code == 200:
            data = response.json()
            
            # 결과가 있는지 확인
            if data.get("documents") and len(data["documents"]) > 0:
                # 첫 번째 결과 사용
                result = data["documents"][0]
                
                # 도로명 주소 또는 지번 주소에서 좌표 추출
                if "road_address" in result and result["road_address"]:
                    lat = float(result["road_address"]["y"])
                    lng = float(result["road_address"]["x"])
                elif "address" in result and result["address"]:
                    lat = float(result["address"]["y"])
                    lng = float(result["address"]["x"])
                else:
                    logger.warning(f"주소에서 좌표를 찾을 수 없습니다: {address}")
                    return None, None
                
                logger.info(f"✅ 변환 성공: {address} -> ({lat}, {lng})")
                return lat, lng
            else:
                logger.warning(f"주소 검색 결과가 없습니다: {address}")
                return None, None
        
        elif response.status_code == 401:
            logger.error("Kakao API 인증 실패. API 키를 확인하세요.")
            return None, None
        
        elif response.status_code == 429:
            logger.error("Kakao API 요청 제한 초과. 잠시 후 다시 시도하세요.")
            return None, None
        
        else:
            logger.error(f"Kakao API 오류: {response.status_code}")
            return None, None
    
    except requests.exceptions.Timeout:
        logger.error(f"Kakao API 요청 타임아웃: {address}")
        return None, None
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Kakao API 요청 실패: {e}")
        return None, None
    
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}")
        return None, None


def test_geocoding():
    """geocoding 테스트 함수"""
    test_addresses = [
        "서울특별시 강남구 테헤란로 123",
        "경기도 수원시 영통구 월드컵로 206",
        "경기 수원시 팔달구 아주로27번길 10-20 1층",
        "존재하지 않는 주소 12345"
    ]
    
    print("\n=== Geocoding 테스트 ===\n")
    
    for address in test_addresses:
        lat, lng = get_coordinates(address)
        if lat and lng:
            print(f"✅ {address}")
            print(f"   위도: {lat}, 경도: {lng}\n")
        else:
            print(f"❌ {address}")
            print(f"   변환 실패\n")


if __name__ == "__main__":
    # API 키 확인
    api_key = os.getenv("KAKAO_REST_API_KEY")
    if not api_key:
        print("❌ KAKAO_REST_API_KEY 환경변수가 설정되지 않았습니다.")
        print("\n다음 단계를 따라 API 키를 설정하세요:")
        print("1. https://developers.kakao.com/ 접속")
        print("2. '내 애플리케이션' -> '애플리케이션 추가하기'")
        print("3. REST API 키 복사")
        print("4. 환경변수 설정:")
        print("   - Windows: set KAKAO_REST_API_KEY=your_api_key")
        print("   - Linux/Mac: export KAKAO_REST_API_KEY=your_api_key")
    else:
        print(f"✅ Kakao API 키 확인됨: {api_key[:10]}...")
        test_geocoding()

