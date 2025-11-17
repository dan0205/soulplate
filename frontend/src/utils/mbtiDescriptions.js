/**
 * 음식 취향 MBTI 타입 설명
 */

export const MBTI_TYPE_DESCRIPTIONS = {
  "SQAF": {
    "name": "미식가 감성파",
    "description": "강렬한 맛을 즐기며, 고급 재료와 멋진 분위기, 친절한 서비스를 중시하는 타입",
    "recommendations": ["파인다이닝 레스토랑", "감성 카페", "프리미엄 맛집"]
  },
  "SQAI": {
    "name": "품격있는 미식가",
    "description": "품질과 분위기를 중시하지만, 서비스보다는 음식 자체에 집중하는 타입",
    "recommendations": ["조용한 고급 레스토랑", "품질 중심 맛집", "분위기 좋은 술집"]
  },
  "SQCF": {
    "name": "실용적 미식가",
    "description": "품질 좋은 음식과 친절한 서비스를 원하지만, 접근성도 중요하게 생각하는 타입",
    "recommendations": ["동네 맛집", "품질 좋은 프랜차이즈", "친절한 가게"]
  },
  "SQCI": {
    "name": "맛집 헌터",
    "description": "강렬한 맛과 품질을 최우선으로, 접근성 좋은 곳이라면 어디든 찾아가는 타입",
    "recommendations": ["숨은 맛집", "로컬 맛집", "품질 최우선 식당"]
  },
  "SVAF": {
    "name": "감성 외식러",
    "description": "강한 맛과 가성비, 좋은 분위기와 서비스를 모두 원하는 균형잡힌 타입",
    "recommendations": ["분위기 좋은 맛집", "가성비 좋은 감성 카페", "SNS 핫플"]
  },
  "SVAI": {
    "name": "분위기 중시형",
    "description": "맛과 가성비, 분위기는 중요하지만 혼자 조용히 즐기는 것을 선호",
    "recommendations": ["혼밥 카페", "분위기 좋은 술집", "조용한 맛집"]
  },
  "SVCF": {
    "name": "합리적 외식러",
    "description": "강한 맛과 가성비, 편의성과 친절한 서비스를 중시하는 실속형",
    "recommendations": ["가성비 맛집", "동네 맛집", "편한 분위기 식당"]
  },
  "SVCI": {
    "name": "효율적 미식가",
    "description": "맛과 가성비, 접근성을 최우선으로 빠르게 식사하는 타입",
    "recommendations": ["가성비 맛집", "배달 맛집", "빠른 식당"]
  },
  "MQAF": {
    "name": "우아한 식도락가",
    "description": "부드러운 맛과 품질, 멋진 분위기와 서비스를 중시하는 세련된 타입",
    "recommendations": ["고급 레스토랑", "브런치 카페", "프리미엄 디저트 카페"]
  },
  "MQAI": {
    "name": "조용한 미식가",
    "description": "담백한 맛과 품질, 분위기를 중시하며 혼자만의 시간을 즐기는 타입",
    "recommendations": ["조용한 카페", "품질 좋은 일식당", "분위기 좋은 와인바"]
  },
  "MQCF": {
    "name": "편안한 미식가",
    "description": "담백한 맛과 품질, 편의성과 친절함을 중시하는 균형잡힌 타입",
    "recommendations": ["동네 맛집", "편한 카페", "친절한 식당"]
  },
  "MQCI": {
    "name": "건강 지향형",
    "description": "담백하고 품질 좋은 음식, 접근성을 중시하는 건강한 식사 추구형",
    "recommendations": ["샐러드 전문점", "건강식 레스토랑", "웰빙 카페"]
  },
  "MVAF": {
    "name": "가족 외식러",
    "description": "부드러운 맛, 가성비, 편의성과 친절한 서비스를 모두 중시하는 가족 외식형",
    "recommendations": ["가족 레스토랑", "뷔페", "친절한 한식당"]
  },
  "MVAI": {
    "name": "혼밥 마스터",
    "description": "담백한 맛과 가성비, 분위기를 중시하며 혼자 식사를 즐기는 타입",
    "recommendations": ["혼밥 맛집", "조용한 카페", "분위기 좋은 식당"]
  },
  "MVCF": {
    "name": "실속 외식러",
    "description": "담백한 맛, 가성비, 편의성, 서비스를 모두 고려하는 가장 균형잡힌 타입",
    "recommendations": ["동네 맛집", "가성비 한식당", "편한 카페"]
  },
  "MVCI": {
    "name": "실속 혼밥러",
    "description": "담백한 맛, 가성비, 편의성을 중시하며 혼자 조용히 먹는 것을 선호",
    "recommendations": ["배달 맛집", "편의점", "간단한 식사"]
  }
};

/**
 * MBTI 타입 정보를 가져옵니다
 * @param {string} mbtiType - MBTI 타입 (예: "SQAF")
 * @returns {object} 타입 정보 (name, description, recommendations)
 */
export const getMBTIInfo = (mbtiType) => {
  return MBTI_TYPE_DESCRIPTIONS[mbtiType] || {
    name: "알 수 없음",
    description: "타입 정보를 찾을 수 없습니다",
    recommendations: []
  };
};

