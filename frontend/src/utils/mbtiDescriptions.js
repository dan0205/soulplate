/**
 * 음식 취향 MBTI 타입 설명 (새 축 시스템: S/M, A/O, P/C, A/O)
 * 축 1: S(Strong) vs M(Mild), 축 2: A(Ambiance) vs O(Optimized), 
 * 축 3: P(Premium) vs C(Cost Effective), 축 4: A(All Together) vs O(On My Own)
 */

export const MBTI_TYPE_DESCRIPTIONS = {
  "SAPA": {
    "name": "프리미엄 소셜러",
    "description": "강렬한 맛을 즐기며, 프리미엄 품질과 멋진 분위기, 함께 즐기는 서비스를 중시하는 타입",
    "recommendations": ["파인다이닝 레스토랑", "프리미엄 바", "고급 한정식"]
  },
  "SAPO": {
    "name": "품격있는 독립가",
    "description": "강렬한 맛과 프리미엄 품질을 중시하며, 멋진 분위기에서 혼자만의 시간을 즐기는 타입",
    "recommendations": ["조용한 고급 레스토랑", "프리미엄 와인바", "품격 있는 일식당"]
  },
  "SOCA": {
    "name": "가성비 소셜러",
    "description": "강렬한 맛과 가성비를 중시하며, 효율적인 접근성과 함께 즐기는 서비스를 선호하는 타입",
    "recommendations": ["가성비 좋은 맛집", "분위기 좋은 술집", "동네 인기 맛집"]
  },
  "SOCO": {
    "name": "실용적 독립가",
    "description": "강렬한 맛과 가성비를 최우선으로, 효율적이고 접근성 좋은 곳에서 혼자 즐기는 타입",
    "recommendations": ["가성비 맛집", "배달 맛집", "효율적인 식당"]
  },
  "MAPA": {
    "name": "우아한 소셜러",
    "description": "부드러운 맛과 프리미엄 품질을 중시하며, 멋진 분위기에서 함께 즐기는 것을 선호하는 타입",
    "recommendations": ["고급 레스토랑", "브런치 카페", "프리미엄 디저트 카페"]
  },
  "MAPO": {
    "name": "조용한 미식가",
    "description": "부드러운 맛과 프리미엄 품질을 중시하며, 멋진 분위기에서 혼자만의 시간을 즐기는 타입",
    "recommendations": ["조용한 카페", "품질 좋은 일식당", "분위기 좋은 와인바"]
  },
  "MOCA": {
    "name": "균형잡힌 소셜러",
    "description": "부드러운 맛과 가성비를 중시하며, 효율적이고 접근성 좋은 곳에서 함께 즐기는 것을 선호하는 타입",
    "recommendations": ["가족 레스토랑", "뷔페", "친절한 한식당"]
  },
  "MOCO": {
    "name": "실속 혼밥러",
    "description": "부드러운 맛과 가성비를 중시하며, 효율적이고 접근성 좋은 곳에서 혼자 조용히 즐기는 타입",
    "recommendations": ["혼밥 맛집", "조용한 카페", "편의점 식사"]
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

