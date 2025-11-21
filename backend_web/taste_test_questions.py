"""
음식 취향 테스트 질문 (새 버전)
- 간단 테스트: 8개 질문 (각 축당 2문항)
- 심화 테스트: 24개 질문 (각 축당 6문항)
- 5점 선택지 (상황 기반 질문)
"""

# 간단 테스트 질문 (8개)
QUICK_TEST_QUESTIONS = [
    # 축 1: 맛의 강도 - Strong vs Mild (2문)
    {
        "id": 1,
        "axis": "flavor_intensity",
        "section": "맛의 강도",
        "question": "스트레스를 받았을 때, 가장 먼저 생각나는 음식은?",
        "options": [
            "혀가 얼얼할 정도로 매운 마라탕이나 불닭",
            "매콤달콤한 떡볶이나 찌개류",
            "상관없음",
            "따뜻한 국수나 부드러운 빵",
            "담백하고 맑은 평양냉면이나 죽"
        ]
    },
    {
        "id": 2,
        "axis": "flavor_intensity",
        "section": "맛의 강도",
        "question": "\"이 집 간이 좀 센데?\"라는 말을 들었을 때 내 반응은?",
        "options": [
            "\"완전 내 스타일! 밥 비벼 먹으면 딱이겠네.\"",
            "\"맛있는데? 짭짤해야 밥맛이 돌지.\"",
            "\"그냥 보통인데?\"",
            "\"물 좀 부을까? 나한테는 좀 짜네.\"",
            "\"으... 너무 자극적이다. 혀가 아려.\""
        ]
    },
    
    # 축 2: 분위기 vs 효율 - Ambiance vs Optimized (2문)
    {
        "id": 3,
        "axis": "dining_environment",
        "section": "분위기 vs 효율",
        "question": "맛집을 고를 때, 용서할 수 없는 최악의 상황은?",
        "options": [
            "인테리어가 구리고 화장실이 더러운 곳 (감성 파괴)",
            "분위기가 어수선하고 플레이팅이 대충인 곳",
            "상관없음",
            "주차장이 없고 역에서 15분 걸어가야 하는 곳",
            "음식 나오는 데 30분 넘게 걸리는 곳 (시간 낭비)"
        ]
    },
    {
        "id": 4,
        "axis": "dining_environment",
        "section": "분위기 vs 효율",
        "question": "인스타그램에 올릴 음식 사진을 찍는다면?",
        "options": [
            "조명, 구도, 배경까지 완벽하게 세팅하고 찍는다.",
            "음식이 나오면 예쁘게 몇 장 찍어본다.",
            "가끔 찍긴 한다.",
            "대충 한 장 찍거나, '인증샷'용으로만 찍는다.",
            "찍을 시간에 한 젓가락이라도 더 빨리 먹는다."
        ]
    },
    
    # 축 3: 비용 기준 - Premium vs Cost-effective (2문)
    {
        "id": 5,
        "axis": "price_sensitivity",
        "section": "비용 기준",
        "question": "한 끼 식사에 10만 원을 써야 한다면?",
        "options": [
            "\"나를 위한 선물!\" 망설임 없이 결제한다.",
            "특별한 날이라면 충분히 쓸 수 있다.",
            "고민되지만 메뉴에 따라 다르다.",
            "좀 부담스럽다. 차라리 2-3번 나눠서 먹고 싶다.",
            "\"미쳤어?\" 그 돈이면 국밥이 몇 그릇인데."
        ]
    },
    {
        "id": 6,
        "axis": "price_sensitivity",
        "section": "비용 기준",
        "question": "식당 리뷰를 볼 때 가장 눈여겨보는 키워드는?",
        "options": [
            "#오마카세 #파인다이닝 #최고급재료",
            "#분위기맛집 #서비스최고",
            "#적당한가격",
            "#가성비갑 #양많음",
            "#무한리필 #할인행사 #혜자"
        ]
    },
    
    # 축 4: 식사 인원 - All together vs On my own (2문)
    {
        "id": 7,
        "axis": "dining_company",
        "section": "식사 인원",
        "question": "금요일 저녁, 가장 이상적인 식사 계획은?",
        "options": [
            "친구들을 모아 왁자지껄한 핫플에서 파티",
            "친한 지인 한두 명과 맛있는 것 먹으며 수다",
            "상황에 따라 다름",
            "이어폰 꽂고 영상 보며 즐기는 혼밥",
            "집에서 편한 옷 입고 혼자 시켜 먹는 배달 음식"
        ]
    },
    {
        "id": 8,
        "axis": "dining_company",
        "section": "식사 인원",
        "question": "식당에서 4인 테이블에 혼자 앉게 된다면?",
        "options": [
            "너무 외롭고 민망해서 밥이 안 넘어간다.",
            "누구라도 불러내고 싶다.",
            "별생각 없다.",
            "오히려 넓어서 좋다.",
            "\"이 구역의 왕은 나.\" 전혀 신경 안 쓴다."
        ]
    }
]


# 심화 테스트 추가 질문 (16개 = 24 - 8)
DEEP_TEST_ADDITIONAL_QUESTIONS = [
    # 축 1: 맛의 강도 추가 4문 (Q9-Q12)
    {
        "id": 9,
        "axis": "flavor_intensity",
        "section": "맛의 강도",
        "question": "곰탕이나 설렁탕이 나왔다. 국물을 한 입 먹어보니 간이 거의 안 되어 있다. 당신의 행동은?",
        "options": [
            "소금, 후추, 다대기(양념장), 깍두기 국물까지 넣어서 빨갛게 만든다.",
            "소금과 후추를 넉넉히 뿌려 짭짤하게 맞춘다.",
            "소금을 조금만 넣어서 적당히 간을 한다.",
            "일단 김치랑 먹어보고, 정 싱거우면 소금을 아주 조금 넣는다.",
            "아무것도 넣지 않는다. 슴슴한 고기 육수 본연의 맛을 즐긴다."
        ]
    },
    {
        "id": 10,
        "axis": "flavor_intensity",
        "section": "맛의 강도",
        "question": "\"정말 맛있는 디저트\"의 기준은?",
        "options": [
            "입안 가득 퍼지는 꾸덕하고 진한 초콜릿/카라멜의 강렬한 단맛.",
            "달달한 생크림 케이크나 마카롱.",
            "적당히 달달한 정도.",
            "\"많이 달지 않아서 맛있네\"라는 말이 나오는 담백한 빵.",
            "쑥, 흑임자, 혹은 과일 본연의 단맛만 살짝 나는 것."
        ]
    },
    {
        "id": 11,
        "axis": "flavor_intensity",
        "section": "맛의 강도",
        "question": "음식을 찍어 먹는 '소스'에 대한 당신의 철학은?",
        "options": [
            "소스 맛으로 먹는다. 듬뿍 찍어서 소스가 흐를 정도로 먹어야 제맛.",
            "넉넉하게 찍어 먹는 편이다.",
            "적당량만 찍는다.",
            "살짝만 찍거나, 굳이 안 찍어도 된다.",
            "소스는 재료 맛을 해친다. 소금만 살짝 찍거나 그냥 먹는다."
        ]
    },
    {
        "id": 12,
        "axis": "flavor_intensity",
        "section": "맛의 강도",
        "question": "친구가 \"이 집 평양냉면(또는 건강식) 진짜 잘해\"라고 데려갔다. 맛을 본 당신의 속마음은?",
        "options": [
            "\"행주 빤 물 맛 아니야? 무슨 맛으로 먹는지 모르겠다.\"",
            "\"음... 좀 심심하네. 식초랑 겨자 좀 많이 넣어야겠다.\"",
            "\"깔끔하네. 가끔 먹을 만하다.\"",
            "\"오, 담백하고 개운한데? 계속 들어가는 맛이다.\"",
            "\"와, 육향 미쳤다. 슴슴한데 깊은 맛이 나네. 인생 맛집 등극!\""
        ]
    },
    
    # 축 2: 분위기 vs 효율 추가 4문 (Q13-Q16)
    {
        "id": 13,
        "axis": "dining_environment",
        "section": "분위기 vs 효율",
        "question": "SNS에서 핫한 맛집을 발견했다. 웨이팅이 2시간이라는데?",
        "options": [
            "\"기다림도 맛의 일부!\" 오픈런을 해서라도 무조건 먹고 사진 찍는다.",
            "1시간 정도는 기다릴 수 있다. 폰 보면서 기다리면 된다.",
            "상황 봐서 줄이 좀 짧으면 기다리고, 아니면 다른 데 간다.",
            "\"굳이?\" 맛이 비슷하다면 옆에 있는 줄 없는 식당 간다.",
            "\"밥 먹는데 무슨 줄을 서?\" 웨이팅 있는 식당은 내 사전에 없다."
        ]
    },
    {
        "id": 14,
        "axis": "dining_environment",
        "section": "분위기 vs 효율",
        "question": "식당을 고를 때 '인테리어'가 차지하는 비중은?",
        "options": [
            "1순위. 맛이 평범해도 인테리어가 예쁘면 간다. 사진이 잘 나와야 함.",
            "중요한 편이다. 분위기가 좋으면 맛도 더 좋게 느껴진다.",
            "깨끗하기만 하면 된다.",
            "인테리어보다는 의자가 편하고 테이블이 넓은 게 좋다.",
            "전혀 상관없다. 맛만 있다면 허름한 창고 바닥에서 먹어도 된다."
        ]
    },
    {
        "id": 15,
        "axis": "dining_environment",
        "section": "분위기 vs 효율",
        "question": "음식 사진을 찍는 당신의 스타일은?",
        "options": [
            "조명, 구도 세팅하고 일행들 못 먹게 막은 뒤 10장 이상 찍는다.",
            "음식이 나오면 예쁘게 2~3장 정도 찍어서 인스타 스토리에 올린다.",
            "가끔 생각나면 한 장 정도 찍는다.",
            "\"아 맞다, 사진.\" 먹다가 생각나서 대충 찍거나 안 찍는다.",
            "음식 나오자마자 숟가락부터 나간다. 갤러리에 음식 사진이 없다."
        ]
    },
    {
        "id": 16,
        "axis": "dining_environment",
        "section": "분위기 vs 효율",
        "question": "식당 위치를 선정할 때 허용 가능한 이동 거리는?",
        "options": [
            "맛과 분위기만 좋다면 왕복 3시간 거리의 교외나 지방도 간다.",
            "차로 30분~1시간 정도는 기분 전환 겸 갈 수 있다.",
            "대중교통으로 가기 편한 곳이 좋다.",
            "집이나 회사 근처, 도보 10분 컷이 편하다.",
            "현관 앞. 배달이 되거나 슬리퍼 신고 나갈 수 있는 곳이어야 한다."
        ]
    },
    
    # 축 3: 비용 기준 추가 4문 (Q17-Q20)
    {
        "id": 17,
        "axis": "price_sensitivity",
        "section": "비용 기준",
        "question": "특별한 날이 아닌 평범한 저녁, 한 끼에 10만 원을 쓴다면?",
        "options": [
            "\"나를 위한 투자!\" 전혀 아깝지 않고 오히려 기분 좋다.",
            "퀄리티만 좋다면 충분히 쓸 수 있다.",
            "메뉴에 따라 다르지만 조금 고민된다.",
            "\"너무 과소비인데...\" 손이 떨려서 결제하기 힘들다.",
            "\"미쳤어?\" 그 돈이면 국밥이 10그릇이다. 절대 안 쓴다."
        ]
    },
    {
        "id": 18,
        "axis": "price_sensitivity",
        "section": "비용 기준",
        "question": "'오마카세'나 '파인다이닝'에 대한 생각은?",
        "options": [
            "취미 생활이다. 셰프의 설명과 대접받는 느낌을 사랑한다.",
            "기념일이나 기분 내고 싶을 때는 꼭 가고 싶다.",
            "누가 사주면 가지만, 내 돈 내고는 자주는 안 간다.",
            "\"거품이 좀 심한 것 같은데...\" 가격 대비 만족도가 낮다.",
            "\"찔끔찔끔 줘서 감질나.\" 차라리 고기 뷔페 가서 배 터지게 먹는다."
        ]
    },
    {
        "id": 19,
        "axis": "price_sensitivity",
        "section": "비용 기준",
        "question": "메뉴판을 볼 때 시선이 먼저 가는 곳은?",
        "options": [
            "메뉴 설명과 구성. 가격은 나중에 보거나 신경 안 쓴다.",
            "먹고 싶은 메뉴를 먼저 고르고 가격을 확인한다.",
            "가격과 메뉴를 번갈아 보며 비교한다.",
            "가격을 먼저 훑어보고 예산 내에서 메뉴를 고른다.",
            "'점심 특선', '할인 메뉴', '세트 구성'부터 찾는다."
        ]
    },
    {
        "id": 20,
        "axis": "price_sensitivity",
        "section": "비용 기준",
        "question": "\"무한리필\" 식당에 대한 이미지는?",
        "options": [
            "\"질 낮은 재료를 쓰겠지.\" 맛없을 것 같아서 피한다.",
            "퀄리티가 보장된 곳(호텔 뷔페 등) 아니면 잘 안 간다.",
            "회식이나 모임 때는 간다.",
            "\"가성비 좋잖아?\" 배고플 때 가면 이득이다.",
            "\"여기가 천국!\" 맛보다 양이다. 배 터지게 먹을 수 있어서 사랑한다."
        ]
    },
    
    # 축 4: 식사 인원 추가 4문 (Q21-Q24)
    {
        "id": 21,
        "axis": "dining_company",
        "section": "식사 인원",
        "question": "4인 테이블이 꽉 차 있는 맛집에 혼자 들어가야 한다면?",
        "options": [
            "절대 못 들어간다. 차라리 굶거나 편의점 간다.",
            "들어갈 순 있지만, 남들 시선이 신경 쓰여서 밥이 코로 들어간다.",
            "스마트폰 보면서 먹으면 괜찮다.",
            "전혀 상관없다. 맛있는 걸 먹으러 왔을 뿐이다.",
            "오히려 좋다. 넓은 테이블 다 쓰고 여유롭게 즐긴다."
        ]
    },
    {
        "id": 22,
        "axis": "dining_company",
        "section": "식사 인원",
        "question": "가장 이상적인 식사 시간의 대화량은?",
        "options": [
            "끊임없이 떠들어야 한다. 대화 없는 식사는 체할 것 같다.",
            "적당히 이야기 나누며 즐겁게 먹는 게 좋다.",
            "먹을 땐 먹고, 다 먹고 나서 이야기하는 게 좋다.",
            "조용한 게 좋다. 대화하느라 먹는 흐름 끊기는 게 싫다.",
            "이어폰 꽂고 영상 보거나 나만의 생각에 잠겨 먹는 게 최고다."
        ]
    },
    {
        "id": 23,
        "axis": "dining_company",
        "section": "식사 인원",
        "question": "정말 맛있는 인생 맛집을 발견했을 때 드는 생각은?",
        "options": [
            "당장 친구들이나 가족에게 연락해서 \"여기 대박이야, 조만간 같이 오자!\"",
            "친한 친구 한 명 정도 데려오고 싶다.",
            "다음에 누구랑 올지 고민해 본다.",
            "\"아, 맛있다.\" 나중에 또 혼자 와서 먹어야겠다고 생각한다.",
            "나만 알고 싶다. 사람 많아지면 곤란하니 아무에게도 안 알린다."
        ]
    },
    {
        "id": 24,
        "axis": "dining_company",
        "section": "식사 인원",
        "question": "고기 굽는 식당이나 뷔페에 혼자 가는 난이도는?",
        "options": [
            "Lv. 99 (불가능). 상상만 해도 식은땀이 난다.",
            "어렵다. 정말 먹고 싶으면 눈치 보며 도전해볼 수도 있다.",
            "남들이 보든 말든 상관없는데, 혼자 굽기 귀찮아서 잘 안 간다.",
            "가끔 간다. 혼자 먹으면 고기 안 뺏겨서 좋다.",
            "즐긴다. 혼자서 고기 3인분 굽고 볶음밥까지 클리어하는 게 취미다."
        ]
    }
]


# 심화 테스트 = 간단 + 추가
DEEP_TEST_QUESTIONS = QUICK_TEST_QUESTIONS + DEEP_TEST_ADDITIONAL_QUESTIONS


def calculate_mbti_with_probabilities(answers: list, test_type: str = "quick") -> dict:
    """
    답변 리스트로부터 직접 MBTI 타입과 확률 계산 (단순 선형 방식)
    
    Args:
        answers: [1, 3, 5, 2, ...] 형태의 답변 리스트 (각 값은 1-5)
        test_type: 'quick' or 'deep'
    
    Returns:
        dict: {
            "type": "SAPA",
            "axis_scores": {
                "flavor_intensity": {"S": 73, "M": 27},
                "dining_environment": {"A": 82, "O": 18},
                "price_sensitivity": {"P": 65, "C": 35},
                "dining_company": {"A": 88, "O": 12}
            }
        }
    """
    questions_per_axis = 2 if test_type == "quick" else 6
    
    # 각 축별로 답변 추출
    axis1_answers = answers[0:questions_per_axis]  # 맛의 강도
    axis2_answers = answers[questions_per_axis:questions_per_axis*2]  # 분위기 vs 효율
    axis3_answers = answers[questions_per_axis*2:questions_per_axis*3]  # 비용 기준
    axis4_answers = answers[questions_per_axis*3:questions_per_axis*4]  # 식사 인원
    
    # 단순 선형 매핑: 1→100% A, 2→75% A, 3→50%, 4→25% A, 5→0% A
    def answer_to_percentage(answer):
        return (5 - answer) * 25  # A축 비율
    
    # 각 축별 평균 계산
    s_pct = int(sum(answer_to_percentage(a) for a in axis1_answers) / len(axis1_answers))
    m_pct = 100 - s_pct
    
    a_env_pct = int(sum(answer_to_percentage(a) for a in axis2_answers) / len(axis2_answers))
    o_env_pct = 100 - a_env_pct
    
    p_pct = int(sum(answer_to_percentage(a) for a in axis3_answers) / len(axis3_answers))
    c_pct = 100 - p_pct
    
    a_company_pct = int(sum(answer_to_percentage(a) for a in axis4_answers) / len(axis4_answers))
    o_company_pct = 100 - a_company_pct
    
    # MBTI 타입 결정
    axis1 = "S" if s_pct > 50 else "M"
    axis2 = "A" if a_env_pct > 50 else "O"
    axis3 = "P" if p_pct > 50 else "C"
    axis4 = "A" if a_company_pct > 50 else "O"
    
    mbti_type = f"{axis1}{axis2}{axis3}{axis4}"
    
    return {
        "type": mbti_type,
        "axis_scores": {
            "flavor_intensity": {"S": s_pct, "M": m_pct},
            "dining_environment": {"A": a_env_pct, "O": o_env_pct},
            "price_sensitivity": {"P": p_pct, "C": c_pct},
            "dining_company": {"A": a_company_pct, "O": o_company_pct}
        }
    }


def mbti_axes_to_absa_synthetic(axis_scores: dict, mbti_type: str) -> dict:
    """
    MBTI 축 확률로부터 합성 ABSA 피처 생성 (Cold Start 해결)
    
    Args:
        axis_scores: {
            "flavor_intensity": {"S": 73, "M": 27},
            "dining_environment": {"A": 82, "O": 18},
            "price_sensitivity": {"P": 65, "C": 35},
            "dining_company": {"A": 88, "O": 12}
        }
        mbti_type: "SAPA"
    
    Returns:
        dict: 51개 ABSA 키에 대한 합성 값
    """
    absa_synthetic = {}
    
    # 축 1: 맛 성향 (S vs M)
    s_ratio = axis_scores["flavor_intensity"]["S"] / 100.0
    m_ratio = axis_scores["flavor_intensity"]["M"] / 100.0
    
    # Strong 계열 ABSA
    absa_synthetic["매운맛_긍정"] = s_ratio * 0.9
    absa_synthetic["짠맛_긍정"] = s_ratio * 0.8
    absa_synthetic["느끼함_긍정"] = s_ratio * 0.6
    
    # Mild 계열 ABSA
    absa_synthetic["담백함_긍정"] = m_ratio * 0.9
    absa_synthetic["단맛_긍정"] = m_ratio * 0.7
    absa_synthetic["고소함_긍정"] = m_ratio * 0.6
    
    # 축 2: 분위기 vs 효율 (A vs O)
    a_env_ratio = axis_scores["dining_environment"]["A"] / 100.0
    o_env_ratio = axis_scores["dining_environment"]["O"] / 100.0
    
    # Ambiance 계열
    absa_synthetic["분위기_긍정"] = a_env_ratio * 0.95
    absa_synthetic["쾌적함/청결도_긍정"] = a_env_ratio * 0.8
    absa_synthetic["공간_긍정"] = a_env_ratio * 0.7
    absa_synthetic["대기_긍정"] = a_env_ratio * 0.4
    
    # Optimized 계열
    absa_synthetic["주차_긍정"] = o_env_ratio * 0.85
    absa_synthetic["대기_부정"] = o_env_ratio * 0.9
    absa_synthetic["대기_중립"] = o_env_ratio * 0.3
    
    # 축 3: 가격 (P vs C)
    p_ratio = axis_scores["price_sensitivity"]["P"] / 100.0
    c_ratio = axis_scores["price_sensitivity"]["C"] / 100.0
    
    # Premium 계열
    absa_synthetic["품질/신선도_긍정"] = p_ratio * 0.95
    absa_synthetic["맛_긍정"] = p_ratio * 0.9
    
    # Cost-effective 계열
    absa_synthetic["가격_긍정"] = c_ratio * 0.9
    absa_synthetic["양_긍정"] = c_ratio * 0.85
    
    # 축 4: 식사 스타일 (A vs O)
    a_company_ratio = axis_scores["dining_company"]["A"] / 100.0
    o_company_ratio = axis_scores["dining_company"]["O"] / 100.0
    
    # All Together 계열
    absa_synthetic["서비스_긍정"] = a_company_ratio * 0.8
    
    # 분위기는 이미 축2에서 계산되었으므로 max로 업데이트
    if "분위기_긍정" in absa_synthetic:
        absa_synthetic["분위기_긍정"] = max(
            absa_synthetic["분위기_긍정"],
            a_company_ratio * 0.7
        )
    
    # On My Own 계열
    absa_synthetic["소음_부정"] = o_company_ratio * 0.85
    absa_synthetic["서비스_중립"] = o_company_ratio * 0.6
    
    # 나머지 ABSA 키들은 0.0으로 초기화 (51개 전체)
    all_absa_keys = [
        "매운맛_긍정", "매운맛_부정", "매운맛_중립",
        "짠맛_긍정", "짠맛_부정", "짠맛_중립",
        "담백함_긍정", "담백함_부정", "담백함_중립",
        "단맛_긍정", "단맛_부정", "단맛_중립",
        "고소함_긍정", "고소함_부정", "고소함_중립",
        "느끼함_긍정", "느끼함_부정", "느끼함_중립",
        "양_긍정", "양_부정", "양_중립",
        "맛_긍정", "맛_부정", "맛_중립",
        "품질/신선도_긍정", "품질/신선도_부정", "품질/신선도_중립",
        "가격_긍정", "가격_부정", "가격_중립",
        "분위기_긍정", "분위기_부정", "분위기_중립",
        "쾌적함/청결도_긍정", "쾌적함/청결도_부정", "쾌적함/청결도_중립",
        "소음_긍정", "소음_부정", "소음_중립",
        "공간_긍정", "공간_부정", "공간_중립",
        "주차_긍정", "주차_부정", "주차_중립",
        "서비스_긍정", "서비스_부정", "서비스_중립",
        "대기_긍정", "대기_부정", "대기_중립"
    ]
    
    for key in all_absa_keys:
        if key not in absa_synthetic:
            absa_synthetic[key] = 0.0
    
    return absa_synthetic


def calculate_mbti_with_probabilities_and_absa(answers: list, test_type: str) -> dict:
    """
    MBTI 타입, 축 확률, 합성 ABSA를 모두 반환
    
    Args:
        answers: [1, 3, 5, 2, ...] 형태의 답변 리스트
        test_type: 'quick' or 'deep'
    
    Returns:
        dict: {
            "mbti_type": "SAPA",
            "axis_scores": {...},
            "absa_features": {...}  # 합성된 51개 ABSA
        }
    """
    # 1. MBTI 계산
    mbti_result = calculate_mbti_with_probabilities(answers, test_type)
    
    # 2. ABSA 합성
    synthetic_absa = mbti_axes_to_absa_synthetic(
        mbti_result["axis_scores"],
        mbti_result["type"]
    )
    
    return {
        "mbti_type": mbti_result["type"],
        "axis_scores": mbti_result["axis_scores"],
        "absa_features": synthetic_absa
    }


def calculate_mbti_type(answers: list, test_type: str = "quick") -> str:
    """
    편의 함수: MBTI 타입만 반환
    """
    result = calculate_mbti_with_probabilities(answers, test_type)
    return result["type"]


# MBTI 타입별 설명 (16개)
MBTI_TYPE_DESCRIPTIONS = {
    "SAPA": {
        "emoji": "🔥",
        "name": "도파민 추구 미식회장",
        "catchphrase": "인생 뭐 있어? 오늘 밤은 화려하게!",
        "description": "맵고 짜고 비싼 음식을 힙한 곳에서 친구들과 즐깁니다. 스트레스는 자극으로 풉니다.",
        "recommendations": ["파인다이닝 레스토랑", "프리미엄 바", "고급 한정식"],
        "recommend": [
            "마라 훠궈 무한리필: 가장 매운 단계 필수",
            "프리미엄 다이닝 펍: 조명 화려하고 음악 빵빵한 곳",
            "킹크랩/랍스터: 비주얼 압도적인 메뉴"
        ],
        "avoid": [
            "슴슴한 사찰음식 전문점 (무맛이라 싫음)",
            "조용히 책 읽어야 하는 북카페",
            "플레이팅 대충 나오는 기사식당"
        ]
    },
    "SAPO": {
        "emoji": "🍷",
        "name": "나를 위한 마라카세",
        "catchphrase": "내 취향은 소중하니까.",
        "description": "혼자서도 최고급으로 즐깁니다. 자극적인 맛을 우아하고 프라이빗하게 즐기는 고독한 미식가입니다.",
        "recommendations": ["조용한 고급 레스토랑", "프리미엄 와인바", "품격 있는 일식당"],
        "recommend": [
            "1인 셰프 바(Bar): 독한 위스키와 안주",
            "프리미엄 혼고기: 최고급 특수부위",
            "매운맛 오마카세: 퓨전 다이닝"
        ],
        "avoid": [
            "시끄러운 회식 장소 (기 빨림)",
            "가성비 무한리필집 (질 떨어져서 싫음)",
            "합석해야 하는 시장 국밥집"
        ]
    },
    "SACA": {
        "emoji": "📸",
        "name": "가성비 핫플 사냥꾼",
        "catchphrase": "야, 여기 인스타에서 난리 났대!",
        "description": "힙하고 맛도 있는데 가격까지 착한 곳을 찾아냅니다. 웨이팅도 친구와 함께라면 즐겁습니다.",
        "recommendations": ["가성비 좋은 맛집", "분위기 좋은 술집", "동네 인기 맛집"],
        "recommend": [
            "을지로 노포: 감성 터지는 야장 삼겹살",
            "대학가 퓨전 포차: 안주 킬러들의 성지",
            "줄 서는 떡볶이 맛집: 맵단짠의 정석"
        ],
        "avoid": [
            "비싸기만 하고 맛없는 호텔 뷔페",
            "인테리어 구린 동네 백반집",
            "정적 흐르는 격식 있는 식당"
        ]
    },
    "SACO": {
        "emoji": "🤳",
        "name": "프로 혼밥 인스타그래머",
        "catchphrase": "혼자 먹어도 느낌 있게.",
        "description": "분위기 좋은 창가 자리에서 자극적인 맛을 즐기며 SNS에 올릴 사진을 찍습니다.",
        "recommendations": ["바 테이블 라멘집", "감성 타코 가게", "수제버거 맛집"],
        "recommend": [
            "바 테이블 라멘집: 매운 돈코츠 라멘",
            "감성 타코 가게: 혼자서 타코 3개 순삭",
            "수제버거 맛집: 육즙 가득 패티"
        ],
        "avoid": [
            "4인 테이블만 있는 패밀리 레스토랑",
            "조명 너무 밝은 김밥천국",
            "플레이팅 없는 기사식당"
        ]
    },
    "SOPA": {
        "emoji": "⚡",
        "name": "속전속결 맛집 파괴자",
        "catchphrase": "맛있는 건 먹고 싶은데, 기다리는 건 질색이야.",
        "description": "돈을 더 내더라도 예약 확정이 되고, 서비스가 빠르며 맛도 확실한 곳을 선호합니다.",
        "recommendations": ["구워주는 고깃집", "중식 코스 요리", "호텔 뷔페 룸"],
        "recommend": [
            "구워주는 고깃집: 손 하나 까딱 안 해도 됨",
            "중식 코스 요리: 예약 필수, 빠른 서빙",
            "호텔 뷔페 룸: 동선 짧고 음식 다양"
        ],
        "avoid": [
            "웨이팅 2시간 필수인 인스타 핫플",
            "직접 재료 손질해서 먹는 샤브샤브",
            "주차장 없는 골목 식당"
        ]
    },
    "SOPO": {
        "emoji": "🛵",
        "name": "1인분 10만원 배달러",
        "catchphrase": "집이 최고야, 맛은 포기 못 해.",
        "description": "가장 편한 옷차림으로 집에서 최고급 요리를 즐깁니다. 배달비 따위는 신경 쓰지 않습니다.",
        "recommendations": ["특상 우나기동", "호텔 투고 박스", "프리미엄 모듬회"],
        "recommend": [
            "특상 우나기동(장어덮밥): 배달로 즐기는 보양식",
            "호텔 투고(To-go) 박스: 퀄리티 보장",
            "프리미엄 모듬회: 두툼하게 썬 회"
        ],
        "avoid": [
            "브레이크 타임 걸리는 식당",
            "직접 가서 줄 서야 먹을 수 있는 곳",
            "조리 시간이 오래 걸리는 슬로우 푸드"
        ]
    },
    "SOCA": {
        "emoji": "🍖",
        "name": "무한리필 원정대장",
        "catchphrase": "양 많고, 맛있고, 싸고!",
        "description": "전투적으로 먹습니다. 격식 따위는 필요 없고 배부르고 자극적인 게 최고입니다.",
        "recommendations": ["마라탕", "고기 뷔페", "기사식당"],
        "recommend": [
            "마라탕 & 꿔바로우: 재료 듬뿍 담아도 저렴",
            "고기 뷔페: 눈치 안 보고 고기 흡입",
            "기사식당 제육볶음: 맛과 스피드 보장"
        ],
        "avoid": [
            "양 쥐꼬리만 한 파인다이닝",
            "음식 나오는 데 30분 걸리는 곳",
            "조용히 먹어야 하는 분위기"
        ]
    },
    "SOCO": {
        "emoji": "🏪",
        "name": "편의점 만렙 귀차니스트",
        "catchphrase": "귀찮지만 맛없는 건 싫어.",
        "description": "식사는 생존입니다. 빠르고 싸게 해결하되, 맵고 짠 자극은 포기할 수 없습니다.",
        "recommendations": ["편의점", "패스트푸드", "컵밥"],
        "recommend": [
            "불닭볶음면 + 삼각김밥: 편의점 꿀조합",
            "패스트푸드 신메뉴: 키오스크로 1분 컷",
            "컵밥/도시락: 설거지 필요 없음"
        ],
        "avoid": [
            "주문 방식 복잡한 서브웨이 (커스텀 귀찮음)",
            "코스 요리 (너무 길어)",
            "직원이 말 거는 식당"
        ]
    },
    "MAPA": {
        "emoji": "🥗",
        "name": "청담동 브런치 모임장",
        "catchphrase": "건강하고 예쁘게, 우아하게.",
        "description": "자극적이지 않은 건강한 맛과 고급스러운 분위기를 선호합니다. 식사는 대화의 연장선입니다.",
        "recommendations": ["고급 레스토랑", "브런치 카페", "프리미엄 디저트 카페"],
        "recommend": [
            "호텔 애프터눈 티 세트: 사진 필수, 맛도 깔끔",
            "평양냉면 맛집: 슴슴한 맛의 미학",
            "비건 파인다이닝: 속이 편한 코스 요리"
        ],
        "avoid": [
            "맵고 짜고 위생 안 좋은 노포",
            "음악 소리 너무 큰 펍",
            "빨간 국물 튀는 전골집"
        ]
    },
    "MAPO": {
        "emoji": "🧘",
        "name": "고독한 미슐랭 탐험가",
        "catchphrase": "재료 본연의 맛을 느껴봐.",
        "description": "조용한 곳에서 셰프의 정성이 담긴 요리를 음미합니다. 시끄러운 건 질색입니다.",
        "recommendations": ["스시 오마카세", "프리미엄 솥밥", "티 오마카세"],
        "recommend": [
            "스시 오마카세: 흰 살 생선의 풍미",
            "프리미엄 솥밥: 갓 지은 밥맛",
            "티(Tea) 오마카세: 차와 다과"
        ],
        "avoid": [
            "시장통 국밥집 (너무 소란스러움)",
            "간이 센 프랜차이즈 식당",
            "합석 권유하는 식당"
        ]
    },
    "MACA": {
        "emoji": "🧸",
        "name": "연남동 감성 지킴이",
        "catchphrase": "아기자기하고 귀여운 게 좋아.",
        "description": "예쁘고 감성 넘치는 공간에서 부담 없는 가격의 브런치나 디저트를 즐깁니다.",
        "recommendations": ["일본 가정식", "주택 개조 카페", "과일 산도"],
        "recommend": [
            "일본 가정식: 정갈한 한 상 차림",
            "주택 개조 카페: 포토존 많은 곳",
            "과일 산도 & 푸딩: 눈으로 먼저 먹는 맛"
        ],
        "avoid": [
            "투박한 아재 입맛 식당",
            "욕쟁이 할머니 식당",
            "조명 어두운 술집"
        ]
    },
    "MACO": {
        "emoji": "🌿",
        "name": "숨은 골목 맛집 발굴러",
        "catchphrase": "소소하지만 확실한 행복.",
        "description": "사람 많은 핫플보다는 동네의 조용하고 따뜻한 가게를 선호합니다.",
        "recommendations": ["동네 베이커리", "조용한 찻집", "수제 요거트"],
        "recommend": [
            "동네 작은 베이커리: 갓 구운 소금빵",
            "조용한 찻집: 창밖 보며 멍때리기",
            "수제 요거트 가게: 건강한 한 끼"
        ],
        "avoid": [
            "단체 손님 많은 고깃집",
            "웨이팅 있는 인스타 맛집",
            "향이 강한 마라탕집"
        ]
    },
    "MOPA": {
        "emoji": "🧭",
        "name": "실패 없는 맛집 네비게이션",
        "catchphrase": "깔끔하고, 빠르고, 맛있게.",
        "description": "비즈니스 미팅이나 가족 모임에 적합한, 호불호 없이 누구나 만족할 고급 식당을 찾습니다.",
        "recommendations": ["룸 있는 일식", "모던 한식", "샤브샤브 뷔페"],
        "recommend": [
            "룸 있는 일식 코스: 프라이빗하고 정갈함",
            "모던 한식 반상: 1인 1상으로 깔끔하게",
            "샤브샤브 뷔페: 취향껏 골라 먹기"
        ],
        "avoid": [
            "호불호 갈리는 향신료(고수 등) 전문점",
            "음식 늦게 나오는 곳",
            "주차 불편한 곳"
        ]
    },
    "MOPO": {
        "emoji": "💼",
        "name": "성공한 자의 간편식",
        "catchphrase": "내 몸은 자산이다.",
        "description": "시간은 금입니다. 빠르고 간편하게 먹지만, 영양 성분과 퀄리티는 절대 타협하지 않습니다.",
        "recommendations": ["프리미엄 샐러드", "백화점 도시락", "고급 단백질 쉐이크"],
        "recommend": [
            "프리미엄 샐러드 구독: 탄단지 완벽 비율",
            "백화점 식품관 도시락: 검증된 맛",
            "고급 단백질 쉐이크: 바쁠 땐 마시는 걸로"
        ],
        "avoid": [
            "줄 서서 사 먹는 길거리 음식",
            "직접 요리해야 하는 밀키트 (귀찮음)",
            "기름진 패스트푸드"
        ]
    },
    "MOCA": {
        "emoji": "🍚",
        "name": "기사식당 섭외 담당",
        "catchphrase": "밥은 역시 집밥 스타일이지.",
        "description": "편안한 분위기에서 부담 없이 배불리 먹을 수 있는 백반집을 사랑합니다.",
        "recommendations": ["기사식당", "콩나물 국밥", "칼국수"],
        "recommend": [
            "기사식당 백반: 불백, 생선구이",
            "24시간 콩나물 국밥: 빠르고 든든함",
            "칼국수 & 수제비: 후루룩 먹기 좋음"
        ],
        "avoid": [
            "양 적고 비싼 카페 브런치",
            "옷 차려입고 가야 하는 레스토랑",
            "향신료 강한 외국 음식"
        ]
    },
    "MOCO": {
        "emoji": "🥪",
        "name": "삼각김밥 소믈리에",
        "catchphrase": "배만 채우면 돼, 자극적인 건 싫어.",
        "description": "식사는 미션 클리어하듯이. 빠르고 간단하게, 속 편한 음식을 선호합니다.",
        "recommendations": ["계란 샌드위치", "맑은 우동", "미숫가루"],
        "recommend": [
            "계란 샌드위치: 부드럽고 간편함",
            "맑은 우동/국수: 자극 없는 국물",
            "미숫가루/두유: 마시는 식사"
        ],
        "avoid": [
            "매운 떡볶이 (속 쓰림)",
            "먹기 불편한 대형 수제버거",
            "뜨거운 뚝배기 음식 (식혀 먹기 귀찮음)"
        ]
    }
}
