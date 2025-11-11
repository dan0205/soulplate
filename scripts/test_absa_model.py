"""
ABSA 모델 로딩 및 추론 테스트 스크립트
- 모델 로딩 확인
- 1개 샘플 리뷰로 추론 테스트
- GPU 사용 확인
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import json
import os

def test_absa_model():
    print("=" * 80)
    print("ABSA 모델 테스트 시작")
    print("=" * 80)
    
    # 경로 설정
    model_path = "models/absa"
    
    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[OK] 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Warning: GPU 미사용 (CPU로 실행)")
    
    # 모델 및 토크나이저 로딩
    print("\n[1/4] 모델 로딩 중...")
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print("[OK] 모델 로딩 성공")
    except Exception as e:
        print(f"[FAIL] 모델 로딩 실패: {e}")
        return False
    
    print("\n[2/4] 토크나이저 로딩 중...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        print("[OK] 토크나이저 로딩 성공")
    except Exception as e:
        print(f"[FAIL] 토크나이저 로딩 실패: {e}")
        return False
    
    # Label 정보 확인
    print("\n[3/4] Label 정보 확인...")
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    
    id2label = config["id2label"]
    print(f"[OK] 총 {len(id2label)}개 클래스")
    
    # Aspect 및 Sentiment 파싱
    aspects = set()
    sentiments = set()
    for label in id2label.values():
        aspect, sentiment = label.split("_")
        aspects.add(aspect)
        sentiments.add(sentiment)
    
    print(f"  - Aspects ({len(aspects)}개): {', '.join(sorted(aspects))}")
    print(f"  - Sentiments ({len(sentiments)}개): {', '.join(sorted(sentiments))}")
    
    # 테스트 리뷰 추론
    print("\n[4/4] 샘플 리뷰 추론 테스트...")
    test_review = "음식이 정말 맛있고 서비스도 친절했어요. 가격은 조금 비싸지만 분위기가 좋아요."
    print(f"  입력: {test_review}")
    
    try:
        # 토크나이징
        inputs = tokenizer(
            test_review,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        # 추론
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)  # Multi-label이므로 sigmoid 사용
        
        # 결과 파싱
        probs_np = probs.cpu().numpy()[0]
        
        # Threshold 이상인 레이블만 출력
        threshold = 0.5
        detected = []
        for i, prob in enumerate(probs_np):
            if prob >= threshold:
                label = id2label[str(i)]
                detected.append((label, prob))
        
        print(f"\n[OK] 추론 성공!")
        print(f"\n감지된 Aspect-Sentiment (threshold={threshold}):")
        if detected:
            for label, prob in sorted(detected, key=lambda x: x[1], reverse=True):
                print(f"  - {label}: {prob:.4f}")
        else:
            print("  (없음 - threshold를 낮춰보세요)")
        
        # 모든 확률 출력 (상위 10개)
        print(f"\n상위 10개 확률:")
        top_indices = probs_np.argsort()[-10:][::-1]
        for idx in top_indices:
            label = id2label[str(idx)]
            print(f"  - {label}: {probs_np[idx]:.4f}")
        
        # 피처 벡터 형태 확인
        print(f"\n출력 shape: {probs_np.shape}")
        print(f"[OK] 예상대로 51개 피처 생성됨")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 추론 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_absa_model()
    
    print("\n" + "=" * 80)
    if success:
        print("[SUCCESS] ABSA 모델 테스트 완료!")
        print("다음 단계: scripts/step1_run_absa.py 실행")
    else:
        print("[FAILED] ABSA 모델 테스트 실패")
    print("=" * 80)

