"""
Step 1: ABSA 추론 (Colab 버전)
Google Colab에서 실행하세요.

실행 방법:
1. Colab에서 새 노트북 생성
2. 이 파일의 코드를 셀에 복사
3. review_100k_translated.csv를 Colab에 업로드
4. 실행 후 review_absa_features.csv 다운로드
"""

# ============================================================================
# 셀 1: 환경 설정
# ============================================================================

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 필요한 라이브러리 설치 (이미 설치되어 있을 수 있음)
!pip install transformers torch pandas numpy tqdm -q

# ============================================================================
# 셀 2: 파일 업로드
# ============================================================================

from google.colab import files
import os

# 리뷰 데이터 업로드
print("review_100k_translated.csv 파일을 업로드하세요.")
print("(로컬에서 data/processed/review_100k_translated.csv)")
uploaded = files.upload()

# 업로드 확인
if 'review_100k_translated.csv' not in uploaded:
    print("[ERROR] review_100k_translated.csv 파일이 업로드되지 않았습니다.")
else:
    print(f"[OK] 파일 업로드 완료: {list(uploaded.keys())}")

# ============================================================================
# 셀 3: ABSA 추론 실행
# ============================================================================

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

def load_absa_model(model_path="/content/drive/MyDrive/study-absa-lmkor/final_model"):
    """ABSA 모델 및 토크나이저 로딩"""
    print("[1/4] ABSA 모델 로딩 중...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  디바이스: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # Label 정보 로딩
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    id2label = config["id2label"]
    
    print(f"  [OK] 모델 로딩 완료 (51개 클래스)")
    
    return model, tokenizer, id2label, device

def predict_absa_batch(model, tokenizer, texts, device, batch_size=64):
    """배치 단위로 ABSA 추론"""
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 토크나이징
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        # 추론
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)  # Multi-label classification
        
        all_probs.append(probs.cpu().numpy())
    
    return np.vstack(all_probs)

def run_absa_inference():
    """100k 리뷰에 ABSA 추론 실행"""
    print("=" * 80)
    print("ABSA 추론 시작 (100k 리뷰)")
    print("=" * 80)
    
    # 모델 로딩
    model, tokenizer, id2label, device = load_absa_model()
    
    # 리뷰 데이터 로딩
    print("\n[2/4] 리뷰 데이터 로딩 중...")
    review_path = "review_100k_translated.csv"
    
    if not os.path.exists(review_path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {review_path}")
        return False
    
    reviews_df = pd.read_csv(review_path)
    print(f"  [OK] {len(reviews_df):,}개 리뷰 로딩 완료")
    print(f"  컬럼: {list(reviews_df.columns)}")
    
    # 번역된 텍스트 사용
    if 'translated_text' in reviews_df.columns:
        texts = reviews_df['translated_text'].fillna('').astype(str).tolist()
    else:
        print("[WARNING] 'translated_text' 컬럼이 없습니다. 'text' 컬럼 사용")
        texts = reviews_df['text'].fillna('').astype(str).tolist()
    
    # ABSA 추론
    print(f"\n[3/4] ABSA 추론 실행 중... (배치 크기: 64)")
    if torch.cuda.is_available():
        print(f"  예상 시간: 약 10-20분 (GPU)")
    else:
        print(f"  예상 시간: 약 1-2시간 (CPU)")
    
    batch_size = 64 if torch.cuda.is_available() else 32
    chunk_size = 2000  # 진행 상황을 위해 2000개씩 처리
    
    all_features = []
    
    for chunk_start in tqdm(range(0, len(texts), chunk_size), desc="Processing"):
        chunk_end = min(chunk_start + chunk_size, len(texts))
        chunk_texts = texts[chunk_start:chunk_end]
        
        # 배치 추론
        chunk_probs = predict_absa_batch(model, tokenizer, chunk_texts, device, batch_size)
        all_features.append(chunk_probs)
    
    # 결합
    features_array = np.vstack(all_features)
    print(f"\n[OK] 추론 완료! Shape: {features_array.shape}")
    
    # DataFrame 생성
    print("\n[4/4] 결과 저장 중...")
    
    # 컬럼 이름 생성
    feature_columns = []
    for i in range(51):
        label = id2label[str(i)]
        aspect, sentiment = label.split("_")
        col_name = f"absa_{aspect}_{sentiment}"
        feature_columns.append(col_name)
    
    # 피처 DataFrame
    features_df = pd.DataFrame(features_array, columns=feature_columns)
    
    # 원본 리뷰 정보와 결합
    result_df = pd.concat([
        reviews_df[['review_id', 'user_id', 'business_id', 'stars']].reset_index(drop=True),
        features_df
    ], axis=1)
    
    # 저장
    output_path = "review_absa_features.csv"
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"  [OK] 저장 완료: {output_path}")
    print(f"  Shape: {result_df.shape}")
    print(f"  컬럼: review_id, user_id, business_id, stars + {len(feature_columns)}개 ABSA 피처")
    
    # 샘플 확인
    print(f"\n샘플 데이터 (처음 3개 리뷰의 상위 5개 피처):")
    for i in range(min(3, len(result_df))):
        print(f"\n리뷰 {i+1} (stars={result_df.iloc[i]['stars']}):")
        row = features_df.iloc[i]
        top_5_idx = row.argsort()[-5:][::-1]
        for idx in top_5_idx:
            col = feature_columns[idx]
            val = row.iloc[idx]
            print(f"  {col}: {val:.4f}")
    
    return True

# 실행
success = run_absa_inference()

print("\n" + "=" * 80)
if success:
    print("[SUCCESS] ABSA 추론 완료!")
    print("review_absa_features.csv 파일을 다운로드하세요.")
else:
    print("[FAILED] ABSA 추론 실패")
print("=" * 80)

# ============================================================================
# 셀 4: 결과 다운로드
# ============================================================================

from google.colab import files

# 결과 파일 다운로드
if os.path.exists("review_absa_features.csv"):
    print("결과 파일 다운로드 중...")
    files.download("review_absa_features.csv")
    print("[OK] 다운로드 완료!")
    print("\n다음 단계:")
    print("1. 다운로드한 'review_absa_features.csv'를 로컬의 'data/processed/' 폴더에 복사")
    print("2. scripts/step2_aggregate_features.py 실행")
else:
    print("[ERROR] 결과 파일이 생성되지 않았습니다.")

