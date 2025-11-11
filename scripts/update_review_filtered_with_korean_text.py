"""
review_filtered_20_3.csv에 한글 텍스트 추가
- review_100k_absa_with_text.csv의 text 컬럼을 review_id로 매칭하여 추가
- 새로운 파일로 저장: review_filtered_20_3_korean.csv
"""

import pandas as pd
import sys

def update_review_filtered():
    """review_filtered에 한글 텍스트 추가"""
    print("=" * 80)
    print("review_filtered_20_3.csv에 한글 텍스트 추가")
    print("=" * 80)
    
    # 1. 파일 로딩
    print("\n[1/3] 파일 로딩 중...")
    
    try:
        filtered_df = pd.read_csv("data/filtered/review_filtered_20_3.csv", encoding='utf-8-sig')
        print(f"  [OK] review_filtered_20_3.csv: {len(filtered_df):,}개 리뷰")
    except Exception as e:
        print(f"  [ERROR] review_filtered_20_3.csv 로딩 실패: {e}")
        sys.exit(1)
    
    try:
        korean_df = pd.read_csv("data/raw/review_100k_absa_with_text.csv", encoding='utf-8-sig')
        print(f"  [OK] review_100k_absa_with_text.csv: {len(korean_df):,}개 리뷰")
    except Exception as e:
        print(f"  [ERROR] review_100k_absa_with_text.csv 로딩 실패: {e}")
        sys.exit(1)
    
    # 2. review_id로 매칭하여 text 컬럼 추가
    print("\n[2/3] review_id로 매칭하여 한글 텍스트 추가 중...")
    
    # 한글 리뷰에서 review_id와 text만 추출
    korean_text_df = korean_df[['review_id', 'text']].copy()
    
    # merge로 결합 (left join)
    result_df = filtered_df.merge(
        korean_text_df, 
        on='review_id', 
        how='left'
    )
    
    # 매칭 결과 확인
    matched = result_df['text'].notna().sum()
    not_matched = result_df['text'].isna().sum()
    
    print(f"  [OK] 매칭 성공: {matched:,}개 ({matched/len(result_df)*100:.1f}%)")
    if not_matched > 0:
        print(f"  [WARNING] 매칭 실패: {not_matched:,}개 ({not_matched/len(result_df)*100:.1f}%)")
        # 매칭 실패한 리뷰는 빈 문자열로 채움
        result_df['text'] = result_df['text'].fillna('')
    
    # 3. 저장
    print("\n[3/3] 파일 저장 중...")
    
    output_path = "data/filtered/review_filtered_20_3_korean.csv"
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"  [OK] 저장 완료: {output_path}")
    print(f"  총 {len(result_df):,}개 리뷰")
    print(f"  컬럼: {list(result_df.columns)}")
    
    # 샘플 확인
    print("\n[샘플] 처음 3개 리뷰의 텍스트:")
    for idx in range(min(3, len(result_df))):
        text = result_df.iloc[idx]['text']
        text_preview = text[:100] + '...' if len(text) > 100 else text
        print(f"  {idx+1}. {text_preview}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] 완료!")
    print(f"\n새 파일: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    update_review_filtered()

