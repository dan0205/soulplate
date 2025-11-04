"""
S3에 모델 파일 업로드 (Phase 6)
"""

import boto3
import os
from pathlib import Path

# S3 설정
S3_BUCKET = os.getenv('S3_BUCKET', 'two-tower-model-assets')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# 업로드할 파일들
MODEL_DIR = Path('models')
FILES_TO_UPLOAD = [
    'user_tower.pth',
    'item_tower.pth',
    'index.faiss',
    'idx_to_business_id.json'
]

def upload_to_s3():
    """S3에 모델 파일 업로드"""
    print("=" * 60)
    print("Uploading models to S3")
    print("=" * 60)
    
    # S3 클라이언트 생성
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    for filename in FILES_TO_UPLOAD:
        local_path = MODEL_DIR / filename
        s3_key = f"models/{filename}"
        
        if not local_path.exists():
            print(f"Warning: {local_path} not found, skipping...")
            continue
        
        print(f"Uploading {filename}...")
        try:
            s3_client.upload_file(
                str(local_path),
                S3_BUCKET,
                s3_key,
                ExtraArgs={'ContentType': 'application/octet-stream'}
            )
            print(f"  ✓ Uploaded {filename} to s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            print(f"  ✗ Failed to upload {filename}: {e}")
            raise
    
    print("\n" + "=" * 60)
    print("All models uploaded successfully!")
    print("=" * 60)

if __name__ == "__main__":
    upload_to_s3()

