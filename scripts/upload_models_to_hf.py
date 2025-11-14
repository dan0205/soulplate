"""
HuggingFace Hub에 모델 파일들을 업로드하는 스크립트

사용법:
    python scripts/upload_models_to_hf.py --token YOUR_HF_TOKEN --repo-id YOUR_USERNAME/soulplate-models
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login


def upload_models_to_hf(token: str, repo_id: str = None, private: bool = True):
    """
    모델 파일들을 HuggingFace Hub에 업로드
    
    Args:
        token: HuggingFace Access Token
        repo_id: 저장소 ID (예: 'username/soulplate-models'). None이면 자동으로 생성
        private: Private 저장소 여부 (기본값: True)
    """
    # 로그인
    print(f"🔐 HuggingFace에 로그인 중...")
    login(token=token)
    
    # API 클라이언트 생성
    api = HfApi()
    
    # repo_id가 없으면 자동으로 생성
    if repo_id is None:
        whoami = api.whoami(token=token)
        username = whoami['name']
        repo_id = f"{username}/soulplate-models"
        print(f"📝 저장소 ID 자동 생성: {repo_id}")
    
    # 저장소 생성 (이미 존재하면 무시됨)
    print(f"📦 저장소 생성 중: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            repo_type="model",
            token=token
        )
        print(f"✅ 저장소 생성 완료!")
    except Exception as e:
        print(f"⚠️ 저장소가 이미 존재하거나 생성 실패: {e}")
        # 계속 진행 (이미 존재할 수 있음)
    
    # 프로젝트 루트 디렉토리
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    # 업로드할 파일 목록
    files_to_upload = [
        ("deepfm_ranking.pth", "models/deepfm_ranking.pth"),
        ("multitower_ranking.pth", "models/multitower_ranking.pth"),
        ("tfidf_vectorizer.pkl", "models/tfidf_vectorizer.pkl"),
        ("scaler_params.json", "models/scaler_params.json"),
    ]
    
    # ABSA 모델 파일들
    absa_dir = models_dir / "absa"
    if absa_dir.exists():
        absa_files = [
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt"
        ]
        for filename in absa_files:
            files_to_upload.append((f"absa/{filename}", f"models/absa/{filename}"))
    
    # 파일 업로드
    print(f"\n📤 총 {len(files_to_upload)}개 파일 업로드 시작...\n")
    
    uploaded_count = 0
    failed_files = []
    
    for local_filename, remote_path in files_to_upload:
        local_path = models_dir / local_filename.replace("absa/", "absa/")
        
        if not local_path.exists():
            print(f"⚠️ 파일을 찾을 수 없음: {local_path}")
            failed_files.append(local_filename)
            continue
        
        try:
            file_size = local_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  업로드 중: {local_filename} ({file_size:.2f} MB)")
            
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                token=token,
            )
            
            uploaded_count += 1
            print(f"  ✅ 완료: {local_filename}")
            
        except Exception as e:
            print(f"  ❌ 실패: {local_filename} - {e}")
            failed_files.append(local_filename)
    
    # 결과 요약
    print(f"\n" + "="*60)
    print(f"✨ 업로드 완료!")
    print(f"  - 성공: {uploaded_count}/{len(files_to_upload)} 파일")
    if failed_files:
        print(f"  - 실패: {len(failed_files)} 파일")
        print(f"    {', '.join(failed_files)}")
    print(f"="*60)
    print(f"\n🌐 저장소 URL: https://huggingface.co/{repo_id}")
    print(f"\n다음 단계:")
    print(f"  1. Railway에 환경 변수 설정:")
    print(f"     - HF_TOKEN={token[:10]}...")
    print(f"     - HF_REPO_ID={repo_id}")
    print(f"  2. 코드 커밋 및 푸시")
    print(f"  3. Railway 자동 재배포 확인")
    

def main():
    parser = argparse.ArgumentParser(description="HuggingFace Hub에 모델 업로드")
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace Access Token"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=False,
        default=None,
        help="저장소 ID (예: username/soulplate-models). 미지정시 자동 생성"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Public 저장소로 생성 (기본값: Private)"
    )
    
    args = parser.parse_args()
    
    upload_models_to_hf(
        token=args.token,
        repo_id=args.repo_id,
        private=not args.public
    )


if __name__ == "__main__":
    main()

