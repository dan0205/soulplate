"""
HuggingFace Hub에서 모델 파일을 다운로드하는 모듈

환경 변수:
    HF_TOKEN: HuggingFace Access Token (Private 저장소인 경우 필수)
    HF_REPO_ID: HuggingFace 저장소 ID (기본값: yidj/soulplate-models)
"""
# 체크 로그

import os
import logging
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


def get_hf_config():
    """HuggingFace 설정 가져오기"""
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_REPO_ID", "yidj/soulplate-models")
    return token, repo_id


def download_model_file(filename: str, cache_dir: Optional[str] = None) -> str:
    """
    HuggingFace Hub에서 단일 모델 파일 다운로드
    
    Args:
        filename: 다운로드할 파일명 (예: "models/deepfm_ranking.pth")
        cache_dir: 캐시 디렉토리 (기본값: /tmp/models)
    
    Returns:
        다운로드된 파일의 로컬 경로
    
    Raises:
        Exception: 다운로드 실패 시
    """
    token, repo_id = get_hf_config()
    
    if cache_dir is None:
        cache_dir = "/tmp/models"
    
    try:
        logger.info(f"[HF] 다운로드 시작: {filename} from {repo_id}")
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            cache_dir=cache_dir,
            local_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"[HF] 다운로드 완료: {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"[HF] 다운로드 실패: {filename} - {e}")
        raise


def download_absa_model(cache_dir: Optional[str] = None) -> str:
    """
    HuggingFace Hub에서 ABSA 모델 전체 다운로드
    
    Args:
        cache_dir: 캐시 디렉토리 (기본값: /tmp/models/absa)
    
    Returns:
        다운로드된 모델 디렉토리 경로
    
    Raises:
        Exception: 다운로드 실패 시
    """
    token, repo_id = get_hf_config()
    
    if cache_dir is None:
        cache_dir = "/tmp/models/absa"
    
    try:
        logger.info(f"[HF] ABSA 모델 다운로드 시작 from {repo_id}")
        
        # absa 폴더 전체 다운로드
        local_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns="models/absa/*",
            token=token,
            cache_dir="/tmp/hf_cache",
            local_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        
        # absa 폴더의 실제 경로 반환
        absa_path = Path(local_path) / "models" / "absa"
        if absa_path.exists():
            logger.info(f"[HF] ABSA 모델 다운로드 완료: {absa_path}")
            return str(absa_path)
        else:
            # 경로 구조가 다를 수 있음
            logger.info(f"[HF] ABSA 모델 다운로드 완료: {local_path}")
            return str(local_path)
        
    except Exception as e:
        logger.error(f"[HF] ABSA 모델 다운로드 실패: {e}")
        raise


def ensure_model_file(filename: str, local_fallback: Optional[str] = None) -> Optional[str]:
    """
    모델 파일이 로컬에 있는지 확인하고, 없으면 HuggingFace에서 다운로드
    
    Args:
        filename: 파일명 (예: "models/deepfm_ranking.pth")
        local_fallback: 로컬 폴백 경로 (먼저 확인)
    
    Returns:
        파일 경로 (성공) 또는 None (실패)
    """
    # 1. 로컬 폴백 경로 확인
    if local_fallback and os.path.exists(local_fallback):
        logger.info(f"[Local] 로컬 파일 사용: {local_fallback}")
        return local_fallback
    
    # 2. 캐시 디렉토리 확인
    cache_path = f"/tmp/models/{filename}"
    if os.path.exists(cache_path):
        logger.info(f"[Cache] 캐시 파일 사용: {cache_path}")
        return cache_path
    
    # 3. HuggingFace에서 다운로드 시도
    try:
        return download_model_file(filename)
    except Exception as e:
        logger.warning(f"[HF] 다운로드 실패, 모델 없이 진행: {e}")
        return None


def ensure_absa_model(local_fallback: Optional[str] = None) -> Optional[str]:
    """
    ABSA 모델이 로컬에 있는지 확인하고, 없으면 HuggingFace에서 다운로드
    
    Args:
        local_fallback: 로컬 폴백 경로 (먼저 확인)
    
    Returns:
        모델 디렉토리 경로 (성공) 또는 None (실패)
    """
    # 1. 로컬 폴백 경로 확인
    if local_fallback and os.path.exists(local_fallback):
        logger.info(f"[Local] 로컬 ABSA 모델 사용: {local_fallback}")
        return local_fallback
    
    # 2. 캐시 디렉토리 확인
    cache_path = "/tmp/models/absa"
    if os.path.exists(cache_path) and os.path.exists(os.path.join(cache_path, "config.json")):
        logger.info(f"[Cache] 캐시 ABSA 모델 사용: {cache_path}")
        return cache_path
    
    # 3. HuggingFace에서 다운로드 시도
    try:
        return download_absa_model()
    except Exception as e:
        logger.warning(f"[HF] ABSA 다운로드 실패, 모델 없이 진행: {e}")
        return None

