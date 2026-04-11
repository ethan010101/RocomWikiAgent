"""Load .env and set Hugging Face Hub endpoint before any huggingface_hub import."""
from __future__ import annotations

import os
from pathlib import Path


def init_hf_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    root = Path(__file__).resolve().parent.parent
    if load_dotenv:
        load_dotenv(root / ".env")
        load_dotenv()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    endpoint = os.getenv("HF_ENDPOINT", "").strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")

    if not os.getenv("HF_HUB_DOWNLOAD_TIMEOUT"):
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

    # 嵌入权重已在 Hub 缓存或 EMBEDDING_MODEL_NAME 为本地目录时设为 true，可避免每次启动/重载对镜像发 HEAD（易 10060）。
    if os.getenv("HF_EMBEDDING_OFFLINE", "").lower() in ("1", "true", "yes"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
