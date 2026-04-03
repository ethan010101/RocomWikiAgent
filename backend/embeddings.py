"""
建库（build_kb）与在线检索（agent）必须使用同一向量化模型，否则 FAISS 维度/分布不一致。

长期方案默认：中文检索向量化模型 BGE-small-zh，配合 normalize_embeddings 做余弦语义检索。
"""
import os


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5").strip()
    device = os.getenv("EMBEDDING_DEVICE", "cpu").strip()
    normalize = os.getenv("EMBEDDING_NORMALIZE", "true").lower() in ("1", "true", "yes")

    model_kwargs = {"device": device}
    if os.getenv("EMBEDDING_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes"):
        model_kwargs["trust_remote_code"] = True

    kw: dict = {"model_name": model_name, "model_kwargs": model_kwargs}
    if normalize:
        kw["encode_kwargs"] = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(**kw)


def embedding_model_label() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5").strip()
