"""LLM 网关兼容：部分 OpenAI 兼容端不支持 response_format / structured output。"""
from __future__ import annotations

import os


def rag_llm_structured_output_enabled() -> bool:
    """为 true 时使用 LangChain with_structured_output；否则走纯文本 + JSON 解析。"""
    return os.getenv("RAG_LLM_STRUCTURED_OUTPUT", "").lower() in ("1", "true", "yes")
