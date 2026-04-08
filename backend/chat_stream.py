"""SSE 流式对话：用 OpenAI 兼容 SDK 读取 DeepSeek reasoning_content + 正文 content。"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncIterator

from backend import agent as ag

# DeepSeek 思考模式说明：https://api-docs.deepseek.com/zh-cn/guides/thinking_mode


def _deepseek_thinking_extra_body() -> dict[str, Any] | None:
    """
    文档：除使用 model=deepseek-reasoner 外，可对 deepseek-chat 等传入
    extra_body={"thinking": {"type": "enabled"}} 开启思考模式（OpenAI SDK）。
    """
    raw = os.getenv("DEEPSEEK_THINKING", "").strip().lower()
    if raw in ("1", "true", "yes", "on", "enabled"):
        return {"thinking": {"type": "enabled"}}
    return None


def _use_openai_sdk_stream() -> bool:
    """需在原始流里读 reasoning_content 时用 SDK 直连。"""
    v = os.getenv("RAG_STREAM_OPENAI_SDK", "auto").strip().lower()
    model = (os.getenv("OPENAI_MODEL_NAME") or "").lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    if _deepseek_thinking_extra_body() is not None:
        return True
    return (
        "reasoner" in model
        or "deepseek-r1" in model
        or model.startswith("o1")
        or model.startswith("o3")
    )


def _reasoning_model(model: str) -> bool:
    m = model.lower()
    return (
        "reasoner" in m
        or "deepseek-r1" in m
        or m.startswith("o1")
        or m.startswith("o3")
    )


def _skip_temperature_for_thinking(model: str) -> bool:
    """思考模式下 temperature 等不生效，本项目不再传入。"""
    return _reasoning_model(model) or _deepseek_thinking_extra_body() is not None


def _lc_messages_to_openai(messages: list[Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = m.type
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        c = m.content
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            parts: list[str] = []
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text") or "")
                elif isinstance(block, str):
                    parts.append(block)
                else:
                    parts.append(str(block))
            text = "".join(parts)
        else:
            text = str(c) if c is not None else ""
        out.append({"role": role, "content": text})
    return out


def _delta_reasoning_and_content(delta: Any) -> tuple[str, str]:
    """从 OpenAI 兼容 delta 取 (reasoning_delta, content_delta)。"""
    reasoning, content = "", ""
    if delta is None:
        return reasoning, content
    rc = getattr(delta, "reasoning_content", None)
    if isinstance(rc, str) and rc:
        reasoning = rc
    raw_c = getattr(delta, "content", None)
    if isinstance(raw_c, str) and raw_c:
        content = raw_c
    if not reasoning and hasattr(delta, "model_extra") and delta.model_extra:
        ex = delta.model_extra
        for key in ("reasoning_content", "reasoning"):
            v = ex.get(key)
            if isinstance(v, str) and v:
                reasoning = v
                break
    return reasoning, content


async def _stream_via_openai_sdk(prompt_val: Any) -> AsyncIterator[dict]:
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model = os.getenv("OPENAI_MODEL_NAME", "deepseek-reasoner")
    temperature = float(os.getenv("RAG_TEMPERATURE", "0.1"))

    oai_msgs = _lc_messages_to_openai(prompt_val.to_messages())
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    kw: dict[str, Any] = {"model": model, "messages": oai_msgs, "stream": True}
    extra = _deepseek_thinking_extra_body()
    if extra is not None:
        kw["extra_body"] = extra
    if not _skip_temperature_for_thinking(model):
        kw["temperature"] = temperature
    stream = await client.chat.completions.create(**kw)

    header = False
    async for chunk in stream:
        chs = chunk.choices
        if not chs:
            continue
        delta = chs[0].delta
        r, c = _delta_reasoning_and_content(delta)
        if r:
            if not header:
                yield {"type": "reasoning", "delta": "【思考过程】\n"}
                header = True
            yield {"type": "reasoning", "delta": r}
        if c:
            yield {"type": "content", "delta": c}


async def _stream_via_langchain(llm: Any, prompt_val: Any) -> AsyncIterator[dict]:
    header = False
    async for chunk in llm.astream(prompt_val):
        ak = getattr(chunk, "additional_kwargs", None) or {}
        r = ""
        for key in ("reasoning_content", "reasoning", "thinking"):
            v = ak.get(key)
            if isinstance(v, str):
                r += v
        c = getattr(chunk, "content", None)
        c = c if isinstance(c, str) else ""
        if r:
            if not header:
                yield {"type": "reasoning", "delta": "【思考过程】\n"}
                header = True
            yield {"type": "reasoning", "delta": r}
        if c:
            yield {"type": "content", "delta": c}


async def iter_rag_stream_events(
    runner: Any, user_input: str, eval_capture: dict | None = None
) -> AsyncIterator[dict]:
    def _job() -> tuple[list, str]:
        docs = ag._gather_docs(user_input, runner._retriever, runner._vectorstore)
        return docs, ag._format_docs(docs)

    try:
        docs, ctx = await asyncio.to_thread(_job)
        if eval_capture is not None:
            eval_capture["docs"] = docs
            eval_capture["t_after_retrieval"] = time.perf_counter()
    except Exception as e:
        yield {"type": "error", "message": f"检索失败: {e}"}
        yield {"type": "done"}
        return

    yield {"type": "status", "message": "检索完成，正在生成…"}

    prompt_val = await runner._prompt.ainvoke({"context": ctx, "input": user_input})

    try:
        if _use_openai_sdk_stream():
            async for evt in _stream_via_openai_sdk(prompt_val):
                yield evt
        else:
            async for evt in _stream_via_langchain(runner._llm, prompt_val):
                yield evt
    except Exception as e:
        yield {"type": "error", "message": f"生成失败: {e}"}
        yield {"type": "done"}
        return

    yield {"type": "done"}
