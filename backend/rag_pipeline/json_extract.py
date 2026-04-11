"""从 LLM 回复中稳健取出 JSON 对象（兼容 markdown 围栏、多段 JSON、reasoning 字段）。"""
from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```", re.MULTILINE)


def fenced_inner_texts(text: str) -> list[str]:
    if not text:
        return []
    return [m.group(1).strip() for m in _FENCE_RE.finditer(text) if m.group(1).strip()]


def ai_message_concat_text(msg: Any) -> str:
    """
    合并 AIMessage 中可能承载正文的字段，便于下游 JSON 解析。
    部分网关把「思考」放在 additional_kwargs，最终答案很短或为空。
    """
    parts: list[str] = []
    c = getattr(msg, "content", None)
    if isinstance(c, list):
        c = "".join(
            getattr(b, "text", str(b)) if not isinstance(b, str) else b for b in c
        )
    if isinstance(c, str) and c.strip():
        parts.append(c.strip())
    ak = getattr(msg, "additional_kwargs", None) or {}
    if isinstance(ak, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            v = ak.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
                break
    return "\n".join(parts) if parts else ""


def iter_json_dicts(text: str) -> list[dict]:
    """
    从整段文本及 ``` ``` 围栏内扫描所有顶层 JSON 对象（贪心 { 起点 + raw_decode）。
    同一对象可能重复出现，调用方可去重。
    """
    if not (text or "").strip():
        return []
    decoder = json.JSONDecoder()
    blobs = [(text or "").strip()] + fenced_inner_texts(text)
    out: list[dict] = []
    seen: set[str] = set()
    for blob in blobs:
        s = blob.strip()
        if not s:
            continue
        for i, ch in enumerate(s):
            if ch != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(s, i)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            sig = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(obj)
    return out
