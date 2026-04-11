"""上轮助手输出的 turn 上下文（recommended_entity / entities / info_type）解析与检索组合。"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Sequence

from .session_state import _clean_entity

logger = logging.getLogger(__name__)

RAG_TURN_JSON_LINE_PREFIX = "<<RAG_TURN_JSON>>"


def normalize_last_turn(raw: Any) -> dict[str, Any]:
    """从 context_state.last_turn 或等价结构得到规范 dict。"""
    if not isinstance(raw, dict):
        return {}
    rec = str(raw.get("recommended_entity") or "").strip()
    ents = raw.get("entities")
    if not isinstance(ents, list):
        ents = []
    ents = [_clean_entity(str(x).strip()) for x in ents if str(x).strip()]
    ents = [x for x in ents if x and len(x) >= 2][:24]
    it = raw.get("info_type")
    if isinstance(it, list):
        it_parts = [str(x).strip() for x in it if str(x).strip()]
    elif isinstance(it, str):
        it_parts = [p for p in re.split(r"[\s,，、]+", it.strip()) if p.strip()]
    else:
        it_parts = []
    it_parts = it_parts[:12]
    return {"recommended_entity": rec, "entities": ents, "info_type": it_parts}


def merge_inherited_with_extract(
    last_turn: dict[str, Any],
    current_entities: tuple[str, ...],
    current_info_types: tuple[str, ...],
) -> tuple[list[str], tuple[str, ...]]:
    """
    本轮实体列表 = 上轮 entities + 上轮 recommended_entity（若有）+ 本轮抽取。
    本轮 info_types = 上轮维度词 + 本轮抽取（去重保序）。
    """
    lt = normalize_last_turn(last_turn)
    base: list[str] = []
    seen: set[str] = set()
    for x in lt.get("entities") or []:
        s = _clean_entity(str(x).strip())
        if s and s not in seen and len(s) >= 2:
            seen.add(s)
            base.append(s)
    rec = (lt.get("recommended_entity") or "").strip()
    if rec and rec not in seen:
        seen.add(rec)
        base.append(rec)
    for x in current_entities:
        s = _clean_entity(str(x).strip())
        if s and s not in seen and len(s) >= 2:
            seen.add(s)
            base.append(s)

    it_seen: set[str] = set()
    merged_it: list[str] = []
    for part in lt.get("info_type") or []:
        s = str(part).strip()
        if s and s not in it_seen:
            it_seen.add(s)
            merged_it.append(s)
    for x in current_info_types:
        s = str(x or "").strip()
        if s and s not in it_seen:
            it_seen.add(s)
            merged_it.append(s)
    return base, tuple(merged_it)


def info_type_dimensions(info_types: Sequence[str]) -> list[str]:
    dims = [str(x).strip() for x in info_types if str(x).strip()]
    return dims if dims else []


def cartesian_seeds(entities: Sequence[str], dims: Sequence[str]) -> list[str]:
    """每个实体 × 每个维度词一条检索种子；无维度时仅实体。"""
    cap = max(int(os.getenv("RAG_CARTESIAN_SEED_CAP", "24")), 4)
    ents = [str(e).strip() for e in entities if str(e).strip()]
    dms = [str(d).strip() for d in dims if str(d).strip()]
    if not ents:
        return []
    if not dms:
        return ents[:cap]
    out: list[str] = []
    seen: set[str] = set()
    for e in ents:
        for d in dms:
            s = f"{e} {d}".strip()
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= cap:
                return out
    return out


def parse_answer_turn_json(full: str) -> tuple[str, dict[str, Any] | None]:
    """
    从助手正文末尾剥离 <<RAG_TURN_JSON>>{...} 行；返回 (展示用正文, 解析出的 dict 或 None)。
    """
    text = (full or "").rstrip()
    if not text:
        return "", None
    lines = text.split("\n")
    if not lines:
        return text, None
    last = lines[-1].strip()
    if last.startswith(RAG_TURN_JSON_LINE_PREFIX):
        raw_json = last[len(RAG_TURN_JSON_LINE_PREFIX) :].strip()
        body = "\n".join(lines[:-1]).rstrip()
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning("turn_context JSON parse failed: %r", raw_json[:200])
            return text, None
        if not isinstance(obj, dict):
            return text, None
        rec = str(obj.get("recommended_entity") or "").strip()
        ents = obj.get("entities")
        if not isinstance(ents, list):
            ents = []
        ents = [_clean_entity(str(x).strip()) for x in ents if str(x).strip()]
        ents = [x for x in ents if x and len(x) >= 2][:24]
        it_raw = obj.get("info_type")
        if isinstance(it_raw, list):
            it_list = [str(x).strip() for x in it_raw if str(x).strip()]
        else:
            it_list = [p for p in re.split(r"[\s,，、]+", str(it_raw or "").strip()) if p]
        it_list = it_list[:12]
        meta = {
            "recommended_entity": rec,
            "entities": ents,
            "info_type": it_list,
        }
        return body, meta
    # 兼容：最后一行整行 JSON（无前缀）
    if last.startswith("{") and "recommended_entity" in last:
        body = "\n".join(lines[:-1]).rstrip()
        try:
            obj = json.loads(last)
        except json.JSONDecodeError:
            return text, None
        if isinstance(obj, dict) and "recommended_entity" in obj:
            rec = str(obj.get("recommended_entity") or "").strip()
            ents = obj.get("entities")
            if not isinstance(ents, list):
                ents = []
            ents = [_clean_entity(str(x).strip()) for x in ents if str(x).strip()]
            ents = [x for x in ents if x][:24]
            it_raw = obj.get("info_type")
            if isinstance(it_raw, list):
                it_list = [str(x).strip() for x in it_raw if str(x).strip()]
            else:
                it_list = [p for p in re.split(r"[\s,，、]+", str(it_raw or "").strip()) if p]
            return body, {"recommended_entity": rec, "entities": ents, "info_type": it_list[:12]}
    return text, None


def format_turn_context_for_prompt(meta: dict[str, Any] | None) -> str:
    """写入 human 模板顶部的简短说明（若有）；供模型理解上轮结构化输出。"""
    if not meta:
        return ""
    rec = str(meta.get("recommended_entity") or "").strip()
    ents = meta.get("entities") or []
    it = meta.get("info_type") or []
    if isinstance(ents, str):
        ents = [ents] if ents.strip() else []
    if not rec and not ents and not it:
        return ""
    parts = [
        "【上轮助手结构化输出（仅用于指代与检索，非事实来源）】",
        f"- recommended_entity（最推荐实体）: {rec or '（空）'}",
        f"- entities: {', '.join(str(x) for x in ents) if ents else '（空）'}",
        f"- info_type 维度词: {', '.join(str(x) for x in it) if it else '（空）'}",
    ]
    return "\n".join(parts) + "\n"
