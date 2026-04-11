from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from .json_extract import ai_message_concat_text, iter_json_dicts
from .llm_compat import rag_llm_structured_output_enabled
from .session_state import _clean_entity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityExtractResult:
    """LLM 实体抽取：供会话状态与检索种子共用。"""

    entities: tuple[str, ...]
    info_types: tuple[str, ...]

    @staticmethod
    def empty() -> "EntityExtractResult":
        return EntityExtractResult((), ())


class _EntityExtractOut(BaseModel):
    entities: list[str] = Field(
        default_factory=list,
        description="本轮用户问题涉及的 Wiki 词条名，按首次出现顺序；无关则空",
    )
    info_type: list[str] = Field(
        default_factory=list,
        description="用户关心的信息维度关键词列表，如：特性、种族值、进化；无则空数组",
    )

    @field_validator("info_type", mode="before")
    @classmethod
    def _coerce_info_type_list(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        if isinstance(v, str):
            return list(_normalize_info_types(v))
        if isinstance(v, list):
            return v
        return []


def format_dialog_block(
    history_messages: list | None,
    *,
    max_turns: int,
    max_chars: int,
    current_user_input: str = "",
) -> str:
    rows: list[tuple[str, str]] = []
    if history_messages:
        for m in history_messages:
            if not isinstance(m, dict):
                continue
            t = (m.get("type") or "").strip().lower()
            if t == "user":
                text = m.get("text") or m.get("content") or ""
                role = "用户"
            elif t == "assistant":
                text = m.get("answer") or m.get("text") or m.get("content") or ""
                role = "助手"
            else:
                continue
            text = str(text).strip()
            if not text:
                continue
            rows.append((role, text))

    cur = str(current_user_input or "").strip()
    if cur:
        if not rows or rows[-1] != ("用户", cur):
            rows.append(("用户", cur))

    if not rows:
        return "（无历史）"
    if max_turns > 0:
        rows = rows[-(max_turns * 2) :]
    lines = [f"{role}：{text}" for role, text in rows]
    if max_chars <= 0:
        return "\n".join(lines)
    used = 0
    kept: list[str] = []
    for line in reversed(lines):
        n = len(line) + 1
        if kept and used + n > max_chars:
            break
        if not kept and n > max_chars:
            kept.append(line[: max(max_chars - 1, 1)] + "…")
            break
        kept.append(line)
        used += n
    kept.reverse()
    return "\n".join(kept) if kept else "（无历史）"


def _normalize_entities(raw: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in raw:
        s = _clean_entity(str(x or "").strip())
        if not s or len(s) < 2 or len(s) > 24:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_info_types(raw: Any) -> tuple[str, ...]:
    """兼容旧版 JSON 中单字符串、空格/逗号分隔、或字符串数组。"""
    if raw is None:
        return ()
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return ()
        parts = re.split(r"[\s,，、]+", s)
        raw_list = [p for p in parts if p.strip()]
    elif isinstance(raw, list):
        raw_list = [str(x) for x in raw]
    else:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for x in raw_list:
        s = str(x or "").strip()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s)
        if len(s) > 48:
            s = s[:48]
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= 12:
            break
    return tuple(out)


def _pick_entity_profile_dict(text: str) -> dict | None:
    """多段 JSON / markdown 围栏时取最后一个含 entities 或 info_type 的对象。"""
    last: dict | None = None
    for obj in iter_json_dicts(text):
        if "entities" in obj or "info_type" in obj:
            last = obj
    return last


def _parse_json_profile(text: str) -> EntityExtractResult | None:
    obj = _pick_entity_profile_dict(text)
    if not isinstance(obj, dict):
        return None
    raw_ents = obj.get("entities")
    if isinstance(raw_ents, str):
        raw_ents = [raw_ents]
    ents: list[str] = []
    if isinstance(raw_ents, list):
        ents = _normalize_entities([str(x) for x in raw_ents])
    it = _normalize_info_types(obj.get("info_type"))
    return EntityExtractResult(tuple(ents), it)


def entity_extract_profile_llm(
    history_messages: list | None,
    llm: Any,
    *,
    current_user_input: str = "",
) -> EntityExtractResult:
    """
    从多轮对话 + 本轮用户句中抽取实体列表与 info_type 维度列表（检索与 prompt 共用）。
    解析或调用失败时返回空结果（无规则回退）。
    """
    if llm is None:
        return EntityExtractResult.empty()

    max_turns = max(int(os.getenv("ENTITY_EXTRACT_MAX_TURNS", "10")), 1)
    max_chars = max(int(os.getenv("ENTITY_EXTRACT_MAX_CHARS", "4500")), 200)

    try:
        dialog = format_dialog_block(
            history_messages,
            max_turns=max_turns,
            max_chars=max_chars,
            current_user_input=current_user_input,
        )

        model_name = os.getenv("ENTITY_EXTRACT_MODEL_NAME", "").strip()
        extract_llm = llm.bind(model=model_name) if model_name else llm
        temp = float(os.getenv("ENTITY_EXTRACT_TEMPERATURE", "0"))
        extract_llm = extract_llm.bind(temperature=temp)

        system = (
            "你是洛克王国 Wiki 检索助手。根据「用户-助手」对话，输出结构化字段：\n"
            "1) entities：仅收录 **本轮用户最后一问** 中明确涉及、可能需要分别查词条的具体名称"
            "（精灵/宠物名、技能名、任务名、活动名、地图名等），按在该问句中首次出现顺序，通常 0～4 个。\n"
            "   不要把整段旧对话里出现过的实体全部列出；指代类问题若可由下文代词解析链处理，不必为指代词本身造实体。\n"
            "2) info_type：**字符串数组**，从本轮用户问题中提炼其关心的信息维度关键词（用户原话或常用说法即可），"
            "如：[\"特性\"]、[\"种族值\",\"进化\"]、[\"新手\",\"对比\"]；每个元素一个维度词，不要整句；"
            "若没有明显维度则填空数组 []。\n"
            "3) 排除泛称：宠物、精灵、任务、攻略、活动、图鉴、BWIKI、洛克王国、王国、小洛克、训练师、"
            "用户、助手 等不要放入 entities。\n"
            "4) 每个实体为简短名词短语，不要整句。"
        )
        human = f"以下是对话正文，请输出 entities 与 info_type：\n\n{dialog}"
        if not rag_llm_structured_output_enabled():
            human += (
                "\n\n只输出一个 JSON 对象，键为 entities（字符串数组）与 info_type（字符串数组），"
                "不要使用 markdown 代码块，不要其它说明文字。"
            )

        if rag_llm_structured_output_enabled():
            try:
                structured = extract_llm.with_structured_output(_EntityExtractOut)
                out = structured.invoke(
                    [SystemMessage(content=system), HumanMessage(content=human)]
                )
                if isinstance(out, _EntityExtractOut):
                    ents = _normalize_entities(out.entities)
                    its = _normalize_info_types(out.info_type)
                    return EntityExtractResult(tuple(ents), its)
                if isinstance(out, dict):
                    ents = _normalize_entities(list(out.get("entities") or []))
                    its = _normalize_info_types(out.get("info_type"))
                    return EntityExtractResult(tuple(ents), its)
            except Exception as e:
                logger.warning("entity_extract structured_output failed: %s", e)

        msg = extract_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        text = ai_message_concat_text(msg)
        parsed = _parse_json_profile(text)
        if parsed is not None:
            return parsed
        logger.warning("entity_extract: no parseable profile in model text")
        return EntityExtractResult.empty()
    except Exception as e:
        logger.warning("entity_extract failed: %s", e)
        return EntityExtractResult.empty()


def timeline_from_history_llm(
    history_messages: list | None,
    llm: Any,
    *,
    current_user_input: str = "",
) -> list[str]:
    """
    兼容旧调用：返回实体列表（顺序与 entity_extract_profile_llm 一致）。
    """
    prof = entity_extract_profile_llm(
        history_messages, llm, current_user_input=current_user_input
    )
    return list(prof.entities)
