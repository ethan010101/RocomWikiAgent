from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from .entity_extract_llm import format_dialog_block
from .json_extract import ai_message_concat_text, iter_json_dicts
from .llm_compat import rag_llm_structured_output_enabled
from .session_state import _clean_entity

logger = logging.getLogger(__name__)


def _multi_min_entities() -> int:
    return max(int(os.getenv("PRONOUN_MULTI_MIN_ENTITIES", "2")), 2)


def _multi_max_entities() -> int:
    lo = _multi_min_entities()
    return max(int(os.getenv("PRONOUN_MULTI_MAX_ENTITIES", "6")), lo)


class _PronounResolveSchema(BaseModel):
    action: Literal["resolved", "clarify", "multi_answer"] = Field(
        description="resolved=已唯一确定指代实体；clarify=需用户澄清；"
        "multi_answer=无明确推荐且多只并列时，列出多只精灵名供分别检索"
    )
    subject: str = Field(default="", description="action=resolved 时的精灵/词条名，简短")
    clarify_message: str = Field(
        default="",
        description="action=clarify 时给用户的完整中文回复，可含选项列表",
    )
    multi_subjects: list[str] = Field(
        default_factory=list,
        description="action=multi_answer 时，对话中并列讨论的全部相关精灵短名，顺序与对话一致，至少 min 个",
    )
    brief_reason: str = Field(default="", description="内部备注，一两句中文")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("action") == "dual_answer":
            data["action"] = "multi_answer"
        if data.get("dual_subjects") is not None and not data.get("multi_subjects"):
            data["multi_subjects"] = data["dual_subjects"]
        return data


@dataclass
class PronounResolveLLMResult:
    action: Literal["resolved", "clarify", "multi_answer", "pass_through"]
    subject: str = ""
    clarify_message: str = ""
    multi_subjects: list[str] = field(default_factory=list)
    brief_reason: str = ""

    def trace_dict(self) -> dict:
        return {
            "action": self.action,
            "subject": self.subject,
            "multi_subjects": self.multi_subjects,
            "brief_reason": self.brief_reason,
        }


def _parse_pronoun_json(text: str) -> dict | None:
    raw: dict | None = None
    for obj in iter_json_dicts(text):
        if isinstance(obj.get("action"), str):
            raw = obj
    if raw is None:
        return None
    if raw.get("action") == "dual_answer":
        raw["action"] = "multi_answer"
    if "dual_subjects" in raw and "multi_subjects" not in raw:
        raw["multi_subjects"] = raw["dual_subjects"]
    return raw


def _normalize_multi_subjects(xs: list[str]) -> list[str]:
    max_n = _multi_max_entities()
    out: list[str] = []
    seen: set[str] = set()
    for x in xs:
        s = _clean_entity(str(x or "").strip())
        if not s or len(s) < 2 or len(s) > 20:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_n:
            break
    return out


def _default_clarify(timeline: list[str], user_question: str) -> str:
    tl = [str(x).strip() for x in timeline if str(x).strip()]
    qfrag = _strip_q_mark(user_question)
    if len(tl) >= 2:
        tail = tl[-min(len(tl), 5) :]
        opts = "、".join(tail)
        return (
            f"您说的「它」可能指其中一只：{opts}。请直接回复精灵名，或说明想了解哪一只的「{qfrag}」。"
        )
    if len(tl) == 1:
        return (
            f"若您指的是「{tl[0]}」，请直接确认或继续提问；否则请写出精灵全名。"
        )
    return (
        "我还不能确定「它」指哪一只精灵。请写出精灵名，或先说明你在对比哪几只。"
    )


def _strip_q_mark(q: str) -> str:
    return str(q or "").strip().rstrip("？?")


def maybe_summarize_dialog_for_pronoun(dialog: str, llm: Any) -> str:
    """可选：用 LLM 压缩对话，再交给指代模型。"""
    if not dialog or dialog == "（无历史）":
        return dialog
    if os.getenv("PRONOUN_HISTORY_SUMMARY", "false").lower() not in ("1", "true", "yes"):
        return dialog
    try:
        model_name = os.getenv("PRONOUN_SUMMARY_MODEL_NAME", "").strip()
        mdl = llm.bind(model=model_name) if model_name else llm
        mdl = mdl.bind(temperature=0)
        sys = (
            "将下列「用户 / 助手」对话压缩为要点列表（最多约 400 字）。必须保留：\n"
            "- 对比或提及了哪些具体精灵/词条名；\n"
            "- 助手是否明确推荐其中一只（若有，写出被推荐者名字及理由关键词）；\n"
            "- 与用户当前追问相关的结论。\n"
            "不得编造对话里未出现的信息。输出纯文本列表即可，不要 JSON。"
        )
        msg = mdl.invoke(
            [SystemMessage(content=sys), HumanMessage(content=dialog)]
        )
        text = getattr(msg, "content", None) or ""
        if isinstance(text, list):
            text = "".join(
                getattr(b, "text", str(b)) if not isinstance(b, str) else b for b in text
            )
        text = str(text).strip()
        return text if len(text) >= 20 else dialog
    except Exception as e:
        logger.warning("pronoun history summary failed: %s", e)
        return dialog


def resolve_pronoun_with_llm(
    *,
    user_input: str,
    history_messages: list | None,
    entity_timeline: list[str],
    focus_entity: str,
    llm: Any,
) -> PronounResolveLLMResult:
    """
    用语义模型判定「它/该/这个」等指代对象，或输出澄清话术 / 多实体并列检索。
    失败时返回 pass_through，由上游保留规则 resolve 结果。
    """
    if llm is None:
        return PronounResolveLLMResult(action="pass_through")

    max_turns = max(int(os.getenv("PRONOUN_CTX_MAX_TURNS", "10")), 1)
    max_chars = max(int(os.getenv("PRONOUN_CTX_MAX_CHARS", "4500")), 200)

    raw_dialog = format_dialog_block(
        history_messages,
        max_turns=max_turns,
        max_chars=max_chars,
        current_user_input=user_input,
    )
    dialog = maybe_summarize_dialog_for_pronoun(raw_dialog, llm)

    timeline_txt = "、".join(str(x).strip() for x in entity_timeline if str(x).strip()) or "（无）"
    focus_s = str(focus_entity or "").strip() or "（无）"
    min_m = _multi_min_entities()
    max_m = _multi_max_entities()

    model_name = os.getenv("PRONOUN_RESOLVE_MODEL_NAME", "").strip()
    mdl = llm.bind(model=model_name) if model_name else llm
    mdl = mdl.bind(temperature=float(os.getenv("PRONOUN_RESOLVE_TEMPERATURE", "0")))

    system = (
        "你是洛克王国对话系统中的「指代消解」模块。根据对话与实体线索，判断用户当前句中的"
        "「它/他/她/该/这个/那个」等代词最可能指哪只精灵（或哪个已讨论的词条）。\n\n"
        "判定规则：\n"
        "1) 若助手在上一轮（或最近回复）中**明确推荐**了某一只精灵/选项（例如明确说「更适合新手」"
        "「更推荐」「建议选」等并指向具体名字），且用户追问与推荐结论强相关，则 action=resolved，"
        "subject 填**被推荐的那只**的标准短名称。\n"
        f"2) 若对话中**同时讨论了两只及以上**精灵，但**没有**明确推荐其中一只，且用户用代词追问具体属性"
        f"（如进化路线、技能、种族值等），则 action=multi_answer；multi_subjects 须列出**全部**"
        f"并列讨论的精灵短名（与 entity_timeline 用词尽量一致），至少 {min_m} 个、至多 {max_m} 个，"
        "顺序与对话中出现或对比顺序一致。\n"
        "3) 若仍无法确定（信息不足、代词含糊），则 action=clarify，在 clarify_message "
        "中写一段礼貌的中文，请用户点选或写出精灵名；可列出候选。\n"
        "4) subject / multi_subjects 不要带「的」「这只」等赘字；不要编造未在对话出现的精灵名。\n"
        "5) brief_reason 用一两句说明你的判断依据（中文）。"
    )
    human = (
        f"【本轮用户问题】\n{user_input.strip()}\n\n"
        f"【对话正文（可能已压缩）】\n{dialog}\n\n"
        f"【实体时间线（供对齐用词，非事实依据）】\n{timeline_txt}\n"
        f"【当前焦点实体（若有）】\n{focus_s}\n"
    )
    if not rag_llm_structured_output_enabled():
        human += (
            "\n\n只输出一个 JSON 对象，键：action（resolved|clarify|multi_answer 之一）、subject、"
            "clarify_message、multi_subjects（字符串数组）、brief_reason。不要使用 markdown 代码块，不要其它文字。"
        )

    if rag_llm_structured_output_enabled():
        try:
            structured = mdl.with_structured_output(_PronounResolveSchema)
            out = structured.invoke([SystemMessage(content=system), HumanMessage(content=human)])
            if isinstance(out, _PronounResolveSchema):
                return _schema_to_result(out, entity_timeline, user_input)
            if isinstance(out, dict):
                ps = _PronounResolveSchema.model_validate(out)
                return _schema_to_result(ps, entity_timeline, user_input)
        except Exception as e:
            logger.warning("pronoun_resolve structured_output failed: %s", e)

    try:
        msg = mdl.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        text = ai_message_concat_text(msg)
        raw = _parse_pronoun_json(text)
        if raw and isinstance(raw.get("action"), str):
            ps = _PronounResolveSchema.model_validate(raw)
            return _schema_to_result(ps, entity_timeline, user_input)
    except Exception as e:
        logger.warning("pronoun_resolve json fallback failed: %s", e)

    return PronounResolveLLMResult(action="pass_through")


def _schema_to_result(
    out: _PronounResolveSchema,
    entity_timeline: list[str],
    user_input: str,
) -> PronounResolveLLMResult:
    act = out.action
    min_m = _multi_min_entities()
    if act == "resolved":
        subj = _clean_entity((out.subject or "").strip())
        if subj:
            return PronounResolveLLMResult(
                action="resolved",
                subject=subj,
                brief_reason=out.brief_reason,
            )
        msg = (out.clarify_message or "").strip() or _default_clarify(entity_timeline, user_input)
        return PronounResolveLLMResult(action="clarify", clarify_message=msg, brief_reason=out.brief_reason)

    if act == "clarify":
        msg = (out.clarify_message or "").strip() or _default_clarify(entity_timeline, user_input)
        return PronounResolveLLMResult(action="clarify", clarify_message=msg, brief_reason=out.brief_reason)

    if act == "multi_answer":
        multi = _normalize_multi_subjects(list(out.multi_subjects or []))
        if len(multi) >= min_m:
            return PronounResolveLLMResult(
                action="multi_answer",
                multi_subjects=multi,
                brief_reason=out.brief_reason,
            )
        msg = (out.clarify_message or "").strip() or _default_clarify(entity_timeline, user_input)
        return PronounResolveLLMResult(action="clarify", clarify_message=msg, brief_reason=out.brief_reason)

    return PronounResolveLLMResult(action="pass_through")


def multi_min_entities() -> int:
    """供 orchestrate 判断 multi_answer 分支是否与配置一致。"""
    return _multi_min_entities()


def multi_max_entities() -> int:
    return _multi_max_entities()
