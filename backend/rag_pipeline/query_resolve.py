from __future__ import annotations

import re

from .types import QueryKind, ResolvedQuery


def _explicit_entity_hint(q: str) -> str:
    generic = {"技能", "特性", "进化", "路线", "进化链", "属性", "强度", "怎么", "什么"}
    bad = ("告诉我", "我想知道", "刚才", "提问", "宠物", "第一个", "第一只", "第二只", "第三只", "最后一只", "最后一个")
    for pat in (
        r"([\u4e00-\u9fffA-Za-z0-9·]{2,16})的(?:技能|特性|进化路线|进化链|进化|属性|性格)",
        r"([\u4e00-\u9fffA-Za-z0-9·]{2,16})(?:的)?(?:技能|特性|进化路线|进化链|进化|属性)",
        r"([\u4e00-\u9fffA-Za-z0-9·]{2,16})的性格",
    ):
        ms = re.findall(pat, q or "")
        if not ms:
            continue
        cand = sorted(ms, key=len, reverse=True)[0].strip("，。！？,.!? ")
        if not cand or cand in generic or any(x in cand for x in bad):
            continue
        return cand
    return ""


def _zh_num(tok: str) -> int:
    m = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    return m.get(tok, 0)


def resolve_query(question: str, session_state: dict) -> ResolvedQuery:
    """
    规则解析问句类型与检索主体（不调用 LLM）。
    新意图类型请在模块内增分支，并在 types.QueryKind 中扩展。
    """
    q = (question or "").strip()
    timeline = []
    if isinstance(session_state, dict):
        arr = session_state.get("entity_timeline")
        if isinstance(arr, list):
            timeline = [str(x).strip() for x in arr if str(x).strip()]

    focus = ""
    if isinstance(session_state, dict):
        focus = str(session_state.get("focus_entity") or "").strip()

    if re.search(r"(最后一只|最后一个)", q):
        if not timeline:
            return ResolvedQuery(
                kind="ordinal",
                subject="",
                ordinal_index=-1,
                raw_question=q,
                reason="ordinal_last_no_timeline",
            )
        return ResolvedQuery(
            kind="ordinal",
            subject=timeline[-1],
            ordinal_index=-1,
            raw_question=q,
            reason="ordinal_last",
        )

    m_num = re.search(r"第\s*(\d+)\s*(?:个|只)?(?:宠物|精灵)?|第\s*(\d+)\s*(?:个|只)(?!.*宠物)", q)
    if m_num:
        n = int((m_num.group(1) or m_num.group(2) or "0"))
        if n > 0:
            if n <= len(timeline):
                return ResolvedQuery(
                    kind="ordinal",
                    subject=timeline[n - 1],
                    ordinal_index=n,
                    raw_question=q,
                    reason="ordinal_index",
                )
            return ResolvedQuery(
                kind="ordinal",
                subject="",
                ordinal_index=n,
                raw_question=q,
                reason="ordinal_out_of_range",
            )

    m_cn = re.search(
        r"第\s*([一二三四五六七八九十两]+)\s*(?:个|只)?(?:宠物|精灵)?|第\s*([一二三四五六七八九十两]+)\s*(?:个|只)(?!.*宠物)",
        q,
    )
    if m_cn:
        tok = m_cn.group(1) or m_cn.group(2) or ""
        n = _zh_num(tok)
        if n > 0:
            if n <= len(timeline):
                return ResolvedQuery(
                    kind="ordinal",
                    subject=timeline[n - 1],
                    ordinal_index=n,
                    raw_question=q,
                    reason="ordinal_index_zh",
                )
            return ResolvedQuery(
                kind="ordinal",
                subject="",
                ordinal_index=n,
                raw_question=q,
                reason="ordinal_out_of_range_zh",
            )

    # 不用单字「其」，避免命中「其他」等词中的子串
    if re.search(r"(它|他|她|该|这个|那个)(?:的|呢|是|会|能)?", q) or re.search(
        r"^(它|他|她|该|这个|那个)[^？?]*[？?]?$", q
    ):
        subj = focus or (timeline[-1] if timeline else "")
        if subj:
            return ResolvedQuery(
                kind="pronoun",
                subject=subj,
                ordinal_index=0,
                raw_question=q,
                reason="pronoun",
            )
        return ResolvedQuery(
            kind="pronoun",
            subject="",
            ordinal_index=0,
            raw_question=q,
            reason="pronoun_no_timeline",
        )

    hint = _explicit_entity_hint(q)
    if hint:
        return ResolvedQuery(
            kind="explicit_entity",
            subject=hint,
            ordinal_index=0,
            raw_question=q,
            reason="explicit_pattern",
        )

    return ResolvedQuery(
        kind="general",
        subject="",
        ordinal_index=0,
        raw_question=q,
        reason="general",
    )


def retrieval_seed_question(resolved: ResolvedQuery, user_input: str) -> str:
    """拼进向量检索的主查询（主体 + 原问句），general 时仅原问句由调用方加后缀。"""
    q = (user_input or "").strip()
    subj = (resolved.subject or "").strip()
    if subj and subj not in q:
        return f"{subj} {q}".strip()
    return q
