from __future__ import annotations


def _clean_entity(v: str) -> str:
    s = str(v or "").strip()
    if s.endswith("的") and len(s) >= 3:
        s = s[:-1]
    return s


def build_session_state(
    context_state: dict | None,
    _history_messages: list | None,
    *,
    resolved_subject: str = "",
    history_inferred_entities: list[str] | None = None,
) -> dict:
    """
    合并前端 context_state 与从 history 推断的实体时间线。
    返回结构：entity_timeline, focus_entity（仅用于指代/摘要，不作事实来源）。

    history_inferred_entities：须由上游用大模型生成（见 entity_extract_llm）；为 None 时视为无历史实体。
    _history_messages：保留参数以兼容旧调用；实体时间线不再从原始 history 规则解析。
    """
    base = context_state if isinstance(context_state, dict) else {}
    tline: list[str] = []

    raw_tl = base.get("entity_timeline")
    if isinstance(raw_tl, list):
        for it in raw_tl:
            s = _clean_entity(str(it or ""))
            if s and s not in tline:
                tline.append(s)

    inferred = list(history_inferred_entities) if history_inferred_entities else []
    for ent in inferred:
        if ent not in tline:
            tline.append(ent)

    rs = _clean_entity(resolved_subject)
    if rs and rs not in tline:
        tline.append(rs)

    focus = _clean_entity(str(base.get("focus_entity") or ""))
    if not focus and tline:
        focus = tline[-1]
    if rs:
        focus = rs

    return {"entity_timeline": tline, "focus_entity": focus}


def format_session_summary(state: dict | None, *, timeline_max: int) -> str:
    if not isinstance(state, dict):
        return "（无）"
    tl = state.get("entity_timeline")
    timeline: list[str] = []
    if isinstance(tl, list):
        for x in tl:
            s = str(x).strip()
            if s:
                timeline.append(s)
    timeline = timeline[:timeline_max]
    focus = str(state.get("focus_entity") or "").strip()
    if not timeline and not focus:
        return "（无）"
    head = f"当前焦点实体：{focus}" if focus else "当前焦点实体：（无）"
    if not timeline:
        return head
    return head + "\n已提及实体顺序：" + " -> ".join(timeline)
