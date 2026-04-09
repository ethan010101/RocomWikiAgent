from __future__ import annotations

import re


def _clean_entity(v: str) -> str:
    s = str(v or "").strip()
    if s.endswith("的") and len(s) >= 3:
        s = s[:-1]
    return s


def extract_entities_from_user_text(text: str) -> list[str]:
    """从单条用户话术中抽取可能的精灵名（轻量规则，供时间线用）。"""
    t = str(text or "").strip()
    if not t:
        return []
    t = t[2:].strip() if t.startswith(("你：", "用户：")) else t
    out: list[str] = []
    seen: set[str] = set()
    bad = ("告诉我", "我想知道", "刚才", "对吧", "提问", "宠物", "精灵")
    generic_kw = ("图鉴", "筛选", "编号", "任务", "攻略", "活动", "页面")

    def push(v: str) -> None:
        s = _clean_entity(v)
        if not s or len(s) < 2 or len(s) > 16:
            return
        if re.search(r"[，。！？,.!?：:]", s):
            return
        if re.search(r"(技能|特性|进化|路线|属性|是什么|怎么|第一只|第二只|最后一只|最后一个)", s):
            return
        if any(k in s for k in generic_kw) and len(s) >= 3:
            return
        if any(b in s for b in bad):
            return
        if re.search(r"[我你他她它您]", s):
            return
        if s in seen:
            return
        seen.add(s)
        out.append(s)

    for pat in (
        r"([\u4e00-\u9fffA-Za-z0-9·]{2,16})的(?:技能|特性|进化路线|进化链|进化|属性|性格)",
        r"([\u4e00-\u9fffA-Za-z0-9·]{2,16})(?:的)?(?:技能|特性|进化路线|进化链|进化|属性)",
        r"([\u4e00-\u9fffA-Za-z0-9·]{2,16})的性格",
    ):
        for m in re.findall(pat, t):
            push(m)
    if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9·]{2,12}", t):
        push(t)
    return out


_ASSISTANT_NOISE = frozenset(
    {
        "洛克王国",
        "BWIKI",
        "王国",
        "图鉴",
        "小洛克",
        "训练师",
        "精灵",
        "宠物",
        "这个",
        "那个",
        "该",
        "本",
        "此",
        "以下",
        "上述",
    }
)


def extract_entities_from_assistant_text(text: str) -> list[str]:
    """
    从助手回复中抽取可能的词条名（《》、Markdown 链接标题、「根据…《》」等）。
    用于指代消解：用户说「它」时，上一句答案里往往已出现明确精灵名。
    """
    t = str(text or "").strip()
    if not t:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def push(raw: str) -> None:
        s = _clean_entity(raw)
        if not s or len(s) < 2 or len(s) > 16:
            return
        if re.search(r"[，。！？,.!?；;]", s):
            return
        if s in _ASSISTANT_NOISE:
            return
        if re.search(r"[我你他她它您]", s):
            return
        if s in seen:
            return
        seen.add(s)
        out.append(s)

    for m in re.findall(r"《([\u4e00-\u9fffA-Za-z0-9·．.]{2,20})》", t):
        base = m.split(".")[-1].strip() if "." in m else m
        push(base)
    for m in re.findall(r"【([\u4e00-\u9fffA-Za-z0-9·]{2,16})】", t):
        push(m)
    for m in re.findall(
        r"\[([\u4e00-\u9fffA-Za-z0-9·]{2,16})\s*[-－]\s*[^\]\n]+\]",
        t,
    ):
        push(m)
    for m in re.findall(r"根据[^《\n]{0,24}《([\u4e00-\u9fffA-Za-z0-9·]{2,16})》", t):
        push(m)
    for m in re.findall(r"关于([\u4e00-\u9fffA-Za-z0-9·]{2,12})的", t):
        push(m)

    return out


def timeline_from_history(history_messages: list | None) -> list[str]:
    if not isinstance(history_messages, list):
        return []
    out: list[str] = []
    for m in history_messages:
        if not isinstance(m, dict):
            continue
        typ = (m.get("type") or "").strip().lower()
        if typ == "user":
            txt = str(m.get("text") or m.get("content") or "").strip()
            for ent in extract_entities_from_user_text(txt):
                if ent not in out:
                    out.append(ent)
        elif typ == "assistant":
            txt = str(m.get("answer") or m.get("text") or m.get("content") or "").strip()
            for ent in extract_entities_from_assistant_text(txt):
                if ent not in out:
                    out.append(ent)
    return out


def build_session_state(
    context_state: dict | None,
    history_messages: list | None,
    *,
    resolved_subject: str = "",
) -> dict:
    """
    合并前端 context_state 与从 history 推断的实体时间线。
    返回结构：entity_timeline, focus_entity（仅用于指代/摘要，不作事实来源）。
    """
    base = context_state if isinstance(context_state, dict) else {}
    tline: list[str] = []

    raw_tl = base.get("entity_timeline")
    if isinstance(raw_tl, list):
        for it in raw_tl:
            s = _clean_entity(str(it or ""))
            if s and s not in tline:
                tline.append(s)

    for ent in timeline_from_history(history_messages):
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
