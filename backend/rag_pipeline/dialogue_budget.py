from __future__ import annotations

import os


def budget_for_kind(kind: str) -> dict:
    """指代/序号类问句给更大历史预算，事实类更省 token。"""
    k = (kind or "general").strip().lower()
    if k in ("pronoun", "ordinal"):
        return {
            "history_turns": max(int(os.getenv("RAG_CTX_REF_TURNS", "8")), 1),
            "history_chars": max(int(os.getenv("RAG_CTX_REF_CHARS", "2800")), 200),
            "summary_timeline_max": max(int(os.getenv("RAG_CTX_REF_TIMELINE_MAX", "16")), 4),
        }
    return {
        "history_turns": max(int(os.getenv("RAG_CTX_FACT_TURNS", "4")), 0),
        "history_chars": max(int(os.getenv("RAG_CTX_FACT_CHARS", "1600")), 0),
        "summary_timeline_max": max(int(os.getenv("RAG_CTX_FACT_TIMELINE_MAX", "12")), 4),
    }


def trim_lines(lines: list[str], max_chars: int) -> list[str]:
    if max_chars <= 0:
        return lines
    kept: list[str] = []
    used = 0
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
    return kept


def format_history_block(history_messages: list | None, *, query_kind: str) -> str:
    if not history_messages:
        return "（无）"
    rows: list[tuple[str, str]] = []
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
    if not rows:
        return "（无）"
    b = budget_for_kind(query_kind)
    turns = int(b.get("history_turns") or 0)
    if turns > 0:
        rows = rows[-(turns * 2) :]
    lines = [f"{role}：{text}" for role, text in rows]
    max_chars = int(b.get("history_chars") or 0)
    if max_chars <= 0:
        return "\n".join(lines)
    kept = trim_lines(lines, max_chars=max_chars)
    return "\n".join(kept) if kept else "（无）"
