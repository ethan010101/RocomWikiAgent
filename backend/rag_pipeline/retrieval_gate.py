from __future__ import annotations

import os

from .types import GateResult, ResolvedQuery


def _subject_hit_rank(doc, subject: str) -> int:
    s = (subject or "").strip()
    if not s:
        return 0
    title = (doc.metadata.get("title") or "").strip()
    src = (doc.metadata.get("source") or "").strip()
    head = (doc.page_content or "")[:220]
    if title == s:
        return 0
    if s in title:
        return 1
    if s in src:
        return 2
    if s in head:
        return 3
    return 9


def subject_hit_count(docs, subject: str) -> int:
    if not subject:
        return len(docs or [])
    n = 0
    for d in docs or []:
        if _subject_hit_rank(d, subject) <= 3:
            n += 1
    return n


def evaluate_gate(resolved: ResolvedQuery, docs: list) -> GateResult:
    """
    检索与生成之间的契约：在此集中判定是否跳过 LLM、返回固定话术。
    新增门控条件请写在本函数的策略分支中，并赋稳定 code 便于日志排查。
    """
    subj = (resolved.subject or "").strip()
    kind = resolved.kind

    if kind == "ordinal" and not subj:
        return GateResult(
            action="respond_direct",
            user_message="我还不能确定你指的是哪一只。请先列出几只精灵，或直接用精灵名提问。",
            code="GATE_ORDINAL_NO_SUBJECT",
            meta={"kind": kind, "reason": resolved.reason},
        )

    if kind == "pronoun" and not subj:
        return GateResult(
            action="respond_direct",
            user_message="我还不能确定「它」指哪只精灵。请先说精灵名，或完整描述你的问题。",
            code="GATE_PRONOUN_NO_SUBJECT",
            meta={"kind": kind, "reason": resolved.reason},
        )

    if not docs:
        return GateResult(
            action="respond_direct",
            user_message="根据当前检索到的 wiki 片段未找到与你问题直接相关的内容；请尝试换一种说法或补充精灵/技能名后再问。",
            code="GATE_NO_DOCUMENTS",
            meta={"kind": kind},
        )

    gate_on = os.getenv("RAG_CONTEXT_SUBJECT_GATE", "false").lower() in ("1", "true", "yes")
    min_hits = max(int(os.getenv("RAG_CONTEXT_SUBJECT_GATE_MIN", "1")), 1)
    if gate_on and subj and subject_hit_count(docs, subj) < min_hits:
        return GateResult(
            action="respond_direct",
            user_message=(
                f"检索结果中未能精确定位到「{subj}」对应词条片段；请确认名称与图鉴一致，或换用更完整的提问。"
            ),
            code="GATE_SUBJECT_WEAK_HIT",
            meta={"subject": subj, "doc_count": len(docs)},
        )

    return GateResult(
        action="proceed",
        user_message="",
        code="GATE_PROCEED",
        meta={"doc_count": len(docs), "subject_hits": subject_hit_count(docs, subj) if subj else len(docs)},
    )
