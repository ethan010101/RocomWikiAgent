from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .dialogue_budget import budget_for_kind, format_history_block
from .query_resolve import resolve_query, retrieval_seed_question
from .retrieval_gate import evaluate_gate
from .session_state import build_session_state, format_session_summary
from .types import PipelineResult, ResolvedQuery


@dataclass
class RAGPrepareResult:
    """prepare 阶段结果：供非流式一次生成与 SSE 流式共用。"""

    output_direct: str | None
    docs: list
    prompt_vars: dict[str, str] | None
    trace: dict[str, Any]


def prepare_context_rag_turn(
    *,
    user_input: str,
    history_messages: list | None,
    context_state: dict | None,
    gather_docs: Callable[..., list],
    format_docs: Callable[[list], str],
    retrieval_query_suffix: Callable[[str], str],
) -> RAGPrepareResult:
    trace: dict[str, Any] = {"version": "rag_pipeline_v1", "stages": {}}

    merged_state = build_session_state(context_state, history_messages, resolved_subject="")
    trace["stages"]["ingest"] = {
        "module": "rag_pipeline.session_state",
        "timeline_len": len(merged_state.get("entity_timeline") or []),
    }

    resolved: ResolvedQuery = resolve_query(user_input, merged_state)
    trace["stages"]["resolve"] = {
        "module": "rag_pipeline.query_resolve",
        "kind": resolved.kind,
        "subject": resolved.subject,
        "reason": resolved.reason,
    }

    merged_state = build_session_state(
        context_state, history_messages, resolved_subject=resolved.subject
    )

    seed = retrieval_seed_question(resolved, user_input)
    main_query = retrieval_query_suffix(seed)
    docs = gather_docs(user_input, main_query)
    trace["stages"]["retrieve"] = {
        "module": "agent._gather_docs",
        "doc_count": len(docs),
        "retrieval_seed": seed[:200],
    }

    gate = evaluate_gate(resolved, docs)
    trace["stages"]["gate"] = {
        "module": "rag_pipeline.retrieval_gate",
        "code": gate.code,
        "action": gate.action,
        "meta": gate.meta,
    }

    if gate.action == "respond_direct":
        return RAGPrepareResult(
            output_direct=gate.user_message,
            docs=docs,
            prompt_vars=None,
            trace=trace,
        )

    b = budget_for_kind(resolved.kind)
    history_block = format_history_block(history_messages, query_kind=resolved.kind)
    summary_block = format_session_summary(
        merged_state, timeline_max=int(b.get("summary_timeline_max") or 12)
    )
    ctx_block = format_docs(docs)
    trace["stages"]["assemble"] = {
        "module": "rag_pipeline.dialogue_budget + session_state",
        "history_chars": len(history_block),
        "summary_chars": len(summary_block),
        "context_chars": len(ctx_block),
    }

    return RAGPrepareResult(
        output_direct=None,
        docs=docs,
        prompt_vars={
            "session_summary": summary_block,
            "history": history_block,
            "context": ctx_block,
            "input": user_input,
        },
        trace=trace,
    )


def run_context_rag_turn(
    *,
    user_input: str,
    history_messages: list | None,
    context_state: dict | None,
    gather_docs: Callable[..., list],
    format_docs: Callable[[list], str],
    prompt: ChatPromptTemplate,
    llm: Any,
    retrieval_query_suffix: Callable[[str], str],
) -> PipelineResult:
    """
    单次 RAG 流水线入口。阶段划分固定，trace 键稳定，便于对照日志定位模块。

    阶段：ingest -> resolve -> retrieve -> gate -> assemble -> generate
    """
    t0 = time.perf_counter()
    prep = prepare_context_rag_turn(
        user_input=user_input,
        history_messages=history_messages,
        context_state=context_state,
        gather_docs=gather_docs,
        format_docs=format_docs,
        retrieval_query_suffix=retrieval_query_suffix,
    )
    trace = prep.trace

    if prep.output_direct is not None:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        trace["stages"]["generate"] = {"module": None, "skipped": True}
        return PipelineResult(
            output=prep.output_direct,
            docs=prep.docs,
            latency_ms=elapsed_ms,
            trace=trace,
        )

    t1 = time.perf_counter()
    prompt_val = prompt.invoke(prep.prompt_vars)
    msg = llm.invoke(prompt_val)
    text = StrOutputParser().invoke(msg)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    trace["stages"]["generate"] = {
        "module": "langchain_openai.ChatOpenAI",
        "llm_ms": round((time.perf_counter() - t1) * 1000, 2),
    }

    return PipelineResult(output=text, docs=prep.docs, latency_ms=elapsed_ms, trace=trace)
