from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .dialogue_budget import budget_for_kind, format_history_block
from .entity_extract_llm import EntityExtractResult, entity_extract_profile_llm
from .pronoun_resolve_llm import (
    multi_max_entities,
    multi_min_entities,
    resolve_pronoun_with_llm,
)
from .query_resolve import (
    resolve_query,
    retrieval_seed_from_llm_entities,
    retrieval_seed_question,
)
from .retrieval_gate import evaluate_gate
from .session_state import build_session_state, format_session_summary
from .turn_context import (
    cartesian_seeds,
    format_turn_context_for_prompt,
    info_type_dimensions,
    merge_inherited_with_extract,
    normalize_last_turn,
    parse_answer_turn_json,
)
from .types import PipelineResult, ResolvedQuery


def _multi_entity_extract_min() -> int:
    return max(int(os.getenv("RAG_MULTI_ENTITY_EXTRACT_MIN", "2")), 2)


def _gather_multi_subject_chunks(
    subjects: list[str],
    *,
    user_input: str,
    seed_for_subject: Callable[[str], str],
    preamble: str,
    gather_docs: Callable[..., list],
    format_docs: Callable[[list], str],
    retrieval_query_suffix: Callable[[str], str],
) -> tuple[list, str]:
    docs_acc: list = []
    ctx_parts: list[str] = []
    for s in subjects:
        seed_s = seed_for_subject(s).strip()
        mq = retrieval_query_suffix(seed_s)
        part_docs = gather_docs(user_input, mq)
        docs_acc.extend(part_docs)
        ctx_parts.append(
            f"=== 与「{s}」相关的 wiki 片段（请分条作答）===\n" + format_docs(part_docs)
        )
    ctx_block = preamble + "\n\n".join(ctx_parts)
    return docs_acc, ctx_block


def _gather_by_seed_strings(
    seeds: list[str],
    *,
    user_input: str,
    preamble: str,
    gather_docs: Callable[..., list],
    format_docs: Callable[[list], str],
    retrieval_query_suffix: Callable[[str], str],
) -> tuple[list, str]:
    docs_acc: list = []
    ctx_parts: list[str] = []
    for i, seed in enumerate(seeds):
        mq = retrieval_query_suffix(seed.strip())
        part_docs = gather_docs(user_input, mq)
        docs_acc.extend(part_docs)
        ctx_parts.append(f"=== 检索组合 {i + 1}: {seed} ===\n" + format_docs(part_docs))
    return docs_acc, preamble + "\n\n".join(ctx_parts)


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
    llm: Any | None = None,
) -> RAGPrepareResult:
    trace: dict[str, Any] = {"version": "rag_pipeline_v1", "stages": {}}

    last_turn_raw: dict | None = None
    if isinstance(context_state, dict):
        lt = context_state.get("last_turn")
        if isinstance(lt, dict):
            last_turn_raw = lt
    prior_turn_context = format_turn_context_for_prompt(
        normalize_last_turn(last_turn_raw) if last_turn_raw else None
    )

    raw_extract = entity_extract_profile_llm(
        history_messages, llm, current_user_input=user_input
    )
    merged_ent_list, merged_info_tuple = merge_inherited_with_extract(
        last_turn_raw or {},
        raw_extract.entities,
        raw_extract.info_types,
    )
    extract_profile = EntityExtractResult(
        tuple(merged_ent_list),
        merged_info_tuple,
    )
    inferred_entities = list(extract_profile.entities)
    merged_state = build_session_state(
        context_state,
        history_messages,
        resolved_subject="",
        history_inferred_entities=inferred_entities,
    )
    trace["stages"]["ingest"] = {
        "module": "rag_pipeline.session_state",
        "timeline_len": len(merged_state.get("entity_timeline") or []),
        "entity_extract": "llm" if llm is not None else "skipped_no_llm",
        "entity_extract_entities_raw": list(raw_extract.entities),
        "entity_extract_info_types_raw": list(raw_extract.info_types),
        "entity_extract_entities_merged": list(extract_profile.entities),
        "entity_extract_info_types_merged": list(extract_profile.info_types),
        "last_turn_in": normalize_last_turn(last_turn_raw) if last_turn_raw else {},
    }

    resolved: ResolvedQuery = resolve_query(user_input, merged_state)
    trace["stages"]["resolve"] = {
        "module": "rag_pipeline.query_resolve",
        "kind": resolved.kind,
        "subject": resolved.subject,
        "reason": resolved.reason,
    }

    last_turn = normalize_last_turn(last_turn_raw)
    use_recommended_retrieval = (
        bool(last_turn.get("recommended_entity")) and resolved.kind != "ordinal"
    )

    if use_recommended_retrieval:
        rec = str(last_turn.get("recommended_entity") or "").strip()
        dims = info_type_dimensions(extract_profile.info_types)
        seeds = cartesian_seeds([rec], dims)
        if not seeds:
            seeds = [rec]
        docs_acc, ctx_block = _gather_by_seed_strings(
            seeds,
            user_input=user_input,
            preamble=(
                "【本轮说明】已根据上一轮助手给出的「最推荐实体」与本轮信息维度组合检索；"
                "请围绕推荐实体与参考资料作答；末尾给出参考链接（去重）。\n\n"
            ),
            gather_docs=gather_docs,
            format_docs=format_docs,
            retrieval_query_suffix=retrieval_query_suffix,
        )
        gate_resolved = ResolvedQuery(
            kind="general",
            subject=rec,
            ordinal_index=0,
            raw_question=user_input,
            reason="turn_recommended_seed",
        )
        trace["stages"]["retrieve"] = {
            "module": "agent._gather_docs",
            "doc_count": len(docs_acc),
            "retrieval_seed": "recommended_x_info_dims",
            "recommended_entity": rec,
            "seeds": seeds[:32],
        }
        gate = evaluate_gate(gate_resolved, docs_acc)
        trace["stages"]["gate"] = {
            "module": "rag_pipeline.retrieval_gate",
            "code": gate.code,
            "action": gate.action,
            "meta": gate.meta,
        }
        if gate.action == "respond_direct":
            return RAGPrepareResult(
                output_direct=gate.user_message,
                docs=docs_acc,
                prompt_vars=None,
                trace=trace,
            )
        b = budget_for_kind("general")
        history_block = format_history_block(history_messages, query_kind="general")
        summary_block = format_session_summary(
            merged_state, timeline_max=int(b.get("summary_timeline_max") or 12)
        )
        trace["stages"]["assemble"] = {
            "module": "rag_pipeline.dialogue_budget + session_state",
            "history_chars": len(history_block),
            "summary_chars": len(summary_block),
            "context_chars": len(ctx_block),
        }
        return RAGPrepareResult(
            output_direct=None,
            docs=docs_acc,
            prompt_vars={
                "prior_turn_context": prior_turn_context,
                "session_summary": summary_block,
                "history": history_block,
                "context": ctx_block,
                "input": user_input,
            },
            trace=trace,
        )

    if llm is not None and resolved.kind == "pronoun":
        pr = resolve_pronoun_with_llm(
            user_input=user_input,
            history_messages=history_messages,
            entity_timeline=list(merged_state.get("entity_timeline") or []),
            focus_entity=str(merged_state.get("focus_entity") or ""),
            llm=llm,
        )
        trace["stages"]["pronoun_llm"] = pr.trace_dict()

        if pr.action == "clarify":
            return RAGPrepareResult(
                output_direct=pr.clarify_message,
                docs=[],
                prompt_vars=None,
                trace=trace,
            )

        if pr.action == "multi_answer" and len(pr.multi_subjects) >= multi_min_entities():
            resolved = ResolvedQuery(
                kind="general",
                subject="",
                ordinal_index=0,
                raw_question=user_input,
                reason="multi_pronoun_llm",
            )
            subjs = pr.multi_subjects
            trace["stages"]["resolve"] = {
                "module": "rag_pipeline.query_resolve+pronoun_llm",
                "kind": resolved.kind,
                "subject": resolved.subject,
                "reason": resolved.reason,
                "multi_subjects": list(subjs),
            }

            n = len(subjs)
            docs_acc, ctx_block = _gather_multi_subject_chunks(
                subjs,
                user_input=user_input,
                seed_for_subject=lambda s: retrieval_seed_question(
                    ResolvedQuery(
                        kind="pronoun",
                        subject=s,
                        ordinal_index=0,
                        raw_question=user_input,
                        reason="multi_seed",
                    ),
                    user_input,
                ),
                preamble=(
                    "【本轮说明】用户用代词追问；对话中有多只并列候选且助手未明确推荐其一。\n"
                    f"以下参考资料按实体分块（共 {n} 块）。请**分别**依据每一块回答用户问题；"
                    "为每一块使用清晰小标题（精灵/词条名），勿混用不同块的设定；末尾给出参考链接（去重）。\n\n"
                ),
                gather_docs=gather_docs,
                format_docs=format_docs,
                retrieval_query_suffix=retrieval_query_suffix,
            )
            trace["stages"]["retrieve"] = {
                "module": "agent._gather_docs",
                "doc_count": len(docs_acc),
                "retrieval_seed": "multi_pronoun",
                "multi_subjects": list(subjs),
            }

            gate = evaluate_gate(resolved, docs_acc)
            trace["stages"]["gate"] = {
                "module": "rag_pipeline.retrieval_gate",
                "code": gate.code,
                "action": gate.action,
                "meta": gate.meta,
            }
            if gate.action == "respond_direct":
                return RAGPrepareResult(
                    output_direct=gate.user_message,
                    docs=docs_acc,
                    prompt_vars=None,
                    trace=trace,
                )

            merged_state = build_session_state(
                context_state,
                history_messages,
                resolved_subject=resolved.subject,
                history_inferred_entities=inferred_entities,
            )
            b = budget_for_kind(resolved.kind)
            history_block = format_history_block(
                history_messages, query_kind=resolved.kind
            )
            summary_block = format_session_summary(
                merged_state, timeline_max=int(b.get("summary_timeline_max") or 12)
            )
            trace["stages"]["assemble"] = {
                "module": "rag_pipeline.dialogue_budget + session_state",
                "history_chars": len(history_block),
                "summary_chars": len(summary_block),
                "context_chars": len(ctx_block),
            }
            return RAGPrepareResult(
                output_direct=None,
                docs=docs_acc,
                prompt_vars={
                    "prior_turn_context": prior_turn_context,
                    "session_summary": summary_block,
                    "history": history_block,
                    "context": ctx_block,
                    "input": user_input,
                },
                trace=trace,
            )

        if pr.action == "resolved" and pr.subject.strip():
            resolved = ResolvedQuery(
                kind="pronoun",
                subject=pr.subject.strip(),
                ordinal_index=0,
                raw_question=user_input,
                reason="pronoun_llm",
            )
            trace["stages"]["resolve"] = {
                "module": "rag_pipeline.query_resolve+pronoun_llm",
                "kind": resolved.kind,
                "subject": resolved.subject,
                "reason": resolved.reason,
            }
        # pass_through：保留 query_resolve 的规则 subject

    merged_state = build_session_state(
        context_state,
        history_messages,
        resolved_subject=resolved.subject,
        history_inferred_entities=inferred_entities,
    )

    subjs_extract = list(extract_profile.entities)[: multi_max_entities()]
    if (
        llm is not None
        and len(subjs_extract) >= _multi_entity_extract_min()
        and resolved.kind in ("general", "explicit_entity")
    ):
        resolved = ResolvedQuery(
            kind="general",
            subject="",
            ordinal_index=0,
            raw_question=user_input,
            reason="multi_entity_extract",
        )
        trace["stages"]["resolve"] = {
            "module": "rag_pipeline.query_resolve+entity_extract",
            "kind": resolved.kind,
            "subject": resolved.subject,
            "reason": resolved.reason,
            "multi_subjects": list(subjs_extract),
            "info_type": list(extract_profile.info_types),
        }
        dims = info_type_dimensions(extract_profile.info_types)
        seeds = cartesian_seeds(subjs_extract, dims)
        n_ext = len(seeds)
        docs_acc, ctx_block = _gather_by_seed_strings(
            seeds,
            user_input=user_input,
            preamble=(
                "【本轮说明】当前问题涉及多个词条，已按「实体 × 信息维度」组合分别检索。\n"
                f"以下共 {n_ext} 块。请**分别**依据每一块回答用户问题；"
                "为每一块使用清晰小标题（精灵/词条名），勿混用不同块的设定；末尾给出参考链接（去重）。\n\n"
            ),
            gather_docs=gather_docs,
            format_docs=format_docs,
            retrieval_query_suffix=retrieval_query_suffix,
        )
        trace["stages"]["retrieve"] = {
            "module": "agent._gather_docs",
            "doc_count": len(docs_acc),
            "retrieval_seed": "multi_entity_cartesian",
            "multi_subjects": list(subjs_extract),
            "seeds": seeds[:32],
            "info_type": list(extract_profile.info_types),
        }

        gate = evaluate_gate(resolved, docs_acc)
        trace["stages"]["gate"] = {
            "module": "rag_pipeline.retrieval_gate",
            "code": gate.code,
            "action": gate.action,
            "meta": gate.meta,
        }
        if gate.action == "respond_direct":
            return RAGPrepareResult(
                output_direct=gate.user_message,
                docs=docs_acc,
                prompt_vars=None,
                trace=trace,
            )

        b = budget_for_kind(resolved.kind)
        history_block = format_history_block(
            history_messages, query_kind=resolved.kind
        )
        summary_block = format_session_summary(
            merged_state, timeline_max=int(b.get("summary_timeline_max") or 12)
        )
        trace["stages"]["assemble"] = {
            "module": "rag_pipeline.dialogue_budget + session_state",
            "history_chars": len(history_block),
            "summary_chars": len(summary_block),
            "context_chars": len(ctx_block),
        }
        return RAGPrepareResult(
            output_direct=None,
            docs=docs_acc,
            prompt_vars={
                "prior_turn_context": prior_turn_context,
                "session_summary": summary_block,
                "history": history_block,
                "context": ctx_block,
                "input": user_input,
            },
            trace=trace,
        )

    if resolved.kind == "ordinal" and (resolved.subject or "").strip():
        subj = resolved.subject.strip()
        dims = info_type_dimensions(extract_profile.info_types)
        seeds = cartesian_seeds([subj], dims)
        if len(seeds) > 1:
            docs_acc, ctx_block = _gather_by_seed_strings(
                seeds,
                user_input=user_input,
                preamble=(
                    "【本轮说明】序数指代已解析到具体实体，已按「实体 × 信息维度」组合检索。\n\n"
                ),
                gather_docs=gather_docs,
                format_docs=format_docs,
                retrieval_query_suffix=retrieval_query_suffix,
            )
            trace["stages"]["retrieve"] = {
                "module": "agent._gather_docs",
                "doc_count": len(docs_acc),
                "retrieval_seed": "ordinal_cartesian",
                "seeds": seeds[:32],
            }
            gate = evaluate_gate(resolved, docs_acc)
            trace["stages"]["gate"] = {
                "module": "rag_pipeline.retrieval_gate",
                "code": gate.code,
                "action": gate.action,
                "meta": gate.meta,
            }
            if gate.action == "respond_direct":
                return RAGPrepareResult(
                    output_direct=gate.user_message,
                    docs=docs_acc,
                    prompt_vars=None,
                    trace=trace,
                )
            b = budget_for_kind(resolved.kind)
            history_block = format_history_block(
                history_messages, query_kind=resolved.kind
            )
            summary_block = format_session_summary(
                merged_state, timeline_max=int(b.get("summary_timeline_max") or 12)
            )
            trace["stages"]["assemble"] = {
                "module": "rag_pipeline.dialogue_budget + session_state",
                "history_chars": len(history_block),
                "summary_chars": len(summary_block),
                "context_chars": len(ctx_block),
            }
            return RAGPrepareResult(
                output_direct=None,
                docs=docs_acc,
                prompt_vars={
                    "prior_turn_context": prior_turn_context,
                    "session_summary": summary_block,
                    "history": history_block,
                    "context": ctx_block,
                    "input": user_input,
                },
                trace=trace,
            )

    seed_llm: str | None = None
    if resolved.kind in ("general", "explicit_entity"):
        seed_llm = retrieval_seed_from_llm_entities(
            resolved,
            user_input,
            inferred_entities,
            list(extract_profile.info_types),
        )
    seed = (
        seed_llm
        if seed_llm is not None
        else retrieval_seed_question(resolved, user_input)
    )
    main_query = retrieval_query_suffix(seed)
    docs = gather_docs(user_input, main_query)
    trace["stages"]["retrieve"] = {
        "module": "agent._gather_docs",
        "doc_count": len(docs),
        "retrieval_seed": seed[:200],
        "info_type": list(extract_profile.info_types),
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
            "prior_turn_context": prior_turn_context,
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
        llm=llm,
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
            turn_context=None,
        )

    t1 = time.perf_counter()
    prompt_val = prompt.invoke(prep.prompt_vars)
    msg = llm.invoke(prompt_val)
    text = StrOutputParser().invoke(msg)
    shown, meta = parse_answer_turn_json(text if isinstance(text, str) else str(text))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    trace["stages"]["generate"] = {
        "module": "langchain_openai.ChatOpenAI",
        "llm_ms": round((time.perf_counter() - t1) * 1000, 2),
    }
    if meta is not None:
        trace["stages"]["turn_context_out"] = meta

    return PipelineResult(
        output=shown,
        docs=prep.docs,
        latency_ms=elapsed_ms,
        trace=trace,
        turn_context=meta,
    )
