import os
import re

from . import hf_setup

hf_setup.init_hf_env()

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from . import embeddings as emb_mod
from .paths import KB_DIR
from .rag_pipeline.orchestrate import prepare_context_rag_turn, run_context_rag_turn
from .rag_pipeline.prompts import HUMAN_TEMPLATE, SYSTEM_RAG

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "16"))
RAG_USE_LEXICAL = os.getenv("RAG_USE_LEXICAL", "false").lower() in ("1", "true", "yes")
RAG_SEARCH_TYPE = os.getenv("RAG_SEARCH_TYPE", "similarity").strip().lower()
RAG_FETCH_K = int(os.getenv("RAG_FETCH_K", "48"))
RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.65"))


def _retrieval_query(user_input: str) -> str:
    """向量检索对短问句不敏感时，拼接领域词提高命中宠物/技能页。"""
    q = user_input.strip()
    extra = os.getenv("RAG_QUERY_SUFFIX", "洛克王国 世界 精灵 宠物 技能 招式 图鉴 BWIKI").strip()
    if not extra:
        return q
    return f"{q}\n{extra}"


def _dedupe_docs(docs, max_docs: int) -> list:
    seen: set[str] = set()
    out = []
    for d in docs:
        key = (d.metadata.get("source") or "") + "\0" + d.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
        if len(out) >= max_docs:
            break
    return out


def _lexical_hits_from_store(vectorstore: FAISS, terms: list[str], total_cap: int = 16) -> list:
    """
    英文向量模型对中文短词相似度弱，对标题/正文头做子串命中，把「喵喵」等拉进上下文。
    """
    dct = getattr(vectorstore.docstore, "_dict", None)
    if not isinstance(dct, dict):
        return []
    terms = [t for t in terms if 2 <= len(t) <= 12]
    if not terms:
        return []
    scored: list[tuple[int, object]] = []
    for doc in dct.values():
        title = (doc.metadata.get("title") or "").strip()
        head = (doc.page_content or "")[:500]
        blob = f"{title}\n{head}"
        rank = 99
        for term in terms:
            if title == term:
                rank = min(rank, 0)
            elif term in title:
                rank = min(rank, 1)
            elif term in head:
                rank = min(rank, 2)
            elif term in blob:
                rank = min(rank, 3)
        if rank < 99:
            scored.append((rank, doc))
    scored.sort(key=lambda x: x[0])
    out = []
    seen: set[str] = set()
    for _, doc in scored:
        key = (doc.metadata.get("source") or "") + "\0" + (doc.page_content or "")[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
        if len(out) >= total_cap:
            break
    return out


def _gather_docs(
    user_input: str,
    retriever,
    vectorstore: FAISS,
    *,
    main_retrieval_query: str | None = None,
) -> list:
    """
    多路检索。main_retrieval_query 由 rag_pipeline 注入（含解析出的主体 + 原问句 + 后缀），
    缺省时退化为仅按当前用户句检索。
    """
    primary = (main_retrieval_query or "").strip() or _retrieval_query(user_input)
    terms = list(dict.fromkeys(re.findall(r"[\u4e00-\u9fff]{2,8}", user_input)))
    lex_cap = int(os.getenv("RAG_LEXICAL_CAP", "16"))
    buckets: list = []
    if RAG_USE_LEXICAL:
        buckets.extend(_lexical_hits_from_store(vectorstore, terms, total_cap=lex_cap))
    buckets.extend(retriever.invoke(primary))
    seen_terms: set[str] = set()
    for term in terms:
        if term == user_input.strip() or term in seen_terms:
            continue
        seen_terms.add(term)
        buckets.extend(retriever.invoke(term))
    cap = min(RAG_TOP_K * 3, 48)
    return _dedupe_docs(buckets, cap)


def _format_docs(docs) -> str:
    blocks = []
    for d in docs:
        src = (d.metadata.get("source") or "").strip()
        title = (d.metadata.get("title") or "").strip()
        head = f"【{title}】({src})" if title else f"【{src}】"
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks) if blocks else "（无检索结果）"


def build_agent():
    """
    返回带 invoke({"input": str}) -> {"output": str} 的对象。
    使用「先检索再生成」，避免工具型 Agent 在部分模型上不调用检索而胡编。
    """
    if not os.path.exists(KB_DIR):
        raise FileNotFoundError("Knowledge base not found. Run: python backend/build_kb.py")

    embeddings = emb_mod.get_embeddings()
    vectorstore = FAISS.load_local(KB_DIR, embeddings, allow_dangerous_deserialization=True)
    if RAG_SEARCH_TYPE == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RAG_TOP_K,
                "fetch_k": max(RAG_FETCH_K, RAG_TOP_K * 2),
                "lambda_mult": RAG_MMR_LAMBDA,
            },
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K})

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "LongCat-Flash-Thinking-2601"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.longcat.chat/openai"),
        temperature=float(os.getenv("RAG_TEMPERATURE", "0.1")),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_RAG),
            ("human", HUMAN_TEMPLATE),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            prior_turn_context=lambda x: "",
            session_summary=lambda x: "（无）",
            history=lambda x: "（无）",
            context=lambda x: _format_docs(
                _gather_docs(x["input"], retriever, vectorstore, main_retrieval_query=None)
            ),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    class _RAGRunner:
        def __init__(self) -> None:
            self._retriever = retriever
            self._vectorstore = vectorstore
            self._prompt = prompt
            self._llm = llm
            self._chain = chain

        def invoke(self, data: dict) -> dict:
            return {"output": self.invoke_with_trace(data)["output"]}

        def invoke_with_trace(self, data: dict) -> dict:
            """
            走 rag_pipeline 分阶段流水线；返回 pipeline 追踪字段便于排查。
            """
            q = data.get("input", "") or ""

            def _gather(ui: str, main_q: str) -> list:
                return _gather_docs(
                    ui,
                    self._retriever,
                    self._vectorstore,
                    main_retrieval_query=main_q,
                )

            result = run_context_rag_turn(
                user_input=q,
                history_messages=data.get("history_messages"),
                context_state=data.get("context_state"),
                gather_docs=_gather,
                format_docs=_format_docs,
                prompt=self._prompt,
                llm=self._llm,
                retrieval_query_suffix=_retrieval_query,
            )
            return {
                "output": result.output,
                "docs": result.docs,
                "latency_ms": result.latency_ms,
                "pipeline": result.trace,
                "turn_context": result.turn_context,
            }

        def retrieve_documents(self, user_input: str) -> list:
            """供离线 RAG 评估：返回与线上一致的检索文档列表（未拼接成 context 字符串）。"""
            return _gather_docs(
                user_input,
                self._retriever,
                self._vectorstore,
                main_retrieval_query=None,
            )

        async def astream_sse_payloads(
            self,
            user_input: str,
            history_messages: list | None = None,
            context_state: dict | None = None,
            eval_capture: dict | None = None,
        ):
            from backend.chat_stream import iter_rag_stream_events

            async for evt in iter_rag_stream_events(
                self,
                user_input,
                history_messages=history_messages,
                context_state=context_state,
                eval_capture=eval_capture,
            ):
                yield evt

    return _RAGRunner()
