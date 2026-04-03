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


def _gather_docs(user_input: str, retriever, vectorstore: FAISS) -> list:
    """默认纯向量检索；RAG_USE_LEXICAL=1 时叠加标题/正文子串命中（旧英文向量模型兜底）。"""
    terms = list(dict.fromkeys(re.findall(r"[\u4e00-\u9fff]{2,8}", user_input)))
    lex_cap = int(os.getenv("RAG_LEXICAL_CAP", "16"))
    buckets: list = []
    if RAG_USE_LEXICAL:
        buckets.extend(_lexical_hits_from_store(vectorstore, terms, total_cap=lex_cap))
    buckets.extend(retriever.invoke(_retrieval_query(user_input)))
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
            (
                "system",
                (
                    "你是洛克王国 BWIKI 助手。下面「参考资料」来自已索引的 wiki 片段。\n"
                    "规则：\n"
                    "1. 只根据参考资料回答；禁止编造技能名、效果、数值、种族值等。\n"
                    "2. 若资料里没有与用户问题直接相关的内容，必须明确说「根据当前检索到的 wiki 片段未找到相关信息」，"
                    "可简要说明片段里实际提到了什么；不要改用你记忆中的旧版页游数据冒充 wiki。\n"
                    "3. 用中文作答；末尾用列表给出参考链接（去重，来自片段中的来源 URL）。"
                ),
            ),
            ("human", "参考资料：\n{context}\n\n用户问题：{input}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: _format_docs(_gather_docs(x["input"], retriever, vectorstore))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    class _RAGRunner:
        def invoke(self, data: dict) -> dict:
            text = chain.invoke({"input": data.get("input", "")})
            return {"output": text}

    return _RAGRunner()
