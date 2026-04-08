"""
使用 RAGAS 对「检索上下文 + RAG 答案」做离线评测：faithfulness、answer_relevancy。

faithfulness：答案中的陈述是否被 retrieved contexts 支持。
answer_relevancy：答案与问题的相关性（含反向提问 + 与原文问题的向量相似度）。

依赖：pip install -r requirements.txt（含 ragas、datasets、pandas）

用法（项目根目录，需 .env、data/kb/）：
  python backend/eval_rag_ragas.py --dataset data/eval/golden.generated.jsonl --limit 5
  python backend/eval_rag_ragas.py --dataset data/eval/golden.generated.jsonl --out data/eval/ragas_last.json

注意：每条样本会触发多次评测侧 LLM 调用 + 答案 relevancy 的本地向量计算，成本与耗时远高于纯检索 eval。

兼容仅支持 chat.completions n=1 的网关（如部分 LongCat 路由）：answer_relevancy 默认 strictness=1，
可通过环境变量 RAGAS_ANSWER_RELEVANCY_STRICTNESS 调整（>1 可能再次触发 400）。

ragas.evaluate() 只接受 ragas.metrics.base.Metric 子类（旧版 faithfulness / AnswerRelevancy），
与 metrics.collections 下的 Faithfulness、AnswerRelevancy 不兼容，故本脚本统一走旧版指标 + LangChain ChatOpenAI + 本地 BGE。
评测模型建议 RAGAS_OPENAI_MODEL_NAME=deepseek-chat（与对话用 reasoner 区分）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from backend import hf_setup

hf_setup.init_hf_env()

warnings.filterwarnings(
    "ignore",
    message=".*evaluate\\(\\) is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*ragas.metrics' is deprecated.*",
    category=DeprecationWarning,
)


def _chunk_text(doc) -> str:
    title = (doc.metadata.get("title") or "").strip()
    body = doc.page_content or ""
    return f"{title}\n{body}"


def _load_jsonl(path: Path) -> list[dict]:
    from backend.eval_rag import load_jsonl

    return load_jsonl(path)


def _build_eval_metrics():
    """
    必须使用 ragas.metrics.base.Metric 子类；ragas.metrics.collections.* 与 evaluate() 不兼容。
    """
    strictness = max(1, int(os.getenv("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "1")))
    from ragas.metrics import faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy

    return [faithfulness, AnswerRelevancy(strictness=strictness)]


def main() -> None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig
    except ImportError as e:
        raise SystemExit(
            "缺少依赖，请在项目根执行: pip install ragas datasets pandas\n"
            f"原始错误: {e}"
        ) from e

    ap = argparse.ArgumentParser(description="RAGAS：faithfulness + answer_relevancy")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=_project_root / "data" / "eval" / "golden.jsonl",
        help="JSONL（需含 question；ground_truth 仅写入导出，RAGAS 本脚本不强制）",
    )
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条；0 表示全部")
    ap.add_argument("--max-contexts", type=int, default=12, help="每条送入 RAGAS 的检索片段条数上限")
    ap.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1500,
        help="单条 context 截断长度，控制 faithfulness 的 token",
    )
    ap.add_argument("--out", type=Path, default=None, help="导出合并明细 JSON（utf-8）")
    ap.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("RAGAS_EVAL_TIMEOUT", "180")),
        help="单指标调用超时（秒），可用环境变量 RAGAS_EVAL_TIMEOUT",
    )
    args = ap.parse_args()

    if not args.dataset.is_file():
        raise SystemExit(f"数据集不存在: {args.dataset}")

    from backend import embeddings as emb_mod
    from backend.agent import build_agent

    rows_in = _load_jsonl(args.dataset)
    if args.limit and args.limit > 0:
        rows_in = rows_in[: args.limit]

    agent = build_agent()
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    meta_ids: list[str] = []

    for row in rows_in:
        q = (row.get("question") or "").strip()
        if not q:
            continue
        iid = str(row.get("id") or "") or f"row_{len(questions)}"
        print(f"[生成答案与上下文] {iid} …", flush=True)
        docs = agent.retrieve_documents(q)
        ctx = []
        for d in docs[: args.max_contexts]:
            t = _chunk_text(d)[: args.max_chunk_chars]
            if t.strip():
                ctx.append(t)
        if not ctx:
            ctx = ["（无检索结果）"]
        out = agent.invoke({"input": q})
        ans = (out.get("output") or "").strip()
        if not ans:
            ans = "（空回答）"

        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
        meta_ids.append(iid)

    if not questions:
        raise SystemExit("没有可评测的问题行")

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )

    from langchain_openai import ChatOpenAI

    lc_embeddings = emb_mod.get_embeddings()
    run_config = RunConfig(timeout=args.timeout)
    metrics = _build_eval_metrics()
    _bu = (os.getenv("OPENAI_API_BASE") or "").strip()
    ev_llm = ChatOpenAI(
        model=os.getenv("RAGAS_OPENAI_MODEL_NAME")
        or os.getenv("OPENAI_MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=_bu or None,
        temperature=0.0,
        timeout=float(args.timeout),
    )
    ev_emb = lc_embeddings

    print("[RAGAS] 运行 faithfulness + answer_relevancy …", flush=True)
    result = evaluate(
        ds,
        metrics=metrics,
        llm=ev_llm,
        embeddings=ev_emb,
        run_config=run_config,
        raise_exceptions=False,
    )

    print(result)

    if args.out:
        try:
            pdf = result.to_pandas()
        except Exception as e:
            print(f"导出明细失败（可忽略）: {e}", flush=True)
        else:
            export = pdf.to_dict(orient="records")
            for i, rec in enumerate(export):
                rec["eval_id"] = meta_ids[i] if i < len(meta_ids) else None
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary_repr": result.__repr__(),
                        "rows": export,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"已写入: {args.out}", flush=True)


if __name__ == "__main__":
    main()
