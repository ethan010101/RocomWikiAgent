"""
离线 RAG 评估：检索命中率 + 可选整链生成检查。

用法（在项目根目录、已配置 .env 且已建 FAISS）：
  python backend/eval_rag.py --dataset data/eval/golden.jsonl
  python backend/eval_rag.py --dataset data/eval/golden.jsonl --with-generation

数据集 JSONL 每行一个对象，字段说明：
  question (必填)                用户问题
  id (可选)                     便于对照日志
  expected_substrings (可选)    列表；每个字符串须出现在至少一条检索片段的标题或正文中
  expected_source_substrings (可选) 列表；每个字符串须出现在至少一条片段的 source URL 中
  must_contain_in_answer (可选)  仅在与 --with-generation 同时使用时检查；答案须包含这些子串
  ground_truth (可选)            本脚本不解析；答案级指标见 backend/eval_rag_ragas.py（RAGAS）
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from backend import hf_setup

hf_setup.init_hf_env()

from backend.agent import build_agent


@dataclass
class ItemResult:
    item_id: str
    question: str
    retrieval_ok: bool
    retrieval_detail: dict = field(default_factory=dict)
    generation_ok: bool | None = None
    answer_preview: str = ""


def _doc_blob(doc) -> str:
    title = (doc.metadata.get("title") or "").strip()
    body = doc.page_content or ""
    return f"{title}\n{body}"


def _check_substrings_in_docs(docs: list, needles: list[str], *, use_source: bool) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for needle in needles:
        if not needle:
            continue
        found = False
        for d in docs:
            if use_source:
                hay = (d.metadata.get("source") or "") + ""
            else:
                hay = _doc_blob(d)
            if needle in hay:
                found = True
                break
        if not found:
            missing.append(needle)
    return (len(missing) == 0, missing)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no} JSON 解析失败: {e}") from e
    return rows


def run_eval(dataset: Path, with_generation: bool, out_json: Path | None) -> int:
    if not dataset.is_file():
        raise SystemExit(f"数据集不存在: {dataset}")

    items = load_jsonl(dataset)
    if not items:
        raise SystemExit("数据集中没有有效行")

    agent = build_agent()
    results: list[ItemResult] = []
    n_ret_ok = 0
    n_ret_labeled = 0
    n_gen_ok = 0
    n_gen_checked = 0
    n_gen_labeled = 0

    for row in items:
        q = (row.get("question") or "").strip()
        if not q:
            continue
        iid = str(row.get("id") or q[:24])
        docs = agent.retrieve_documents(q)

        exp_sub = [x for x in (row.get("expected_substrings") or []) if isinstance(x, str) and x.strip()]
        exp_src = [x for x in (row.get("expected_source_substrings") or []) if isinstance(x, str) and x.strip()]

        ok_text, miss_text = _check_substrings_in_docs(docs, exp_sub, use_source=False)
        ok_src, miss_src = _check_substrings_in_docs(docs, exp_src, use_source=True)
        has_retrieval_labels = bool(exp_sub or exp_src)
        retrieval_ok = (ok_text and ok_src) if has_retrieval_labels else True

        detail = {
            "num_docs": len(docs),
            "missing_in_chunks": miss_text,
            "missing_in_source": miss_src,
            "retrieval_labeled": has_retrieval_labels,
        }
        if has_retrieval_labels:
            n_ret_labeled += 1
            if retrieval_ok:
                n_ret_ok += 1

        gen_ok: bool | None = None
        answer_preview = ""
        must_ans = [x for x in (row.get("must_contain_in_answer") or []) if isinstance(x, str) and x.strip()]

        if with_generation:
            out = agent.invoke({"input": q})
            answer_preview = (out.get("output") or "")[:500]
            n_gen_checked += 1
            if must_ans:
                n_gen_labeled += 1
                gen_ok = all(m in answer_preview for m in must_ans)
                if gen_ok:
                    n_gen_ok += 1
            else:
                gen_ok = None  # 未标注则不判生成

        results.append(
            ItemResult(
                item_id=iid,
                question=q,
                retrieval_ok=retrieval_ok,
                retrieval_detail=detail,
                generation_ok=gen_ok,
                answer_preview=answer_preview,
            )
        )

    total = len(results)
    summary = {
        "dataset": str(dataset),
        "items": total,
        "retrieval_labeled_items": n_ret_labeled,
        "retrieval_pass": n_ret_ok,
        "retrieval_rate_on_labeled": round(n_ret_ok / n_ret_labeled, 4) if n_ret_labeled else None,
        "with_generation": with_generation,
        "generation_runs": n_gen_checked,
        "generation_labeled_items": n_gen_labeled,
        "generation_pass_on_must_contain": n_gen_ok,
        "generation_rate_on_labeled": round(n_gen_ok / n_gen_labeled, 4) if n_gen_labeled else None,
    }

    for r in results:
        labeled = r.retrieval_detail.get("retrieval_labeled")
        if labeled:
            mark = "OK " if r.retrieval_ok else "FAIL"
        else:
            mark = "SKIP"
        print(f"[检索 {mark}] {r.item_id}  docs={r.retrieval_detail['num_docs']}")
        if r.retrieval_detail.get("retrieval_labeled") and not r.retrieval_ok:
            if r.retrieval_detail["missing_in_chunks"]:
                print(f"    正文中未出现: {r.retrieval_detail['missing_in_chunks']}")
            if r.retrieval_detail["missing_in_source"]:
                print(f"    URL 中未出现: {r.retrieval_detail['missing_in_source']}")
        if with_generation and r.generation_ok is not None:
            gm = "OK " if r.generation_ok else "FAIL"
            print(f"  [生成 {gm}] 预览: {r.answer_preview[:120]}...")

    print("---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if out_json:
        payload = {
            "summary": summary,
            "results": [
                {
                    "id": r.item_id,
                    "question": r.question,
                    "retrieval_ok": r.retrieval_ok,
                    "retrieval_detail": r.retrieval_detail,
                    "generation_ok": r.generation_ok,
                    "answer_preview": r.answer_preview,
                }
                for r in results
            ],
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"已写入: {out_json}")

    if n_ret_labeled == 0:
        return 0
    return 0 if n_ret_ok == n_ret_labeled else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 离线评估（检索 + 可选生成）")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=_project_root / "data" / "eval" / "golden.jsonl",
        help="JSONL 评测集路径（默认 data/eval/golden.jsonl）",
    )
    ap.add_argument(
        "--with-generation",
        action="store_true",
        help="调用完整 RAG 链生成答案（消耗 API）；并检查 must_contain_in_answer",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="将明细写入 JSON 文件",
    )
    args = ap.parse_args()
    raise SystemExit(run_eval(args.dataset, args.with_generation, args.out))


if __name__ == "__main__":
    main()
