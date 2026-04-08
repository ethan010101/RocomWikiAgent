"""
半自动生成 RAG 离线评测集（JSONL，与 eval_rag.py 兼容）。

思路：从 raw_pages.jsonl 按与建库相同的切分得到片段 → 调用 LLM 生成「玩家式提问」+
若干 expected_substrings（强制为标题/正文中的原样子串）→ 自动附加 expected_source_substrings
（来自该页 URL，便于检索命中同源词条）。

用法（项目根目录，需 .env 中 OPENAI_*）：
  python backend/gen_eval_golden.py --count 20 --out data/eval/golden.generated.jsonl
  python backend/gen_eval_golden.py --count 10 --seed 7 --raw data/raw_pages.jsonl

耗时主要来自串行 LLM 请求；默认不在请求间 sleep，遇限流可加 --sleep 0.5。

生成后请人工抽检若干条，再合并进 data/eval/golden.jsonl 使用。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

import os

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import paths as project_paths


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def _source_needles(url: str, title: str) -> list[str]:
    """与 eval 中「URL 子串命中」一致：尽量指向当前词条，避免仅 biligame 过宽。"""
    u = (url or "").strip()
    if not u:
        return []
    needles: list[str] = []
    parsed = urlparse(u)
    q = parsed.query or ""
    if "title=" in q:
        for part in q.split("&"):
            if part.startswith("title="):
                needles.append(part[: min(len(part), 120)])
                break
    path = (parsed.path or "").rstrip("/")
    seg = path.split("/")[-1] if path else ""
    if seg and seg.lower() not in ("rocom", "index.php", ""):
        needles.append(seg)
    if not needles:
        needles.append("biligame.com/rocom")
    seen: set[str] = set()
    out: list[str] = []
    for n in needles:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out[:3]


def _load_pages(raw_path: Path) -> list[dict]:
    pages: list[dict] = []
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            url = (row.get("url") or "").strip()
            if len(text) < 120 or not title:
                continue
            pages.append({"title": title, "url": url, "text": text})
    return pages


def _chunks_for_page(page: dict, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    doc = Document(
        page_content=page["text"],
        metadata={"source": page["url"], "title": page["title"]},
    )
    return splitter.split_documents([doc])


def _verify_substrings_in_text(needles: list[str], title: str, chunk: str) -> list[str]:
    blob = f"{title}\n{chunk}"
    ok: list[str] = []
    for s in needles:
        s = (s or "").strip()
        if len(s) < 2:
            continue
        if s in blob:
            ok.append(s)
    return ok


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "LongCat-Flash-Thinking-2601"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.longcat.chat/openai"),
        temperature=float(os.getenv("GEN_EVAL_TEMPERATURE", "0.35")),
    )


def _gen_one(llm: ChatOpenAI, title: str, url: str, chunk_text: str, max_chars: int) -> dict | None:
    body = chunk_text[:max_chars]
    sys_msg = (
        "你是 BWIKI 评测数据构造助手。你必须只根据给定片段与标题输出合法 JSON，"
        "不要编造片段中不存在的游戏设定。"
    )
    human = (
        f"词条标题：{title}\n"
        f"来源 URL：{url}\n\n"
        f"片段正文：\n---\n{body}\n---\n\n"
        "请完成：\n"
        "1) 用中文写 1 个玩家可能问的具体问题（能从该片段回答或片段明确涉及）。\n"
        "2) 给出 2～4 个 expected_substrings：每个必须是「标题或片段正文」中连续出现的原文字符串，"
        "直接复制，不要改写或缩写；用于检查检索是否命中该片段。\n"
        "3) 可选 ground_truth：用一句话写答案要点（便于人工校对）。\n\n"
        "只输出一个 JSON 对象，不要 markdown 代码块，不要其它说明。键名固定为："
        '{"question":"","expected_substrings":[],"ground_truth":""} '
        "若 ground_truth 不写可设为 null 或空字符串。"
    )
    msg = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=human)])
    raw = getattr(msg, "content", None) or str(msg)
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        return None
    q = (data.get("question") or "").strip()
    subs = data.get("expected_substrings") or []
    if not q or not isinstance(subs, list):
        return None
    subs = [str(x).strip() for x in subs if str(x).strip()]
    gt = data.get("ground_truth")
    gt_s = (gt.strip() if isinstance(gt, str) else "") or None
    return {"question": q, "expected_substrings": subs, "ground_truth": gt_s}


def main() -> None:
    ap = argparse.ArgumentParser(description="半自动生成 eval 用 golden JSONL")
    ap.add_argument("--out", type=Path, required=True, help="输出 JSONL 路径")
    ap.add_argument("--count", type=int, default=20, help="成功写入条数目标")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    ap.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="raw_pages.jsonl；默认与 build_kb --from-raw 相同解析规则",
    )
    ap.add_argument("--max-chars", type=int, default=2200, help="送入模型的单片段最大字符")
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="每次 API 调用前等待秒数；默认 0。若遇 429 可设为 0.3～1",
    )
    ap.add_argument("--max-tries-per-chunk", type=int, default=2, help="单片段 LLM+校验失败重试次数")
    args = ap.parse_args()

    raw_path = args.raw or project_paths.resolve_raw_pages_jsonl_for_read()
    if not raw_path.is_file():
        raise SystemExit(f"找不到 raw_pages：{raw_path}")

    pages = _load_pages(raw_path)
    if not pages:
        raise SystemExit("raw_pages 无有效行（需含 title、url、text）")

    rng = random.Random(args.seed)
    rng.shuffle(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    llm = _build_llm()

    written: list[dict] = []
    attempts = 0
    max_attempts = max(args.count * 80, 400)

    for page in pages:
        if len(written) >= args.count:
            break
        if attempts >= max_attempts:
            break
        chunks = _chunks_for_page(page, splitter)
        if not chunks:
            continue
        ch = rng.choice(chunks)
        chunk_text = ch.page_content or ""
        title = (ch.metadata.get("title") or page["title"]).strip()
        url = (ch.metadata.get("source") or page["url"]).strip()
        if len(chunk_text) < 80:
            continue

        item: dict | None = None
        for _ in range(args.max_tries_per_chunk):
            attempts += 1
            time.sleep(args.sleep)
            gen = _gen_one(llm, title, url, chunk_text, args.max_chars)
            if not gen:
                continue
            verified = _verify_substrings_in_text(gen["expected_substrings"], title, chunk_text)
            if len(verified) < 2:
                continue
            src_needles = _source_needles(url, title)
            item = {
                "id": f"auto_{len(written)+1:04d}",
                "question": gen["question"],
                "expected_substrings": verified[:6],
                "expected_source_substrings": src_needles,
                "ground_truth": gen.get("ground_truth"),
            }
            if not item["ground_truth"]:
                del item["ground_truth"]
            break

        if item:
            written.append(item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in written:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"已写入 {len(written)} 条 -> {args.out}")
    if len(written) < args.count:
        print(
            f"未达目标 {args.count} 条（可能 API/解析失败或子串校验未过）。"
            f"可增大 --count 重试、换 --seed，或检查 OPENAI_*。"
        )
    raise SystemExit(0 if written else 1)


if __name__ == "__main__":
    main()
