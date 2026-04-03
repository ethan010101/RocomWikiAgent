import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import hf_setup

hf_setup.init_hf_env()

import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import embeddings as emb_mod
import paths as project_paths
import wiki_sources

DATA_DIR = project_paths.DATA_DIR
KB_DIR = project_paths.KB_DIR
RAW_WRITE = str(project_paths.raw_pages_jsonl_for_write())
REQUEST_TIMEOUT = int(os.getenv("WIKI_REQUEST_TIMEOUT", "30"))
MIN_TEXT_LEN = int(os.getenv("WIKI_MIN_TEXT_LEN", "60"))
WIKI_MODE = os.getenv("WIKI_MODE", "all").strip().lower()
WIKI_MAX_PAGES = int(os.getenv("WIKI_MAX_PAGES", "0"))
PARSE_DELAY = float(os.getenv("WIKI_PARSE_DELAY", "0.05"))


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    content = soup.select_one("#mw-content-text") or soup.body
    text = content.get_text("\n", strip=True) if content else soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_text_via_parse(session: requests.Session, title: str):
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
    }
    try:
        r = session.get(wiki_sources.WIKI_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            return None
        html = data.get("parse", {}).get("text", {}).get("*")
        if not html:
            return None
        return extract_text(html)
    except Exception:
        return None


def fetch_text_via_url(session: requests.Session, url: str):
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            return None
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding or "utf-8"
        return extract_text(r.text)
    except Exception:
        return None


def collect_pages_from_api() -> list[dict]:
    """与 import_lkwiki / import_pets 一致：先列标题，再取正文。"""
    session = wiki_sources.wiki_http_session()
    meta = wiki_sources.list_pages_for_kb(WIKI_MODE, session)
    if WIKI_MAX_PAGES > 0:
        meta = meta[:WIKI_MAX_PAGES]

    results: list[dict] = []
    total = len(meta)
    for i, p in enumerate(meta, 1):
        title = p["title"]
        pref_url = p.get("url") or wiki_sources.title_to_article_url(title)
        text = fetch_text_via_parse(session, title) or ""
        if len(text) < MIN_TEXT_LEN:
            for u in wiki_sources.title_fetch_url_candidates(title, pref_url):
                text = fetch_text_via_url(session, u) or ""
                if len(text) >= MIN_TEXT_LEN:
                    break
        if len(text) >= MIN_TEXT_LEN:
            results.append({"url": pref_url, "title": title, "text": text})
        time.sleep(PARSE_DELAY)
        if i % 50 == 0 or i == total:
            print(f"  已处理 {i}/{total} 个标题，有效正文 {len(results)} 页")
    return results


def build_vector_store(pages: list[dict]) -> int:
    if not pages:
        raise ValueError("No pages to index; check API、WIKI_MODE 或降低 WIKI_MIN_TEXT_LEN。")
    documents = [
        Document(page_content=p["text"], metadata={"source": p["url"], "title": p.get("title", "")})
        for p in pages
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks after split.")

    embeddings = emb_mod.get_embeddings()
    print(f"  向量化模型: {emb_mod.embedding_model_label()}")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(KB_DIR, exist_ok=True)
    vectorstore.save_local(KB_DIR)
    return len(chunks)


def _count_nonempty_lines(path: str) -> int:
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def save_raw_pages(pages: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.isfile(RAW_WRITE):
        old_n = _count_nonempty_lines(RAW_WRITE)
        if old_n > len(pages):
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            bak = f"{RAW_WRITE}.{stamp}.bak"
            shutil.copy2(RAW_WRITE, bak)
            print(
                f"注意: 原 raw_pages.jsonl 有 {old_n} 行，本次只写入 {len(pages)} 条，"
                f"已自动备份到:\n  {bak}"
            )
    with open(RAW_WRITE, "w", encoding="utf-8") as f:
        for p in pages:
            row = {"url": p["url"], "title": p.get("title"), "text": p["text"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pages_from_raw_jsonl(path: str) -> tuple[list[dict], dict]:
    """从已有 raw_pages.jsonl 读入，用于只重建向量库、不重新爬 wiki。返回 (列表, 统计)。"""
    stats = {
        "nonempty_lines": 0,
        "json_error": 0,
        "no_url": 0,
        "short_text": 0,
        "kept": 0,
    }
    if not os.path.isfile(path):
        return [], stats
    out: list[dict] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["nonempty_lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["json_error"] += 1
                continue
            url = (row.get("url") or "").strip()
            text = (row.get("text") or row.get("content") or "").strip()
            if not url:
                stats["no_url"] += 1
                continue
            if len(text) < MIN_TEXT_LEN:
                stats["short_text"] += 1
                continue
            stats["kept"] += 1
            out.append(
                {
                    "url": url,
                    "title": row.get("title"),
                    "text": text,
                }
            )
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="构建洛克 Wiki 知识库（爬取或从 jsonl 重建向量）")
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="只根据已有 JSONL 重建 FAISS，不爬取、不覆盖 jsonl；未指定 --raw-file 时自动在 "
        "data/raw_pages.jsonl 与 backend/data/raw_pages.jsonl 中选非空行更多的一份",
    )
    parser.add_argument(
        "--raw-file",
        default=None,
        help="指定 jsonl；不设则自动选择（也可用环境变量 RAW_PAGES_JSONL）",
    )
    args = parser.parse_args()

    if args.from_raw:
        raw_read = args.raw_file or str(project_paths.resolve_raw_pages_jsonl_for_read())
        if args.raw_file is None:
            print(f"[提示] 未指定 --raw-file，自动读取: {raw_read}")
        print(f"[1/1] 从 {raw_read} 读取（跳过爬取与覆盖 jsonl）")
        pages, st = load_pages_from_raw_jsonl(raw_read)
        print(
            f"文件非空行数: {st['nonempty_lines']} | "
            f"JSON 解析失败: {st['json_error']} | 无 url: {st['no_url']} | "
            f"正文短于 {MIN_TEXT_LEN}: {st['short_text']} | 可用: {st['kept']}"
        )
        if st["nonempty_lines"] > 0 and st["kept"] < st["nonempty_lines"]:
            print(
                "若你预期应有上千条，多半是：当前磁盘上的 jsonl 已被一次「小爬取」覆盖成少量行；"
                "请从备份恢复，或重新运行不带 --from-raw 的完整 build_kb。"
            )
        if not pages:
            print("没有可用数据。请确认路径正确，或调低 WIKI_MIN_TEXT_LEN。")
            sys.exit(1)
        print("[2/2] Building FAISS knowledge base")
        chunk_count = build_vector_store(pages)
        print(f"Done. Total chunks: {chunk_count}")
        print(f"KB path: {KB_DIR}")
        return

    print(f"[1/3] MediaWiki API 拉取列表（模式 WIKI_MODE={WIKI_MODE}，base={wiki_sources.WIKI_BASE_URL}）")
    if WIKI_MAX_PAGES > 0:
        print(f"     最多处理标题数: {WIKI_MAX_PAGES}")
    pages = collect_pages_from_api()
    print(f"Collected pages: {len(pages)}")
    if not pages:
        print("没有可用正文。可尝试: WIKI_MODE=all|pets，或调低 WIKI_MIN_TEXT_LEN。")
        sys.exit(1)

    print("[2/3] Saving raw pages")
    save_raw_pages(pages)

    print("[3/3] Building FAISS knowledge base")
    chunk_count = build_vector_store(pages)
    print(f"Done. Total chunks: {chunk_count}")
    print(f"KB path: {KB_DIR}")


if __name__ == "__main__":
    main()
