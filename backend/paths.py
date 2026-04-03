"""项目根下的 data/ 为规范目录；兼容旧版在 backend/data/ 下的 raw_pages.jsonl。"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = str(PROJECT_ROOT / "data")
KB_DIR = str(PROJECT_ROOT / "data" / "kb")

CANONICAL_RAW_PAGES_JSONL = PROJECT_ROOT / "data" / "raw_pages.jsonl"
LEGACY_RAW_PAGES_JSONL = PROJECT_ROOT / "backend" / "data" / "raw_pages.jsonl"

# 写入、与 Agent 共用：统一落在项目根 data/（避免 cwd 在 backend 时再生成一份）
RAW_PAGES_JSONL = str(CANONICAL_RAW_PAGES_JSONL)


def _count_nonempty_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def resolve_raw_pages_jsonl_for_read() -> Path:
    """
    --from-raw 默认读哪个文件：
    1) 环境变量 RAW_PAGES_JSONL（若设置）
    2) 否则在「根 data」与「backend/data」两份里，选非空行更多的（解决旧脚本 cwd 在 backend 时写错位置）
    """
    env = os.getenv("RAW_PAGES_JSONL", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    ca, cb = CANONICAL_RAW_PAGES_JSONL, LEGACY_RAW_PAGES_JSONL
    if _count_nonempty_lines(cb) > _count_nonempty_lines(ca):
        return cb
    return ca


def raw_pages_jsonl_for_write() -> Path:
    return CANONICAL_RAW_PAGES_JSONL
