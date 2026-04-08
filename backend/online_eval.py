"""
对话完成后的「线上评估」：异步写 JSONL，不阻塞响应。

环境变量（见 .env.example）：
  ONLINE_EVAL_ENABLED=1           开启
  ONLINE_EVAL_LOG_PATH=...        追加写入的 jsonl 路径
  ONLINE_EVAL_QUICK_SCORE=1       可选：用本地 BGE 算 question/answer 向量余弦相似度（粗 relevancy 代理）
  ONLINE_EVAL_MAX_QUESTION_CHARS  日志里问题最大长度（0=不截断）

说明：默认不做 RAGAS（每条多次 LLM，不适合同步挂在对话后）；离线请用 eval_rag_ragas.py。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def enabled() -> bool:
    return os.getenv("ONLINE_EVAL_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def log_path() -> Path:
    raw = os.getenv("ONLINE_EVAL_LOG_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "eval" / "online_eval.jsonl"


def _truncate(s: str, max_len: int) -> str:
    if max_len <= 0 or len(s) <= max_len:
        return s
    return s[:max_len] + "…"


def _quick_score(question: str, answer: str) -> float | None:
    if os.getenv("ONLINE_EVAL_QUICK_SCORE", "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    if not question.strip() or not answer.strip():
        return None
    try:
        import numpy as np

        from backend import embeddings as emb_mod

        emb = emb_mod.get_embeddings()
        qv = np.asarray(emb.embed_query(question), dtype=np.float64)
        av = np.asarray(emb.embed_query(answer[:2000]), dtype=np.float64)
        nq = np.linalg.norm(qv)
        na = np.linalg.norm(av)
        if nq < 1e-12 or na < 1e-12:
            return None
        return float(np.dot(qv, av) / (nq * na))
    except Exception:
        logger.debug("online_eval quick_score failed", exc_info=True)
        return None


def _build_record(
    *,
    question: str,
    answer: str,
    docs: list,
    latency_ms: float,
    route: str,
) -> dict:
    max_q = int(os.getenv("ONLINE_EVAL_MAX_QUESTION_CHARS", "2000"))
    raw_sources: list[dict] = []
    for d in (docs or [])[:24]:
        md = getattr(d, "metadata", None) or {}
        raw_sources.append(
            {
                "source": (md.get("source") or "")[:500],
                "title": (md.get("title") or "")[:200],
            }
        )
    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for s in raw_sources:
        key = (s["source"], s["title"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(s)
        if len(sources) >= 12:
            break

    rec: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "question_hash": hashlib.sha256(question.encode("utf-8", errors="replace")).hexdigest()[:16],
        "question": _truncate(question, max_q),
        "answer_len": len(answer or ""),
        "answer_preview": (answer or "")[:800],
        "num_docs": len(docs or []),
        "top_sources": sources,
        "latency_ms": round(float(latency_ms), 2),
    }
    qs = _quick_score(question, answer)
    if qs is not None:
        rec["quick_answer_relevancy_proxy"] = round(qs, 4)
    return rec


def append_record_sync(record: dict) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def schedule_log(
    *,
    question: str,
    answer: str,
    docs: list,
    latency_ms: float,
    route: str,
) -> None:
    if not enabled():
        return

    def job() -> None:
        try:
            rec = _build_record(
                question=question,
                answer=answer,
                docs=docs,
                latency_ms=latency_ms,
                route=route,
            )
            append_record_sync(rec)
        except Exception:
            logger.exception("online_eval append failed")

    threading.Thread(target=job, daemon=True).start()
