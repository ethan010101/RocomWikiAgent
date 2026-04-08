"""多轮对话持久化：JSON 文件，供前端在清除浏览器缓存后仍能从服务端恢复。"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .paths import PROJECT_ROOT

_DEFAULT_PATH = PROJECT_ROOT / "data" / "conversations.json"
_lock = threading.Lock()


def store_path() -> Path:
    raw = os.getenv("CONVERSATIONS_STORE_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_PATH


def read_state() -> dict:
    """返回 { conversations: list, currentConvId: str|None }。"""
    path = store_path()
    empty = {"conversations": [], "currentConvId": None}
    with _lock:
        if not path.is_file():
            return empty
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return empty
    convs = data.get("conversations")
    if not isinstance(convs, list):
        convs = []
    cid = data.get("currentConvId")
    if cid is not None and not isinstance(cid, str):
        cid = None
    return {"conversations": convs, "currentConvId": cid}


def write_state(conversations: list, current_conv_id: str | None) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "conversations": conversations,
        "currentConvId": current_conv_id,
    }
    tmp = path.with_suffix(".json.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with _lock:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        tmp.replace(path)
