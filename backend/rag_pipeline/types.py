from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

QueryKind = Literal["general", "pronoun", "ordinal", "explicit_entity"]

GateAction = Literal["proceed", "respond_direct"]


@dataclass
class ResolvedQuery:
    """解析后的问句（检索与门控只依赖此结构，不直接扫原始字符串特例）。"""

    kind: QueryKind
    subject: str
    ordinal_index: int
    raw_question: str
    reason: str


@dataclass
class GateResult:
    action: GateAction
    user_message: str
    code: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    output: str
    docs: list
    latency_ms: float
    trace: dict[str, Any]
