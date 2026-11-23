"""错误码与异常定义。

错误码约定：
- E_SAMPLE_INVALID              样本/区域真值/文本描述校验失败（进入失败清单，不致命）；
- E_CONFIG_INVALID              对齐配置/命令流结构性校验失败（致命）；
- E_INTERACT_BUDGET_EXCEEDED    交互延迟预算内无法给出最新完整结果（返回过期摘要）；
- E_SNAPSHOT_STALE              快照版本或配置哈希与当前运行不一致（转入全量重建）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_INTERACT_BUDGET_EXCEEDED = "E_INTERACT_BUDGET_EXCEEDED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


class BenchError(Exception):
    """工作台致命错误（配置/快照不一致等）。"""

    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}


@dataclass
class FailureItem:
    """失败清单条目，顺序稳定、可 JSON 序列化。"""

    code: str
    sample: str
    stage: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "sample": self.sample,
            "stage": self.stage,
            "message": self.message,
            "detail": self.detail,
        }
