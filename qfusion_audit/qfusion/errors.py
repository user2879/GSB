"""错误码与失败清单条目。

加载与运行期的非法输入均不抛出中断流程，而是落入 failure list，
由调度/报告层统一汇总，符合"不中断整体加载"的验收要求。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


E_RECORD_INVALID = "E_RECORD_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_RECOMPUTE_BUDGET_PENDING = "E_RECOMPUTE_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


@dataclass
class Failure:
    code: str
    source: str
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
