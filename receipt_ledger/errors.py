"""业务错误码与失败清单条目。

所有错误均收敛为确定性的 :class:`Failure` 记录，进入失败清单而不抛出
到整轮运行之外，保证单条非法输入不会中断流水线。
"""

from __future__ import annotations

from dataclasses import dataclass

E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_WORKER_UNKNOWN = "E_WORKER_UNKNOWN"
E_REQUEST_INVALID = "E_REQUEST_INVALID"
E_DUPLICATE_REQUEST_ID = "E_DUPLICATE_REQUEST_ID"
E_RECEIPT_MALFORMED = "E_RECEIPT_MALFORMED"
E_RECEIPT_TRUNCATED = "E_RECEIPT_TRUNCATED"
E_NON_FINITE_DURATION = "E_NON_FINITE_DURATION"
E_UNKNOWN_PARAM_VERSION = "E_UNKNOWN_PARAM_VERSION"
E_ATTRIBUTION_UNRESOLVED = "E_ATTRIBUTION_UNRESOLVED"
E_VIEW_STALE = "E_VIEW_STALE"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_INJECTION = "E_INJECTION"


@dataclass(frozen=True, slots=True)
class Failure:
    """失败清单条目，字段顺序即规范化排序依据的一部分。"""

    code: str
    source: str
    line: int
    message: str
    injection_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "source": self.source,
            "line": self.line,
            "message": self.message,
            "injection_id": self.injection_id,
        }


def sort_failures(failures: list[Failure]) -> list[Failure]:
    """并发收集后按固定字典序排序，使失败清单与调度顺序无关。"""

    return sorted(
        failures,
        key=lambda item: (
            item.source,
            item.line,
            item.code,
            item.injection_id or "",
            item.message,
        ),
    )
