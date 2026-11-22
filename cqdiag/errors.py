"""统一错误码与失败清单记录."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_QUANT_WIDTH_INVALID = "E_QUANT_WIDTH_INVALID"
E_GROUP_MISMATCH = "E_GROUP_MISMATCH"
E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
E_BUDGET_EXCEEDED = "E_BUDGET_EXCEEDED"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"


class CqError(Exception):
    """带错误码的业务异常, 进入失败清单而不中断整体运行."""

    def __init__(self, code: str, message: str, context: dict | None = None, **kw):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.context = dict(context or {})
        self.context.update(kw)


@dataclass
class Failure:
    code: str
    message: str
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class FailureList:
    """有序失败清单; 顺序即插入顺序, 保证确定性输出."""

    def __init__(self) -> None:
        self._items: list[Failure] = []

    def add(self, code: str, message: str, **context) -> None:
        self._items.append(Failure(code, message, context))

    def add_error(self, err: CqError) -> None:
        self._items.append(Failure(err.code, err.message, dict(err.context)))

    def extend(self, other: "FailureList") -> None:
        self._items.extend(other._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_list(self) -> list[dict]:
        return [f.to_dict() for f in self._items]
