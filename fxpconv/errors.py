"""业务错误码与失败清单记录。

所有业务失败都以 :class:`Failure` 形式进入稳定有序的失败清单，
单位置异常不会中断其余位置的计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

E_DESC_INVALID = "E_DESC_INVALID"
E_SHAPE_MISMATCH = "E_SHAPE_MISMATCH"
E_BUDGET_INVALID = "E_BUDGET_INVALID"
E_BATCH_BUDGET_EXHAUSTED = "E_BATCH_BUDGET_EXHAUSTED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_INJECT_NONFINITE_SCALE = "E_INJECT_NONFINITE_SCALE"
E_INJECT_QBITS_OUT_OF_RANGE = "E_INJECT_QBITS_OUT_OF_RANGE"
E_INJECT_DATA_OUT_OF_SHAPE = "E_INJECT_DATA_OUT_OF_SHAPE"
E_INJECT_ILLEGAL_STRATEGY = "E_INJECT_ILLEGAL_STRATEGY"

ALL_BUSINESS_CODES = (
    E_DESC_INVALID,
    E_SHAPE_MISMATCH,
    E_BUDGET_INVALID,
    E_BATCH_BUDGET_EXHAUSTED,
    E_SNAPSHOT_STALE,
    E_INJECT_NONFINITE_SCALE,
    E_INJECT_QBITS_OUT_OF_RANGE,
    E_INJECT_DATA_OUT_OF_SHAPE,
    E_INJECT_ILLEGAL_STRATEGY,
)


@dataclass(frozen=True)
class Failure:
    """失败清单中的单条记录（可 JSON 序列化，顺序稳定）。"""

    code: str
    message: str
    stage: str
    sort_key: tuple
    injection_id: str | None = None
    position: int | None = None
    batch: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "sort_key": list(self.sort_key),
            "injection_id": self.injection_id,
            "position": self.position,
            "batch": self.batch,
        }


@dataclass
class FailureLedger:
    """按稳定排序键去重并有序保存失败记录。"""

    _seen: set[tuple] = field(default_factory=set)
    _items: list[Failure] = field(default_factory=list)

    @staticmethod
    def _sort_probe(key: tuple) -> tuple:
        """把异构排序键归一为可比较的 (类型序号, 值) 序列。"""
        probe = []
        for part in key:
            if isinstance(part, bool):
                probe.append((1, int(part)))
            elif isinstance(part, int):
                probe.append((1, part))
            elif isinstance(part, float):
                probe.append((2, part))
            elif part is None:
                probe.append((0, 0))
            else:
                probe.append((3, str(part)))
        return tuple(probe)

    def add(
        self,
        code: str,
        message: str,
        stage: str,
        sort_key: tuple,
        injection_id: str | None = None,
        position: int | None = None,
        batch: int | None = None,
    ) -> None:
        dedup_key = (code, message, stage, tuple(sort_key), injection_id, position, batch)
        if dedup_key in self._seen:
            return
        self._seen.add(dedup_key)
        self._items.append(
            Failure(
                code=code,
                message=message,
                stage=stage,
                sort_key=tuple(sort_key),
                injection_id=injection_id,
                position=position,
                batch=batch,
            )
        )

    def sort(self) -> None:
        self._items.sort(key=lambda item: self._sort_probe(item.sort_key))

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_list(self) -> list[dict[str, Any]]:
        self.sort()
        return [item.to_dict() for item in self._items]

    def codes(self) -> list[str]:
        self.sort()
        return [item.code for item in self._items]
