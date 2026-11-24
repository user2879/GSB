from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


CONFIG_INVALID = "E_CONFIG_INVALID"
SAMPLE_INVALID = "E_SAMPLE_INVALID"
BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
BATCH_COMPUTE = "E_BATCH_COMPUTE"
SNAPSHOT_WRITE = "E_SNAPSHOT_WRITE"
SNAPSHOT_RECOVERY = "E_SNAPSHOT_RECOVERY"


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    batch: int | None = None
    seq: int | None = None
    stage: str | None = None
    path: str | None = None
    injection_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def failure_sort_key(item: Failure | dict[str, Any]) -> tuple[Any, ...]:
    data = item if isinstance(item, dict) else item.to_dict()
    return (
        data.get("batch") is None,
        data.get("batch") if data.get("batch") is not None else -1,
        data.get("seq") is None,
        data.get("seq") if data.get("seq") is not None else -1,
        data.get("stage") or "",
        data.get("path") or "",
        data.get("code") or "",
        data.get("injection_id") or "",
        data.get("message") or "",
    )


def sorted_failures(failures: list[Failure | dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [f.to_dict() if isinstance(f, Failure) else dict(f) for f in failures]
    return sorted(rows, key=failure_sort_key)
