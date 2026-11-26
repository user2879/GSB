from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_INPUT_INVALID = "E_INPUT_INVALID"
E_QUERY_QUOTA_EXCEEDED = "E_QUERY_QUOTA_EXCEEDED"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_VERSION_CONFLICT = "E_VERSION_CONFLICT"
E_SNAPSHOT_RECOVERY = "E_SNAPSHOT_RECOVERY"


@dataclass(frozen=True)
class Failure:
    code: str
    source: str
    line: int | None
    message: str
    item_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "source": self.source,
            "line": self.line,
            "item_id": self.item_id,
            "message": self.message,
        }


class ConfigValidationError(ValueError):
    pass


class VersionConflictError(RuntimeError):
    pass
