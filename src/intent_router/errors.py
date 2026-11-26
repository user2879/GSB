"""Business error codes (E_*) and structured failure records."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Loading / validation failures
E_DUPLICATE_TEXT_ID = "E_DUPLICATE_TEXT_ID"
E_TOKEN_OUT_OF_VOCAB = "E_TOKEN_OUT_OF_VOCAB"
E_EMPTY_TEXT = "E_EMPTY_TEXT"
E_UNKNOWN_CHANNEL = "E_UNKNOWN_CHANNEL"
E_BAD_REVISION_PARENT = "E_BAD_REVISION_PARENT"
E_REVISION_CYCLE = "E_REVISION_CYCLE"
E_DUPLICATE_REVISION_ID = "E_DUPLICATE_REVISION_ID"
E_BAD_WEIGHT = "E_BAD_WEIGHT"
E_BAD_FRACTION_BITS = "E_BAD_FRACTION_BITS"
E_BAD_GEAR = "E_BAD_GEAR"
E_BAD_CONCURRENCY = "E_BAD_CONCURRENCY"
E_BAD_BUDGET = "E_BAD_BUDGET"
E_BAD_THRESHOLD = "E_BAD_THRESHOLD"
E_UNKNOWN_INTENT = "E_UNKNOWN_INTENT"
E_BAD_COMMAND = "E_BAD_COMMAND"
E_BAD_SNAPSHOT = "E_BAD_SNAPSHOT"

# Runtime / scheduling failures
E_TEXT_BUDGET_EXCEEDED = "E_TEXT_BUDGET_EXCEEDED"
E_CHANNEL_QUOTA_EXCEEDED = "E_CHANNEL_QUOTA_EXCEEDED"


@dataclass
class Failure:
    """A non-fatal business failure; valid items keep processing."""

    code: str
    message: str
    entity_type: str = ""
    entity_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Failure":
        return cls(
            code=data["code"],
            message=data["message"],
            entity_type=data.get("entity_type", ""),
            entity_id=data.get("entity_id", ""),
            detail=dict(data.get("detail", {})),
        )


class FailureCollector:
    """Collects E_ failures without raising, preserving insertion order."""

    def __init__(self) -> None:
        self._items: list[Failure] = []

    def add(
        self,
        code: str,
        message: str,
        entity_type: str = "",
        entity_id: str = "",
        **detail: Any,
    ) -> Failure:
        failure = Failure(code, message, entity_type, entity_id, dict(detail))
        self._items.append(failure)
        return failure

    def extend(self, failures: list[Failure]) -> None:
        self._items.extend(failures)

    @property
    def items(self) -> list[Failure]:
        return list(self._items)

    def count_by_code(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for failure in self._items:
            counts[failure.code] = counts.get(failure.code, 0) + 1
        return dict(sorted(counts.items()))

    def __len__(self) -> int:
        return len(self._items)

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]
