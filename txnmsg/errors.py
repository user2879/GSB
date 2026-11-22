"""Business errors shared by the transaction-message ledger."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


@dataclass
class BusinessError(Exception):
    """A deterministic, machine-readable domain failure.

    ``event_seq`` and ``record_seq`` are deliberately separate: an event can
    fail validation while the damaged physical record is a visibility row with
    no independent sequence number.
    """

    code: str
    message: str
    transaction_id: str | None = None
    partition_id: str | None = None
    field: str | None = None
    event_seq: int | None = None
    record_seq: int | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)
    logged: bool = dataclass_field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def exit_code(self) -> int:
        return {
            "E_SPEC": 2,
            "E_REC": 3,
            "E_ORDER": 4,
            "E_SNAPSHOT": 5,
            "E_REPLAY": 6,
        }.get(self.code, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "transaction_id": self.transaction_id,
            "partition_id": self.partition_id,
            "field": self.field,
            "event_seq": self.event_seq,
            "record_seq": self.record_seq,
            "details": self.details,
        }


class SpecError(BusinessError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_SPEC", message, **kwargs)


class RecordError(BusinessError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_REC", message, **kwargs)


class OrderError(BusinessError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_ORDER", message, **kwargs)


class SnapshotError(BusinessError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_SNAPSHOT", message, **kwargs)


class ReplayError(BusinessError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_REPLAY", message, **kwargs)


def error_payload(error: BusinessError) -> dict[str, Any]:
    payload = error.to_dict()
    payload["status"] = "ERROR"
    return payload
