from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    E_SCENE = "E_SCENE"
    E_QUOTA = "E_QUOTA"
    E_SNAPSHOT = "E_SNAPSHOT"
    E_REPLAY = "E_REPLAY"
    E_RACE = "E_RACE"
    E_CAPACITY = "E_CAPACITY"
    E_LEASE = "E_LEASE"
    E_CLOCK = "E_CLOCK"
    E_LEDGER = "E_LEDGER"
    E_ENV = "E_ENV"
    E_EXTERNAL_DRIFT = "E_EXTERNAL_DRIFT"


@dataclass
class BusinessError(Exception):
    code: ErrorCode
    message: str
    field_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    event_seq: Optional[int] = None
    source_id: Optional[str] = None

    def __str__(self) -> str:
        parts = [self.code.value, self.message]
        if self.field_path:
            parts.append(f"field={self.field_path}")
        if self.event_seq is not None:
            parts.append(f"seq={self.event_seq}")
        if self.source_id is not None:
            parts.append(f"source={self.source_id}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "field_path": self.field_path,
            "event_seq": self.event_seq,
            "source_id": self.source_id,
            "details": self.details,
        }
