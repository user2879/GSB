from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainError(Exception):
    code: str
    message: str
    cert_id: str | None = None
    field_path: str | None = None
    message_seq: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "cert_id": self.cert_id,
            "field": self.field_path,
            "message_seq": self.message_seq,
            "details": self.details,
        }


class DuplicateJsonKey(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(f"duplicate JSON key: {key}")
        self.key = key
