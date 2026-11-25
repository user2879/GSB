from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LedgerError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        location = self.details.get("line")
        field_name = self.details.get("field")
        prefix = f"{self.code}: {self.message}"
        if location is not None or field_name:
            prefix += f" (line={location}, field={field_name})"
        return prefix

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class SpecError(LedgerError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_SPEC", message, details)


class FieldError(LedgerError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_FIELD", message, details)


class ReferenceError(LedgerError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_REF", message, details)


class CoverageError(LedgerError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_COVER", message, details)


class LeaseError(LedgerError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_LEASE", message, details)


class ReplayError(LedgerError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_REPLAY", message, details)


class SnapshotError(LedgerError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_SNAPSHOT", message, details)
