from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ReportableError(Exception):
    code: str
    message: str
    entry_id: str | None = None
    field: str | None = None
    volume_id: str | None = None
    block_index: int | None = None
    expected: Any = None
    actual: Any = None

    def __str__(self) -> str:
        prefix = f"{self.code}: {self.message}"
        location = self.location()
        return f"{prefix} ({location})" if location else prefix

    def location(self) -> str:
        parts: list[str] = []
        if self.entry_id is not None:
            parts.append(f"id={self.entry_id}")
        if self.field is not None:
            parts.append(f"field={self.field}")
        if self.volume_id is not None:
            parts.append(f"volume={self.volume_id}")
        if self.block_index is not None:
            parts.append(f"block={self.block_index}")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code, "message": self.message}
        for key in (
            "entry_id",
            "field",
            "volume_id",
            "block_index",
            "expected",
            "actual",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


class SpecError(ReportableError):
    def __init__(
        self,
        message: str,
        entry_id: str | None = None,
        field: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            code="E_SPEC",
            message=message,
            entry_id=entry_id,
            field=field,
            **kwargs,
        )


class CorruptError(ReportableError):
    def __init__(
        self,
        message: str,
        volume_id: str,
        block_index: int,
        expected: str | None = None,
        actual: str | None = None,
        entry_id: str | None = None,
        field: str = "digest",
    ) -> None:
        super().__init__(
            code="E_CORRUPT",
            message=message,
            entry_id=entry_id,
            field=field,
            volume_id=volume_id,
            block_index=block_index,
            expected=expected,
            actual=actual,
        )


class VerifyError(ReportableError):
    def __init__(
        self,
        message: str,
        volume_id: str | None = None,
        block_index: int | None = None,
        field: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        super().__init__(
            code="E_VERIFY",
            message=message,
            field=field,
            volume_id=volume_id,
            block_index=block_index,
            expected=expected,
            actual=actual,
        )


class LedgerError(ReportableError):
    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(code="E_LEDGER", message=message, field=field)
