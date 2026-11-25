from __future__ import annotations

from typing import Any


class QosDiagError(Exception):
    code = "E_QOS"

    def __init__(self, message: str, *, object_id: str | None = None, field: str | None = None):
        super().__init__(message)
        self.message = message
        self.object_id = object_id
        self.field = field

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": self.code,
            "object_id": self.object_id,
            "field": self.field,
            "message": self.message,
        }


class SpecError(QosDiagError):
    code = "E_SPEC"


class VerifyError(QosDiagError):
    code = "E_VERIFY"

    def __init__(
        self,
        message: str,
        *,
        event_seq: int | None = None,
        pipeline_id: str | None = None,
        stage: str | None = None,
        field: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ):
        super().__init__(message, object_id=pipeline_id, field=field)
        self.event_seq = event_seq
        self.pipeline_id = pipeline_id
        self.stage = stage
        self.expected = expected
        self.actual = actual

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "event_seq": self.event_seq,
                "pipeline_id": self.pipeline_id,
                "stage": self.stage,
                "expected": self.expected,
                "actual": self.actual,
            }
        )
        return data


class ShaperError(QosDiagError):
    code = "E_SHAPER"


class LedgerError(QosDiagError):
    code = "E_LEDGER"
