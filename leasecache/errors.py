from __future__ import annotations

from typing import Any


class LeaseError(RuntimeError):
    """A stable, machine-readable component failure.

    ``field`` always identifies the first offending logical field when one is
    known.  Verification failures additionally carry ``artifact_key`` and
    ``node_id`` so that callers can locate the first divergent record.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        expected: Any = None,
        actual: Any = None,
        artifact_key: str | None = None,
        node_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.expected = expected
        self.actual = actual
        self.artifact_key = artifact_key
        self.node_id = node_id
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        for name in ("field", "expected", "actual", "artifact_key", "node_id"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.details:
            result["details"] = self.details
        return result

    def log_fields(self) -> dict[str, Any]:
        fields = self.to_dict()
        fields.pop("ok", None)
        fields.pop("code", None)
        fields.pop("message", None)
        return fields
