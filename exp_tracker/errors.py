"""Business errors and stable error codes exposed by both UI and CLI."""

from __future__ import annotations

from typing import Any


class TrackerError(Exception):
    """Base class for expected business failures."""

    code = "E_STATE"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


class FormatError(TrackerError):
    code = "E_FORMAT"

    def __init__(self, path: str, message: str, **details: Any) -> None:
        details.setdefault("path", path)
        details.setdefault("effective_path", path)
        details.setdefault("reason", message)
        super().__init__(f"E_FORMAT at {path}: {message}", **details)


class RunError(TrackerError):
    code = "E_RUN"

    def __init__(
        self,
        run_id: str,
        chain: list[str],
        reason: str,
        **details: Any,
    ) -> None:
        details.setdefault("run_id", run_id)
        details.setdefault("chain", chain)
        details.setdefault("reason", reason)
        super().__init__(f"E_RUN for {run_id}: {reason}", **details)


class MetricError(TrackerError):
    code = "E_METRIC"

    def __init__(
        self,
        run_id: str,
        step: int,
        field: str,
        reason: str,
        **details: Any,
    ) -> None:
        details.setdefault("run_id", run_id)
        details.setdefault("step", step)
        details.setdefault("field", field)
        details.setdefault("reason", reason)
        super().__init__(
            f"E_METRIC for {run_id} at step {step} field {field}: {reason}",
            **details,
        )


class ReplayError(TrackerError):
    code = "E_REPLAY"

    def __init__(
        self,
        run_id: str,
        version: str,
        basis: str,
        **details: Any,
    ) -> None:
        details.setdefault("run_id", run_id)
        details.setdefault("version", version)
        details.setdefault("basis", basis)
        super().__init__(
            f"E_REPLAY for {run_id} ({version}): {basis}",
            **details,
        )


class StateError(TrackerError):
    code = "E_STATE"
