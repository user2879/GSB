"""Business errors shared by the workbench and command line tools."""

from __future__ import annotations

from typing import Any


E_PLAN = "E_PLAN"
E_CYCLE = "E_CYCLE"
E_SNAPSHOT = "E_SNAPSHOT"
E_REPLAY = "E_REPLAY"
E_STAGE = "E_STAGE"
E_EVENT = "E_EVENT"
E_ABORT = "E_ABORT"
E_LEDGER = "E_LEDGER"


class BusinessError(Exception):
    """An expected, machine-readable failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage_seq: int | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage_seq = stage_seq
        self.field_path = field_path
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "stage_seq": self.stage_seq,
            "field_path": self.field_path,
        }
        if self.details:
            result["details"] = self.details
        return result


class PlanValidationError(BusinessError):
    """A complete plan rejection. No part of the plan has been applied."""

    def __init__(
        self,
        message: str,
        *,
        code: str = E_PLAN,
        stage_seq: int | None = None,
        field_path: str | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            code,
            message,
            stage_seq=stage_seq,
            field_path=field_path,
            details={"rejected": True, "reject_code": E_PLAN, "errors": errors or []},
        )
        self.errors = errors or []


class SimulatedCrash(RuntimeError):
    """Used by tests/demos to model a process dying after a committed running state."""

    def __init__(self, session_id: str, stage_seq: int, tail_seq: int | None) -> None:
        super().__init__(f"simulated crash: session={session_id} stage={stage_seq}")
        self.session_id = session_id
        self.stage_seq = stage_seq
        self.tail_seq = tail_seq
