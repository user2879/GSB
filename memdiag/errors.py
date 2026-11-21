"""Typed domain errors used by both the CLI and the desktop UI."""

from __future__ import annotations

from typing import Any


class MemDiagError(Exception):
    code = "E_DOMAIN"
    exit_code = 2

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        payload.update(self.details)
        return payload


class PlanError(MemDiagError):
    code = "E_PLAN"
    exit_code = 2

    def __init__(
        self,
        message: str,
        *,
        layer_id: str = "<plan>",
        field: str | None = None,
        **details: Any,
    ) -> None:
        super().__init__(
            message,
            layer_id=layer_id,
            field=field,
            first_invalid_layer=layer_id,
            **details,
        )


class DependencyError(MemDiagError):
    code = "E_DEPEND"
    exit_code = 3

    def __init__(self, message: str, **details: Any) -> None:
        details.setdefault("referrers", [])
        super().__init__(message, **details)


class OOMError(MemDiagError):
    code = "E_OOM"
    exit_code = 4

    def __init__(
        self,
        message: str,
        *,
        layer_id: str,
        needed_blocks: int,
        free_blocks: int,
        **details: Any,
    ) -> None:
        super().__init__(
            message,
            layer_id=layer_id,
            needed_blocks=needed_blocks,
            free_blocks=free_blocks,
            **details,
        )


class IncrementalError(MemDiagError):
    code = "E_INCR"
    exit_code = 5

    def __init__(
        self,
        message: str,
        *,
        layer_id: str,
        field: str,
        incremental: Any = None,
        full: Any = None,
        **details: Any,
    ) -> None:
        super().__init__(
            message,
            layer_id=layer_id,
            field=field,
            incremental=incremental,
            full=full,
            first_difference_layer=layer_id,
            **details,
        )


class ReplayError(MemDiagError):
    code = "E_REPLAY"
    exit_code = 6

    def __init__(
        self,
        message: str,
        *,
        layer_id: str,
        field: str,
        recorded: Any = None,
        replayed: Any = None,
        **details: Any,
    ) -> None:
        super().__init__(
            message,
            layer_id=layer_id,
            field=field,
            recorded=recorded,
            replayed=replayed,
            first_difference_layer=layer_id,
            **details,
        )


class SnapshotError(MemDiagError):
    code = "E_SNAPSHOT"
    exit_code = 7


class LedgerError(MemDiagError):
    code = "E_LEDGER"
    exit_code = 7


class RecoveryRequired(MemDiagError):
    code = "E_RECOVERY"
    exit_code = 8


class SampleError(MemDiagError):
    code = "E_SAMPLE"
    exit_code = 2
