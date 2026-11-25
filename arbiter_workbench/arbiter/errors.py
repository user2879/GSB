"""Diagnostic workbench error codes and typed exceptions.

Business codes used across the workbench:

* E_SPEC    : invalid declaration field / object (first offending field reported)
* E_PORT    : loopback port cannot be bound / already in use
* E_TIMEOUT : real socket handshake or read timeout
* E_ORDER   : nondeterministic output across orderings / batch splits
* E_REPLAY  : post-recovery replay mismatch (located to view/key/field)
* E_STATE   : ledger/snapshot state conflict that cannot be reconciled
* E_QUANT   : fixed-point quantization rule violation
* E_INTERNAL: defensive internal invariant failure
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorContext:
    object_id: str = ""
    field: str = ""
    detail: str = ""

    def render(self) -> str:
        parts = [p for p in (self.object_id, self.field, self.detail) if p]
        return " | ".join(parts)


class WorkbenchError(Exception):
    code = "E_INTERNAL"

    def __init__(self, message: str, context: ErrorContext | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext()

    def __str__(self) -> str:
        rendered = self.context.render()
        if rendered:
            return f"{self.code}: {self.message} ({rendered})"
        return f"{self.code}: {self.message}"


class SpecError(WorkbenchError):
    code = "E_SPEC"


class PortError(WorkbenchError):
    code = "E_PORT"


class TimeoutErrorW(WorkbenchError):
    code = "E_TIMEOUT"


class OrderError(WorkbenchError):
    code = "E_ORDER"


class ReplayError(WorkbenchError):
    code = "E_REPLAY"


class StateError(WorkbenchError):
    code = "E_STATE"


class QuantError(WorkbenchError):
    code = "E_QUANT"
