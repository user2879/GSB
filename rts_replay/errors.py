"""Business error codes and error primitives for the RTS replay workbench."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """All business error codes required by the specification."""

    E_SEQ_GAP = "E_SEQ_GAP"
    E_RESOURCE_OVERFLOW = "E_RESOURCE_OVERFLOW"
    E_QUEUE_BLOCKED = "E_QUEUE_BLOCKED"
    E_RNG_DRIFT = "E_RNG_DRIFT"
    E_BUDGET_EXCEEDED = "E_BUDGET_EXCEEDED"
    E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
    E_RULE_RANGE = "E_RULE_RANGE"
    E_INPUT_MALFORMED = "E_INPUT_MALFORMED"


class RtsError(Exception):
    """Hard error that aborts a processing path.

    Soft per-tick business violations (overflow, blocked queue) are collected as
    :class:`TickError` records instead, so a replay can keep synthesising while
    still reporting the violation with full location information.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        tick: int | None = None,
        actor: str | None = None,
        action: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message
        self.tick = tick
        self.actor = actor
        self.action = action
        self.context = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "tick": self.tick,
            "actor": self.actor,
            "action": self.action,
            "context": self.context,
        }


@dataclass(frozen=True)
class TickError:
    """A non-fatal business violation located at (tick, actor, action)."""

    code: str
    message: str
    tick: int
    actor: str | None = None
    action: str | None = None
    seq: int | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "tick": self.tick,
            "actor": self.actor,
            "action": self.action,
            "seq": self.seq,
            "context": self.context,
        }
