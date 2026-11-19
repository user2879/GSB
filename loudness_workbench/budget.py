"""Interaction latency budget with deterministic frame degradation.

The monitor only governs *refresh cadence*. Degraded frames skip the UI
repaint but never touch measurement state, so accumulated loudness and the
final report are unaffected. A pluggable clock keeps the headless demo
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import BUDGET_OVERRUN_LIMIT, UI_REFRESH_BUDGET_MS
from .errors import E_BUDGET_EXCEEDED, WorkbenchError


@dataclass
class BudgetReport:
    blocks_scheduled: int = 0
    frames_painted: int = 0
    frames_degraded: int = 0
    consecutive_overruns: int = 0
    max_elapsed_ms: float = 0.0
    budget_ms: float = UI_REFRESH_BUDGET_MS


class BudgetMonitor:
    def __init__(self, budget_ms: float = UI_REFRESH_BUDGET_MS,
                 overrun_limit: int = BUDGET_OVERRUN_LIMIT,
                 clock: Callable[[], float] | None = None) -> None:
        self.budget_ms = budget_ms
        self.overrun_limit = overrun_limit
        self._clock = clock or _monotonic_ms
        self.report = BudgetReport(budget_ms=budget_ms)
        self._mark: float | None = None

    def mark_start(self) -> None:
        self._mark = self._clock()

    def mark_end(self) -> bool:
        """Return True when the frame is painted, False when degraded/skipped."""
        if self._mark is None:
            raise RuntimeError("mark_end called without mark_start")
        elapsed = self._clock() - self._mark
        self._mark = None
        rep = self.report
        rep.blocks_scheduled += 1
        rep.max_elapsed_ms = max(rep.max_elapsed_ms, elapsed)
        if elapsed > self.budget_ms:
            rep.frames_degraded += 1
            rep.consecutive_overruns += 1
            if rep.consecutive_overruns >= self.overrun_limit:
                raise WorkbenchError(
                    E_BUDGET_EXCEEDED,
                    f"{rep.consecutive_overruns} consecutive overruns "
                    f"(elapsed={elapsed:.2f} ms > budget={self.budget_ms:.2f} ms)",
                )
            return False
        rep.consecutive_overruns = 0
        rep.frames_painted += 1
        return True

    def to_dict(self) -> dict:
        return {
            "blocks_scheduled": self.report.blocks_scheduled,
            "frames_painted": self.report.frames_painted,
            "frames_degraded": self.report.frames_degraded,
            "consecutive_overruns": self.report.consecutive_overruns,
            "max_elapsed_ms": round(self.report.max_elapsed_ms, 6),
            "budget_ms": self.report.budget_ms,
        }


def _monotonic_ms() -> float:
    import time

    return time.monotonic() * 1000.0
