"""Graceful process signal handling.

A *handleable* termination (SIGTERM/SIGINT) flushes the snapshot, cache and
ledger and emits an exit report.  A forced ``SIGKILL`` cannot be caught by
definition; in that case recovery relies solely on the persisted manifest and no
exit report is promised.
"""

from __future__ import annotations

import signal
from typing import Any, Callable, Dict, List, Optional

from .engine import VADEngine


class GracefulExit(BaseException):
    """Raised after a handleable exit signal is received."""


def install_handlers(engine: VADEngine,
                     on_exit: Optional[Callable[[Dict[str, Any]], None]] = None
                     ) -> List[int]:
    """Install SIGTERM/SIGINT handlers that produce an exit report."""
    installed: List[int] = []

    def handler(signum, _frame):
        try:
            engine._auto_snapshot()
            report = engine.exit_report(
                reason=f"signal_{signum}")
            if on_exit is not None:
                on_exit(report)
        finally:
            raise GracefulExit(f"received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
            installed.append(sig)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            continue
    return installed
