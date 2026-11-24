from __future__ import annotations

import signal
from types import FrameType
from typing import Any


class SignalInterrupt(RuntimeError):
    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum} before snapshot commit boundary")
        self.signum = signum


class SignalGate:
    """Install signal handlers that only interrupt at deterministic boundaries."""

    def __init__(self) -> None:
        self.stop_requested = False
        self.signals: list[int] = []
        self._previous: dict[int, Any] = {}

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)

            def handler(received: int, _frame: FrameType | None, gate: "SignalGate" = self) -> None:
                gate.stop_requested = True
                gate.signals.append(received)
                raise SignalInterrupt(received)

            signal.signal(signum, handler)

    def restore(self) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)

    def raise_if_stopped(self) -> None:
        if self.stop_requested and self.signals:
            raise SignalInterrupt(self.signals[-1])
