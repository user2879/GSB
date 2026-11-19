from __future__ import annotations

import os
import signal
import threading
import time
from types import FrameType
from typing import Any


class ShutdownController:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.signal_number: int | None = None
        self._held_stages: set[str] = set()
        self._previous_handlers: dict[int, Any] = {}

    @property
    def requested(self) -> bool:
        return self.event.is_set()

    def _handler(self, signum: int, _frame: FrameType | None) -> None:
        self.signal_number = signum
        self.event.set()

    def install(self) -> None:
        signal_numbers = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            signal_numbers.append(signal.SIGBREAK)
        for signum in signal_numbers:
            try:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handler)
            except (OSError, ValueError):
                continue

    def restore(self) -> None:
        for signum, handler in self._previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                continue

    def test_hold(self, stage: str) -> None:
        """Test-only process hold after a stage is written but before commit."""
        target = os.environ.get("PEFT_TEST_HOLD_STAGE")
        if not target or target != stage or stage in self._held_stages:
            return
        self._held_stages.add(stage)
        raw_delay = os.environ.get("PEFT_TEST_HOLD_MS", "800")
        try:
            delay_ms = max(0, int(raw_delay))
        except ValueError:
            delay_ms = 800
        time.sleep(delay_ms / 1000.0)

    def test_hold_after(self, stage: str) -> None:
        """Test-only process hold after a stage has become current."""
        target = os.environ.get("PEFT_TEST_HOLD_AFTER_STAGE")
        if not target or target != stage:
            return
        raw_delay = os.environ.get("PEFT_TEST_HOLD_AFTER_MS", "800")
        try:
            delay_ms = max(0, int(raw_delay))
        except ValueError:
            delay_ms = 800
        time.sleep(delay_ms / 1000.0)

    def signal_name(self) -> str | None:
        if self.signal_number is None:
            return None
        try:
            return signal.Signals(self.signal_number).name
        except ValueError:
            return str(self.signal_number)
