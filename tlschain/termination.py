from __future__ import annotations

import os
import signal
from dataclasses import dataclass, field
from types import FrameType
from typing import Any


@dataclass
class StopController:
    requested: bool = False
    signal_name: str | None = None
    signal_number: int | None = None
    install_errors: list[str] = field(default_factory=list)

    def handler(self, signum: int, frame: FrameType | None) -> None:
        self.requested = True
        self.signal_number = signum
        try:
            self.signal_name = signal.Signals(signum).name
        except ValueError:
            self.signal_name = str(signum)

    def install(self) -> None:
        candidates = ["SIGINT", "SIGTERM"]
        if hasattr(signal, "SIGBREAK"):
            candidates.append("SIGBREAK")
        for name in candidates:
            value = getattr(signal, name, None)
            if value is None:
                continue
            try:
                signal.signal(value, self.handler)
            except (OSError, ValueError) as exc:
                self.install_errors.append(f"{name}:{exc}")

    def request_graceful_signal(self) -> None:
        signal.raise_signal(signal.SIGINT)


def hard_terminate_self() -> None:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.TerminateProcess(handle, 2)
        os._exit(2)
    os.kill(os.getpid(), signal.SIGKILL)


def termination_support() -> dict[str, Any]:
    return {
        "platform": os.name,
        "hard_mechanism": "TerminateProcess" if os.name == "nt" else "SIGKILL",
        "graceful_signals": [
            name
            for name in ("SIGINT", "SIGTERM", "SIGBREAK")
            if hasattr(signal, name)
        ],
        "hard_exit_report_supported": False,
        "recovery_source": "sqlite-ledger",
    }
