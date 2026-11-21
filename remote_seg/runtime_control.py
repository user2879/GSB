"""Process-wide graceful stop signal shared by the early entry point and CLI."""

from __future__ import annotations

import signal
import sys
import threading
from typing import Any


STOP_EVENT = threading.Event()


def handle_stop_signal(signum: int, frame: Any) -> None:
    del frame
    STOP_EVENT.set()
    print(
        f"received signal {signum}; finishing in-flight tile tasks and committing snapshot",
        file=sys.stderr,
    )


def install_signal_handlers() -> tuple[Any, Any]:
    return (
        signal.signal(signal.SIGINT, handle_stop_signal),
        signal.signal(signal.SIGTERM, handle_stop_signal),
    )


def restore_signal_handlers(previous: tuple[Any, Any]) -> None:
    signal.signal(signal.SIGINT, previous[0])
    signal.signal(signal.SIGTERM, previous[1])
