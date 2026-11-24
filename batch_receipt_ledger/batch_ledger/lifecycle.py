from __future__ import annotations

import signal
import threading
from typing import Any, Callable


class GracefulStop:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.received_signals: list[int] = []
        self._previous: dict[int, Any] = {}

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handler)
            except (ValueError, OSError):
                # Non-main threads or platforms without a settable handler.
                pass

    def _handler(self, signum: int, frame: Any) -> None:
        self.received_signals.append(signum)
        self.event.set()

    def request_stop(self) -> None:
        self.event.set()

    @property
    def stop_requested(self) -> bool:
        return self.event.is_set()

    def run_batches(
        self,
        batch_ids: list[str],
        process_batch: Callable[[str], Any],
        commit: Callable[[list[Any]], None],
    ) -> dict[str, Any]:
        completed: list[Any] = []
        completed_ids: list[str] = []
        stopped_after_batch = False
        for batch_id in batch_ids:
            result = process_batch(batch_id)
            completed.append(result)
            completed_ids.append(batch_id)
            # Signals are checked only at batch boundaries, so the current batch finishes.
            if self.stop_requested:
                commit(completed)
                stopped_after_batch = True
                break
        else:
            commit(completed)
        return {
            "completed_ids": completed_ids,
            "completed_count": len(completed_ids),
            "graceful_stop": stopped_after_batch,
            "received_signals": list(self.received_signals),
        }
