"""Interactive session with latency-budget responses and graceful shutdown."""
from __future__ import annotations

import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import apply_edit


class InteractiveSession:
    def __init__(self, input_dir: str | Path, config: dict, work_dir: str | Path):
        from .runner import PipelineRunner

        self.config = config
        self.runner = PipelineRunner(input_dir, config, work_dir)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = None
        self._lock = threading.Lock()
        self._shutdown = False
        self._install_signals()

    def _install_signals(self) -> None:
        def handler(signum, frame):
            # Finish the in-flight frame-stage atomic commit, then leave the
            # process with an explicit success status.
            self.request_shutdown()
            try:
                self.graceful_shutdown()
            finally:
                os._exit(0)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except ValueError:
                # Signals cannot be installed off the main thread (e.g. tests).
                pass

    def request_shutdown(self) -> None:
        with self._lock:
            self._shutdown = True

    def start(self):
        self.runner.reload_inputs()
        with self._lock:
            self._pending = self._executor.submit(self.runner.run_full)
        return self._pending

    def _submit(self, func):
        future = self._executor.submit(func)
        with self._lock:
            self._pending = future
        return future

    def dispatch(self, command: dict) -> dict:
        if command["type"] == "shutdown":
            return self.graceful_shutdown()

        budget = float(self.config["interactive_budget_ms"]) / 1000.0
        with self._lock:
            pending = self._pending
        if pending is not None and not pending.done():
            return self._pending_response(command)

        start = time.perf_counter()
        future = self._submit(lambda: self.runner.run_incremental(command))
        if future.done():
            report = future.result()
            return {
                "code": "OK",
                "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
                "summary": self.runner.latest_summary(),
                "report": future.result(),
                "affected_frames": self.runner._affected_frames(command),
            }
        # Wait only up to the latency budget before returning a pending state.
        if not future.done():
            time.sleep(budget)
        if future.done():
            report = future.result()
            return {
                "code": "OK",
                "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
                "summary": self.runner.latest_summary(),
                "report": report,
                "affected_frames": self.runner._affected_frames(command),
            }
        return self._pending_response(command)

    def _pending_response(self, command: dict) -> dict:
        summary = self.runner.latest_summary()
        return {
            "code": "E_INTERACT_BUDGET_PENDING",
            "budget_ms": self.config["interactive_budget_ms"],
            "latest_completed_summary": summary,
            "affected_frames": self.runner._affected_frames(command),
            "progress": 0.5 if summary else 0.0,
        }

    def collect(self):
        with self._lock:
            pending = self._pending
        if pending is not None:
            return pending.result()
        return None

    def graceful_shutdown(self) -> dict:
        report = self.collect()
        # The current frame-stage artifacts have already been atomically
        # committed by the snapshot store before the future completed.
        self.runner.shutdown()
        self._executor.shutdown(wait=True)
        return {"code": "SHUTDOWN", "exit_code": 0,
                "report": report,
                "manifest": self.runner.snapshot_store.recover()}

    def run_command_stream(self, commands: list[dict]) -> list[dict]:
        responses = []
        self.start()
        self.collect()
        responses.append({"seq": -1, "code": "OK",
                          "summary": self.runner.latest_summary()})
        for command in commands:
            if command["type"] == "shutdown":
                responses.append({"seq": command.get("seq"),
                                  **self.graceful_shutdown()})
                break
            response = self.dispatch(command)
            responses.append({"seq": command.get("seq"), **response})
            # Background recomputation must finish before the next edit so the
            # cache ledger stays deterministic; dispatch itself stays bounded.
            self.collect()
            responses[-1]["completed_report_version"] = self.runner.version
        return responses
