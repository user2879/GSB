"""Single-writer command coordination and latency-budget response semantics."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

from .codes import Code
from .engine import IncrementalEngine
from .snapshot import SnapshotStore


class CommandCoordinator:
    def __init__(
        self,
        engine: IncrementalEngine,
        store: SnapshotStore | None = None,
        budget_ms: int | None = None,
        work_hook: Callable[[IncrementalEngine, dict[str, Any]], None] | None = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.budget_ms = budget_ms if budget_ms is not None else engine.config.interaction_budget_ms
        self.work_hook = work_hook
        self.lock = threading.RLock()
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.last_summary: dict[str, Any] | None = engine.summary()
        self._job_sequence = 0
        self._latest_job: dict[str, Any] | None = None
        self._thread = threading.Thread(target=self._worker, name="ner-command-writer", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            try:
                with self.lock:
                    command_result = self.engine.apply_command(job["command"])
                    if self.work_hook is not None:
                        self.work_hook(self.engine, job["command"])
                    self.engine.run_incremental()
                    snapshot = self.store.save(self.engine) if self.store is not None else None
                    summary = self.engine.summary()
                    self.last_summary = summary
                    job["result"] = {"command": command_result, "snapshot": snapshot, "summary": summary}
            except Exception as exc:  # pragma: no cover - defensive boundary
                job["result"] = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                job["event"].set()
                self.queue.task_done()

    def submit(self, command: dict[str, Any], budget_ms: int | None = None) -> dict[str, Any]:
        self._job_sequence += 1
        target_generation = self.engine.generation + 1
        job = {"id": self._job_sequence, "command": command, "event": threading.Event(), "result": None}
        self._latest_job = job
        self.queue.put(job)
        budget = self.budget_ms if budget_ms is None else budget_ms
        completed = job["event"].wait(max(0.0, budget / 1000.0))

        if completed:
            result = job["result"]
            return {
                "status": "fresh",
                "code": Code.OK,
                "job_id": job["id"],
                "target_generation": target_generation,
                **result,
            }

        previous = self.last_summary
        if previous is None:
            return {
                "status": "pending",
                "code": Code.SUMMARY_PENDING,
                "job_id": job["id"],
                "target_generation": target_generation,
                "latest_generation": 0,
                "stale_generation": None,
                "summary": None,
                "diagnosis": "预算内没有任何已完成摘要",
            }
        stale_summary = dict(previous)
        stale_generation = stale_summary.get("generation")
        stale_summary["stale"] = True
        stale_summary["code"] = Code.STALE_SUMMARY
        return {
            "status": "stale",
            "code": Code.STALE_SUMMARY,
            "job_id": job["id"],
            "target_generation": target_generation,
            "latest_generation": target_generation,
            "stale_generation": stale_generation,
            "generation_gap": target_generation - int(stale_generation or 0),
            "summary": stale_summary,
            "diagnosis": "交互预算内未完成最新代次；返回最近已完成摘要，稍后以权威账本替换",
        }

    def wait_for_latest(self, timeout: float = 10.0) -> dict[str, Any]:
        job = self._latest_job
        if job is None:
            return {"status": "idle", "summary": self.last_summary}
        job["event"].wait(timeout)
        return {"status": "fresh" if job["event"].is_set() else "timeout", **(job.get("result") or {})}

    def shutdown(self) -> None:
        self.queue.put(None)
        self._thread.join(timeout=2.0)


def run_with_deadline(engine: IncrementalEngine, command: dict[str, Any], budget_ms: int, work_ms: int = 0) -> dict[str, Any]:
    """Convenience synchronous API used by CLI and deterministic demos."""
    started = time.perf_counter()
    if work_ms:
        time.sleep(work_ms / 1000.0)
    result = engine.apply_command(command)
    engine.run_incremental()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if elapsed_ms <= budget_ms:
        return {"status": "fresh", "code": Code.OK, "elapsed_ms": elapsed_ms, "command": result, "summary": engine.summary()}
    stale = engine.summary(stale=True, code=Code.STALE_SUMMARY)
    return {
        "status": "stale",
        "code": Code.STALE_SUMMARY,
        "elapsed_ms": elapsed_ms,
        "budget_ms": budget_ms,
        "stale_generation": stale["generation"],
        "summary": stale,
        "command": result,
    }
