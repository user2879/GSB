"""Serialized command worker with latency-budgeted interactive responses."""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import copy

from .cache import ArtifactCache
from .config import RenderConfig
from .errors import E_INTERACT_BUDGET_PENDING, RenderError
from .renderer import DeterministicRenderer, RenderResult
from .state import (
    CommandEffect,
    RenderState,
    apply_command,
    effective_config_fingerprint,
    touch_rejected_command,
)


@dataclass
class _Job:
    index: int
    command: dict[str, Any]
    submitted_at: float
    done: bool = False
    done_at: float | None = None
    progress: dict[str, Any] | None = None
    result: RenderResult | None = None
    effect: CommandEffect | None = None
    error: RenderError | None = None
    status: str = "queued"
    response_latency_ms: float | None = None
    applied_state: RenderState | None = None
    initial_pending: bool = False
    preview_summary: dict[str, Any] | None = None
    done_event: threading.Event = field(default_factory=threading.Event)


class InteractiveEngine:
    """Runs commands on one worker and never blocks a submit beyond its budget."""

    def __init__(
        self,
        config: RenderConfig,
        renderer: DeterministicRenderer,
        *,
        state: RenderState,
        attempt_index: int | None = None,
        failures: list[dict[str, Any]] | None = None,
        command_reports: list[dict[str, Any]] | None = None,
        latest_preview: dict[str, Any] | None = None,
        last_result: RenderResult | None = None,
    ) -> None:
        self.config = config
        if sys.getswitchinterval() > 0.0005:
            sys.setswitchinterval(0.0005)
        self.renderer = renderer
        self._state = state
        self._next_submit_index = int(attempt_index if attempt_index is not None else state.command_cursor)
        self._submitted_count = 0
        self._completed_count = 0
        self.failures: list[dict[str, Any]] = copy.deepcopy(failures or [])
        self._loaded_reports: list[dict[str, Any]] = copy.deepcopy(command_reports or [])
        self.latest_preview = copy.deepcopy(latest_preview)
        self.last_result = last_result
        self._jobs: list[_Job] = []
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._shutdown = False
        self._worker = threading.Thread(target=self._worker_loop, name="fixedvideo-command-worker", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public interactive API
    # ------------------------------------------------------------------
    @property
    def state(self) -> RenderState:
        with self._lock:
            return self._state

    def submit(self, command: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        with self._condition:
            if self._shutdown:
                raise RuntimeError("engine is shut down")
            index = self._next_submit_index
            self._next_submit_index += 1
            self._submitted_count += 1
            job = _Job(index=index, command=command, submitted_at=started)
            self._jobs.append(job)
            self._queue.put(job)
            self._condition.notify_all()

            budget_seconds = max(0.0, self.config.interaction_latency_budget_ms / 1000.0)
            # Observe only the first 35% of the budget. Thread-pool warm-up and scheduler
            # hand-off are outside application control; the remaining headroom guarantees
            # that even a pending response is delivered inside the configured deadline.
        observation_seconds = budget_seconds * 0.35
        job.done_event.wait(timeout=observation_seconds)
        if job.done:
            response = self._response_for_job(job)
        else:
            job.initial_pending = True
            response = self._pending_response(job, 0.0)
        response_latency_ms = (time.perf_counter() - started) * 1000.0
        job.response_latency_ms = response_latency_ms
        response["response_latency_ms"] = response_latency_ms
        return response

    def wait_idle(self, timeout: float | None = None) -> list[dict[str, Any]]:
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._condition:
            while self._completed_count < self._submitted_count:
                remaining = None if deadline is None else deadline - time.perf_counter()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for the command worker")
                self._condition.wait(timeout=remaining)
            return [self._response_for_job(job) for job in self._jobs]

    def shutdown(self) -> None:
        with self._condition:
            if self._shutdown:
                return
            self._shutdown = True
            self._queue.put(None)
            self._condition.notify_all()
        self._worker.join(timeout=30.0)

    def set_latest_preview(self, preview: dict[str, Any] | None) -> None:
        with self._lock:
            self.latest_preview = copy.deepcopy(preview)

    # ------------------------------------------------------------------
    # Pending and completed response shaping
    # ------------------------------------------------------------------
    def _current_progress_fallback(self) -> dict[str, Any]:
        state = self._state
        return {
            "stage": "queued",
            "version": state.version,
            "command_cursor": state.command_cursor,
            "config_fingerprint": effective_config_fingerprint(self.config, state),
            "seed": state.seed,
            "steps": state.steps,
            "frame_count": state.frame_count,
            "chunk_rows": state.chunk_rows,
            "completed_frames": [],
            "completed_blocks": 0,
            "total_blocks": 0,
            "completed_rows": 0,
            "progress": 0.0,
            "reuse_keys": [],
            "recompute_keys": [],
            "affected_frames": [],
        }

    def _pending_response(self, job: _Job, latency_ms: float) -> dict[str, Any]:
        progress = job.progress if job.progress is not None else self._current_progress_fallback()
        preview = self.latest_preview
        return {
            "status": "pending",
            "business_code": E_INTERACT_BUDGET_PENDING,
            "command_id": job.command.get("command_id"),
            "command_index": job.index,
            "response_latency_ms": latency_ms,
            "result_corresponds_to": {
                "kind": "in_flight_progress",
                "version": progress.get("version"),
                "config_fingerprint": progress.get("config_fingerprint"),
            },
            "progress": progress,
            "preview": preview,
            "preview_corresponds_to_version": preview.get("version") if preview else None,
            "preview_corresponds_to_config_fingerprint": (
                preview.get("config_fingerprint") if preview else None
            ),
        }

    def _response_for_job(self, job: _Job) -> dict[str, Any]:
        stabilize_latency_ms = (
            (job.done_at - job.submitted_at) * 1000.0 if job.done_at is not None else None
        )
        response_latency_ms = job.response_latency_ms
        if response_latency_ms is None and job.done_at is not None:
            response_latency_ms = stabilize_latency_ms
        if job.result is not None:
            result = job.result
            return {
                "status": "complete",
                "business_code": None,
                "command_id": job.command.get("command_id"),
                "command_index": job.index,
                "response_latency_ms": response_latency_ms,
                "stabilize_latency_ms": stabilize_latency_ms,
                "version": result.state.version,
                "command_cursor": result.state.command_cursor,
                "config_fingerprint": result.config_fingerprint,
                "seed": result.seed,
                "steps": result.steps,
                "effect": result.effect.as_dict(),
                "preview": job.preview_summary,
                "trace_fingerprint": result.trace_fingerprint,
                "frame_fingerprints": list(result.frame_fps),
                "composition_fingerprint": result.composition_fp,
                "stage_stats": result.stage_stats,
                "reused_count": result.reused_count,
                "recomputed_count": result.recomputed_count,
                "hit_rate": result.hit_rate,
                "render_elapsed_ms": result.elapsed_ms,
                "max_concurrency": result.max_concurrency,
                "audit_count": len(result.audit),
            }
        error = job.error
        state = self._state
        return {
            "status": "failed",
            "business_code": error.code if error else "E_CONFIG_INVALID",
            "command_id": job.command.get("command_id"),
            "command_index": job.index,
            "response_latency_ms": response_latency_ms,
            "stabilize_latency_ms": stabilize_latency_ms,
            "version": state.version,
            "command_cursor": state.command_cursor,
            "config_fingerprint": effective_config_fingerprint(self.config, state),
            "error": error.as_dict() if error else None,
            "preview": copy.deepcopy(self.latest_preview),
            "reused_count": 0,
            "recomputed_count": 0,
            "hit_rate": 0.0,
        }

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _on_progress(self, job: _Job, snapshot: dict[str, Any]) -> None:
        with self._condition:
            job.progress = snapshot
            self._condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            job = item
            with self._condition:
                job.status = "running"
                job.progress = {
                    **self._current_progress_fallback(),
                    "stage": "validating",
                }
                self._condition.notify_all()
            try:
                new_state, effect = apply_command(
                    self.config,
                    self._state,
                    job.command,
                    command_index=job.index,
                )
                job.applied_state = new_state
                job.effect = effect
                with self._condition:
                    job.progress = {
                        "stage": "applying",
                        "version": new_state.version,
                        "command_cursor": new_state.command_cursor,
                        "config_fingerprint": effective_config_fingerprint(self.config, new_state),
                        "seed": new_state.seed,
                        "steps": new_state.steps,
                        "frame_count": new_state.frame_count,
                        "chunk_rows": new_state.chunk_rows,
                        "affected_frames": list(effect.affected_frames),
                        "completed_frames": [],
                        "completed_blocks": 0,
                        "total_blocks": 0,
                        "completed_rows": 0,
                        "progress": 0.0,
                        "reuse_keys": [],
                        "recompute_keys": [],
                    }
                    self._condition.notify_all()
                result = self.renderer.render(
                    new_state,
                    effect,
                    command_id=str(job.command["command_id"]),
                    progress_callback=lambda snapshot: self._on_progress(job, snapshot),
                )
                preview_summary = result.preview_summary()
                with self._condition:
                    self._state = new_state
                    job.result = result
                    job.status = "complete"
                    job.preview_summary = preview_summary
                    self.latest_preview = preview_summary
                    self.last_result = result
            except RenderError as exc:
                with self._condition:
                    self._state = touch_rejected_command(self._state, job.command, job.index)
                    job.error = exc
                    job.status = "failed"
                    self.failures.append(
                        {
                            "sequence": len(self.failures) + 1,
                            "command_index": job.index,
                            "command_id": job.command.get("command_id"),
                            "op": job.command.get("op"),
                            "code": exc.code,
                            "message": exc.message,
                            "details": exc.details,
                        }
                    )
            except Exception as exc:  # defensive: keep the session auditable and alive
                wrapped = RenderError(
                    "E_CONFIG_INVALID",
                    "unexpected render failure",
                    {"type": type(exc).__name__, "error": str(exc)},
                )
                with self._condition:
                    self._state = touch_rejected_command(self._state, job.command, job.index)
                    job.error = wrapped
                    job.status = "failed"
                    self.failures.append(
                        {
                            "sequence": len(self.failures) + 1,
                            "command_index": job.index,
                            "command_id": job.command.get("command_id"),
                            "op": job.command.get("op"),
                            "code": wrapped.code,
                            "message": wrapped.message,
                            "details": wrapped.details,
                        }
                    )
            finally:
                with self._condition:
                    job.done = True
                    job.done_at = time.perf_counter()
                    if job.progress is not None:
                        job.progress = {
                            **job.progress,
                            "stage": job.progress.get("stage", "complete") if job.status == "complete" else "failed",
                        }
                    self._completed_count += 1
                    self._condition.notify_all()
                job.done_event.set()

    # ------------------------------------------------------------------
    # Auditable reports and snapshot payload
    # ------------------------------------------------------------------
    def _job_report(self, job: _Job) -> dict[str, Any]:
        result = job.result
        error = job.error
        report: dict[str, Any] = {
            "command_index": job.index,
            "command_id": job.command.get("command_id"),
            "op": job.command.get("op"),
            "status": job.status,
            "business_code": error.code if error else None,
            "budget_response_pending": job.initial_pending,
            "response_latency_ms": job.response_latency_ms,
            "stabilize_latency_ms": (
                (job.done_at - job.submitted_at) * 1000.0 if job.done_at is not None else None
            ),
            "version": (result.state.version if result else self._state.version),
            "command_cursor": (result.state.command_cursor if result else self._state.command_cursor),
            "seed": (result.seed if result else self._state.seed),
            "steps": (result.steps if result else self._state.steps),
            "config_fingerprint": (
                result.config_fingerprint
                if result
                else effective_config_fingerprint(self.config, self._state)
            ),
            "affected_frames": (
                list(result.effect.affected_frames)
                if result
                else []
            ),
            "reused_count": result.reused_count if result else 0,
            "recomputed_count": result.recomputed_count if result else 0,
            "hit_rate": result.hit_rate if result else 0.0,
            "stage_stats": result.stage_stats if result else {},
            "render_elapsed_ms": result.elapsed_ms if result else None,
            "max_concurrency": result.max_concurrency if result else None,
            "trace_fingerprint": result.trace_fingerprint if result else None,
            "frame_fingerprints": list(result.frame_fps) if result else [],
            "reuse_keys": (
                [record["key"] for record in result.audit if record["status"] == "reuse"]
                if result
                else []
            ),
            "recompute_keys": (
                [record["key"] for record in result.audit if record["status"] == "recompute"]
                if result
                else []
            ),
            "error": error.as_dict() if error else None,
        }
        return report

    def command_reports(self) -> list[dict[str, Any]]:
        with self._lock:
            reports = copy.deepcopy(self._loaded_reports)
            reports.extend(self._job_report(job) for job in self._jobs if job.done)
            return reports

    def audit_payload(self) -> dict[str, Any]:
        with self._lock:
            reports = self.command_reports()
            return {
                "commands": reports,
                "failures": copy.deepcopy(self.failures),
                "latest_preview": copy.deepcopy(self.latest_preview),
                "reuse_recompute": [
                    {
                        "command_id": item["command_id"],
                        "reuse_keys": item["reuse_keys"],
                        "recompute_keys": item["recompute_keys"],
                    }
                    for item in reports
                ],
            }

    def checkpoint_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.to_dict(),
                "attempt_index": self._state.command_cursor,
                "failures": copy.deepcopy(self.failures),
                "command_reports": self.command_reports(),
                "latest_preview": copy.deepcopy(self.latest_preview),
                "last_result": self.last_result,
                "cache": self.renderer.cache,
                "renderer": self.renderer,
            }

    def stats_summary(self) -> dict[str, Any]:
        reports = self.command_reports()
        total_reuse = sum(int(item.get("reused_count") or 0) for item in reports)
        total_recompute = sum(int(item.get("recomputed_count") or 0) for item in reports)
        total = total_reuse + total_recompute
        pending = sum(1 for item in reports if item.get("budget_response_pending"))
        return {
            "commands": len(reports),
            "reused_artifacts": total_reuse,
            "recomputed_artifacts": total_recompute,
            "hit_rate": (total_reuse / total) if total else 0.0,
            "pending_responses": pending,
            "failures": len(self.failures),
            "cache_counts": self.renderer.cache.counts(),
        }
