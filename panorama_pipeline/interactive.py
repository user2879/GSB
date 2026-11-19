from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from .errors import E_INTERACT_BUDGET_PENDING, E_PARAM_INVALID, PipelineError
from .io_utils import canonical_json_bytes, sha256_bytes
from .pipeline import PipelineSession


class InteractiveEngine:
    def __init__(
        self,
        session: PipelineSession,
        *,
        budget_ms: int | None = None,
        worker_wrapper: Callable[[Callable[[], dict[str, Any]]], dict[str, Any]] | None = None,
    ):
        self.session = session
        self.budget_ms = (
            session.config.interaction_budget_ms if budget_ms is None else int(budget_ms)
        )
        if self.budget_ms <= 0:
            raise PipelineError(E_PARAM_INVALID, "interaction budget must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pano-command",
        )
        self._lock = threading.RLock()
        self._command_seq = 0
        self._futures: dict[int, Future] = {}
        self._completed: dict[int, dict[str, Any]] = {}
        self._latest_preview = copy.deepcopy(session.latest_preview)
        self._latest_completed_seq = 0
        self.worker_wrapper = worker_wrapper
        self._projected_params = copy.deepcopy(session.config.params_snapshot())

    @property
    def latest_completed_seq(self) -> int:
        with self._lock:
            return self._latest_completed_seq

    def _requested_version(self, command: dict[str, Any]) -> str:
        projected = copy.deepcopy(self._projected_params)
        if command.get("type") == "set_params":
            if command.get("iou_match_threshold") is not None:
                projected["iou_match_threshold"] = float(
                    command["iou_match_threshold"]
                )
            if command.get("retention_mode") is not None:
                projected["pixel_retention"]["mode"] = str(
                    command["retention_mode"]
                )
            if command.get("min_coverage") is not None:
                projected["pixel_retention"]["min_coverage"] = float(
                    command["min_coverage"]
                )
        self._projected_params = projected
        return sha256_bytes(canonical_json_bytes(projected))

    def submit(self, command: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._command_seq += 1
            command_seq = self._command_seq
            requested_version = self._requested_version(command)
            future = self._executor.submit(
                self._execute_command,
                command_seq,
                command,
                requested_version,
            )
            self._futures[command_seq] = future
        started = time.perf_counter()
        deadline = started + self.budget_ms / 1000.0
        remaining = max(0.0, deadline - time.perf_counter())
        try:
            completed_preview = future.result(timeout=remaining)
        except TimeoutError:
            completed_preview = None
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._completed[command_seq] = {
                    "status": "failed",
                    "is_latest": False,
                    "command_seq": command_seq,
                    "requested_params_snapshot_version": requested_version,
                    "business_code": getattr(exc, "code", "E_INTERNAL"),
                    "message": str(exc),
                    "elapsed_ms": elapsed_ms,
                }
                return dict(self._completed[command_seq])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if completed_preview is not None:
            response = {
                "status": "complete",
                "is_latest": True,
                "command_seq": command_seq,
                "requested_params_snapshot_version": requested_version,
                "params_snapshot_version": completed_preview[
                    "params_snapshot_version"
                ],
                "business_code": "OK",
                "business_codes": list(completed_preview["business_codes"]),
                "elapsed_ms": elapsed_ms,
                "preview": completed_preview,
            }
            with self._lock:
                self._completed[command_seq] = response
            return copy.deepcopy(response)

        with self._lock:
            latest_preview = copy.deepcopy(self._latest_preview)
            latest_seq = self._latest_completed_seq
        if latest_seq >= command_seq:
            with self._lock:
                response = copy.deepcopy(self._completed[command_seq])
            response["elapsed_ms"] = elapsed_ms
            return response
        progress = self.session.progress()
        if latest_preview is not None:
            latest_preview["is_latest"] = False
        response = {
            "status": "pending",
            "is_latest": False,
            "command_seq": command_seq,
            "requested_params_snapshot_version": requested_version,
            "params_snapshot_version": (
                None
                if latest_preview is None
                else latest_preview["params_snapshot_version"]
            ),
            "preview_source_command_seq": latest_seq,
            "business_code": E_INTERACT_BUDGET_PENDING,
            "business_codes": [E_INTERACT_BUDGET_PENDING],
            "elapsed_ms": elapsed_ms,
            "preview": latest_preview,
            "current_progress": progress,
        }
        with self._lock:
            self._completed[command_seq] = copy.deepcopy(response)
        return response

    def _execute_command(
        self,
        command_seq: int,
        command: dict[str, Any],
        requested_version: str,
    ) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            return self.session.execute(
                command,
                command_seq=command_seq,
                progress_callback=None,
            )

        preview = self.worker_wrapper(work) if self.worker_wrapper else work()
        with self._lock:
            self._latest_preview = copy.deepcopy(preview)
            self._latest_completed_seq = command_seq
            existing = self._completed.get(command_seq)
            if existing is not None and existing.get("status") == "pending":
                existing.update(
                    {
                        "status": "complete",
                        "is_latest": True,
                        "requested_params_snapshot_version": requested_version,
                        "params_snapshot_version": preview["params_snapshot_version"],
                        "business_code": "OK",
                        "business_codes": list(preview["business_codes"]),
                        "preview": copy.deepcopy(preview),
                        "current_progress": None,
                    }
                )
        return preview

    def latest_preview(self) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._latest_preview)

    def response_for(self, command_seq: int) -> dict[str, Any] | None:
        with self._lock:
            value = self._completed.get(command_seq)
            if (
                value is not None
                and value.get("status") == "pending"
                and self._latest_completed_seq >= command_seq
                and self._latest_preview is not None
            ):
                preview = copy.deepcopy(self._latest_preview)
                value = {
                    **value,
                    "status": "complete",
                    "is_latest": True,
                    "params_snapshot_version": preview["params_snapshot_version"],
                    "business_code": "OK",
                    "business_codes": list(preview["business_codes"]),
                    "preview": preview,
                    "current_progress": None,
                }
                self._completed[command_seq] = copy.deepcopy(value)
            return None if value is None else copy.deepcopy(value)

    def wait_command(self, command_seq: int, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            future = self._futures[command_seq]
        future.result(timeout=timeout)
        response = self.response_for(command_seq)
        if response is None:
            raise KeyError(f"no response for command {command_seq}")
        return response

    def wait_all(self, timeout: float | None = None) -> list[dict[str, Any]]:
        with self._lock:
            sequences = sorted(self._futures)
        for sequence in sequences:
            self.wait_command(sequence, timeout=timeout)
        return [self.response_for(sequence) for sequence in sequences]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
