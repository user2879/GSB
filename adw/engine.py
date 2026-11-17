"""Background generation scheduler with latency-budget response semantics."""

from __future__ import annotations

import copy
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .canonical import quantize_score
from .config import config_hash, enabled_algorithms
from .data import Dataset, feature_data_fingerprint
from .errors import E_BUDGET_PENDING
from .pipeline import DetectionPipeline, RunOutcome, canonical_result_payload, run_full
from .storage import CacheStats, ContentCache, SnapshotManager, atomic_write_json


def _first_differences(left: Any, right: Any, path: str = "$", limit: int = 20) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    def visit(a: Any, b: Any, current: str) -> None:
        if len(differences) >= limit:
            return
        if type(a) is not type(b):
            differences.append({"path": current, "incremental": a, "full": b})
            return
        if isinstance(a, dict):
            keys = sorted(set(a) | set(b))
            for key in keys:
                if key not in a or key not in b:
                    differences.append({"path": f"{current}.{key}", "incremental": a.get(key), "full": b.get(key)})
                    if len(differences) >= limit:
                        return
                else:
                    visit(a[key], b[key], f"{current}.{key}")
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append({"path": f"{current}.length", "incremental": len(a), "full": len(b)})
                return
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                visit(left_item, right_item, f"{current}[{index}]")
        elif a != b:
            differences.append({"path": current, "incremental": a, "full": b})

    visit(left, right, path)
    return differences


class WorkbenchEngine:
    def __init__(
        self,
        dataset: Dataset,
        config: dict[str, Any],
        *,
        root: str | Path,
        injection_items: list[dict[str, Any]] | None = None,
        injection_labels: dict[str, Any] | None = None,
    ):
        self.dataset = dataset
        self.root = Path(root).resolve()
        self.injection_items = list(injection_items or [])
        self.injection_labels = dict(injection_labels or {})
        self._condition = threading.Condition(threading.RLock())
        self._target_config = copy.deepcopy(config)
        self._generation = 0
        self._completed_generation = 0
        self._affected_by_generation: dict[int, list[str]] = {}
        self._last_payload: dict[str, Any] | None = None
        self._last_diagnostics: dict[str, Any] | None = None
        self._last_outcome: RunOutcome | None = None
        self._progress: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._graceful_stopped = False
        self._force_stopped = False
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._history: list[dict[str, Any]] = []
        self._history_seq = 0
        self._last_consistency: dict[str, Any] | None = None
        self._started_at = time.time()
        if hasattr(sys, "setswitchinterval"):
            sys.setswitchinterval(min(float(sys.getswitchinterval()), 0.001))
        self._data_fingerprint = feature_data_fingerprint(dataset, list(config["data"]["feature_columns"]))
        storage = config["storage"]
        self.snapshot_manager = SnapshotManager(self._resolve(storage["snapshot_dir"]))
        self.cache = ContentCache(self._resolve(storage["cache_dir"]), enabled=True)
        self.recovery = self._restore_snapshots()

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def _restore_snapshots(self) -> dict[str, Any]:
        summary = self.snapshot_manager.recovery_summary(self._data_fingerprint)
        scan = self.snapshot_manager.scan(self._data_fingerprint)
        warmed = 0
        dummy_stats = CacheStats()
        for stage, record in scan["records"].items():
            payload = record.get("payload", {})
            try:
                if stage == "preprocess" and isinstance(payload.get("preprocess_key"), str):
                    self.cache.put(
                        "preprocess",
                        payload["preprocess_key"],
                        {"data": payload["artifact"]["data"]},
                        {"restored_from_snapshot": True},
                        dummy_stats,
                    )
                    warmed += 1
                elif stage == "score" and isinstance(payload.get("algorithm_keys"), dict):
                    for name, key in payload["algorithm_keys"].items():
                        artifact = payload.get("artifacts", {}).get(name)
                        if isinstance(key, str) and isinstance(artifact, dict):
                            self.cache.put(
                                "algorithm",
                                key,
                                artifact,
                                {"algorithm": name, "restored_from_snapshot": True},
                                dummy_stats,
                            )
                            warmed += 1
                elif stage == "decision" and isinstance(payload.get("decision_key"), str):
                    self.cache.put(
                        "decision",
                        payload["decision_key"],
                        {"records": payload.get("records", {})},
                        {"restored_from_snapshot": True},
                        dummy_stats,
                    )
                    warmed += 1
            except (KeyError, OSError, TypeError, ValueError):
                continue
        summary["warmed_cache_entries"] = warmed
        return summary

    def start(self) -> None:
        with self._condition:
            if self._worker is not None:
                return
            if self._generation == 0:
                self._generation = 1
                self._affected_by_generation[1] = list(enabled_algorithms(self._target_config))
            self._worker = threading.Thread(target=self._worker_loop, name="adw-scheduler", daemon=True)
            self._worker.start()
            self._condition.notify_all()

    @property
    def current_config(self) -> dict[str, Any]:
        with self._condition:
            return copy.deepcopy(self._target_config)

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    @property
    def completed_generation(self) -> int:
        with self._condition:
            return self._completed_generation

    @property
    def graceful_stopped(self) -> bool:
        with self._condition:
            return self._graceful_stopped

    @property
    def last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._generation <= self._completed_generation and not self._graceful_stopped:
                    if self._stop_event.is_set():
                        self._graceful_stopped = True
                        self._condition.notify_all()
                        return
                    self._condition.wait(timeout=0.05)
                if self._graceful_stopped:
                    return
                generation = self._generation
                config = copy.deepcopy(self._target_config)
                affected = list(self._affected_by_generation.get(generation, []))
            progress_callback = self._make_progress_callback(generation)
            pipeline = DetectionPipeline(
                self.dataset,
                config,
                cache=self.cache,
                snapshots=self.snapshot_manager,
                injection_items=self.injection_items,
                affected_algorithms=affected,
                stop_event=self._stop_event,
                progress=progress_callback,
            )
            try:
                outcome = pipeline.run()
            except BaseException as exc:
                outcome = RunOutcome(
                    status="error",
                    payload=None,
                    diagnostics={"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                    error=f"{type(exc).__name__}: {exc}",
                )
            with self._condition:
                if outcome.payload is not None:
                    self._last_payload = outcome.payload
                self._last_diagnostics = outcome.diagnostics
                self._last_outcome = outcome
                self._completed_generation = max(self._completed_generation, generation)
                self._progress = None
                if outcome.error:
                    self._last_error = outcome.error
                elif outcome.status == "ok":
                    self._last_error = None
                if outcome.stopped:
                    self._graceful_stopped = True
                self._condition.notify_all()
                if outcome.stopped:
                    return
            if outcome.status == "ok" and outcome.payload is not None:
                try:
                    self.write_report()
                except OSError:
                    pass
            with self._condition:
                if self._stop_event.is_set():
                    self._graceful_stopped = True
                    self._condition.notify_all()
                    return

    def _make_progress_callback(self, generation: int):
        def callback(stage: str, completed: int, total: int, detail: dict[str, Any]) -> None:
            with self._condition:
                self._progress = {
                    "generation": generation,
                    "stage": stage,
                    "completed_samples": int(completed),
                    "total_samples": int(total),
                    "detail": dict(detail),
                }
                self._condition.notify_all()

        return callback

    def submit_config(
        self,
        config: dict[str, Any],
        *,
        affected_algorithms: list[str],
        changed_fields: list[str] | None = None,
        needs_run: bool = True,
        source_command: str = "",
    ) -> dict[str, Any]:
        if self._worker is None:
            self.start()
        started = time.perf_counter()
        budget_ms = float(config["interaction"]["latency_budget_ms"])
        with self._condition:
            self._target_config = copy.deepcopy(config)
            if needs_run:
                self._generation += 1
                requested = self._generation
                self._affected_by_generation[requested] = list(affected_algorithms)
                self._condition.notify_all()
            else:
                requested = self._completed_generation
        if not needs_run:
            response = self._build_response(
                requested_generation=requested,
                budget_ms=budget_ms,
                started=started,
                affected=affected_algorithms,
                pending=False,
                include_result=False,
                code=None,
                changed_fields=changed_fields,
            )
            self._record_response(source_command, requested, response)
            return response
        response = self._wait_for_generation(
            requested,
            budget_ms,
            started,
            affected_algorithms,
            changed_fields=changed_fields,
        )
        self._record_response(source_command, requested, response)
        return response

    def _wait_for_generation(
        self,
        requested: int,
        budget_ms: float,
        started: float,
        affected: list[str],
        *,
        changed_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        # Keep a margin for acquisition of the GIL and response construction. A
        # CPU-bound worker must never turn a budget-limited reply into a blocking one.
        reserve_ms = min(10.0, max(2.0, budget_ms * 0.20))
        wait_ms = max(0.0, budget_ms - reserve_ms)
        deadline = started + wait_ms / 1000.0
        timed_out = False
        with self._condition:
            while self._completed_generation < requested and not self._graceful_stopped:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    timed_out = True
                    break
                self._condition.wait(timeout=remaining)
            pending = timed_out or self._completed_generation < requested
        response = self._build_response(
            requested_generation=requested,
            budget_ms=budget_ms,
            started=started,
            affected=affected,
            pending=pending,
            include_result=False,
            code=E_BUDGET_PENDING if pending else None,
            changed_fields=changed_fields,
        )
        return response

    def _build_response(
        self,
        *,
        requested_generation: int,
        budget_ms: float,
        started: float,
        affected: list[str],
        pending: bool,
        include_result: bool,
        code: str | None,
        changed_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._condition:
            payload = self._last_payload
            diagnostics = self._last_diagnostics
            progress = copy.deepcopy(self._progress)
            completed_generation = self._completed_generation
            graceful = self._graceful_stopped
            last_error = self._last_error
        elapsed_ms = quantize_score((time.perf_counter() - started) * 1000.0)
        over_budget = elapsed_ms > budget_ms
        if pending:
            status = "pending"
        elif last_error and (payload is None or requested_generation > completed_generation):
            status = "error"
            code = code or "E_RUN_ERROR"
        elif graceful and payload is None:
            status = "stopped"
        else:
            status = "ok"
        summary = None
        if payload is not None:
            summary = dict(payload.get("summary", {}))
            summary["result_generation"] = completed_generation
            summary["failed_sample_count"] = summary.get("failed_sample_count", 0)
        else:
            summary = {
                "total_samples": len(self.dataset.sample_ids),
                "completed_samples": 0,
                "anomaly_count": 0,
                "failed_sample_count": 0,
                "selected_algorithms": enabled_algorithms(self.current_config),
                "affected_algorithms": list(affected),
                "result_generation": completed_generation,
            }
        response: dict[str, Any] = {
            "status": status,
            "code": code,
            "message": "latest result available" if status == "ok" else "latest completed summary returned",
            "summary": summary,
            "result": payload if include_result else None,
            "latest_completed_generation": completed_generation,
            "requested_generation": requested_generation,
            "affected_algorithms": list(affected),
            "changed_fields": list(changed_fields or []),
            "progress": progress,
            "interaction": {
                "elapsed_ms": elapsed_ms,
                "budget_ms": quantize_score(budget_ms),
                "over_budget": bool(over_budget),
                "pending": bool(pending),
                "pending_code": E_BUDGET_PENDING if pending else None,
                "requested_generation": requested_generation,
                "completed_generation": completed_generation,
            },
            "recovery": self.recovery,
        }
        if last_error:
            response["error"] = last_error
        return response

    def _record_response(self, command: str, requested: int, response: dict[str, Any]) -> None:
        with self._condition:
            self._history_seq += 1
            interaction = response.get("interaction", {})
            self._history.append(
                {
                    "seq": self._history_seq,
                    "command": command,
                    "requested_generation": requested,
                    "completed_generation": interaction.get("completed_generation"),
                    "elapsed_ms": interaction.get("elapsed_ms"),
                    "budget_ms": interaction.get("budget_ms"),
                    "over_budget": bool(interaction.get("over_budget")),
                    "pending": bool(interaction.get("pending")),
                    "code": response.get("code"),
                }
            )

    def invalid_command_response(self, command: str, failures: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.perf_counter()
        with self._condition:
            budget_ms = float(self._target_config["interaction"]["latency_budget_ms"])
        response = {
            "status": "failed",
            "code": "E_CONFIG_INVALID",
            "message": "command rejected; previous configuration retained",
            "failures": failures,
            "summary": self.status_summary(),
            "result": None,
            "interaction": {
                "elapsed_ms": quantize_score((time.perf_counter() - started) * 1000.0),
                "budget_ms": quantize_score(budget_ms),
                "over_budget": (time.perf_counter() - started) * 1000.0 > budget_ms,
                "pending": False,
                "pending_code": None,
            },
        }
        self._record_response(command, self.generation, response)
        return response

    def status_summary(self) -> dict[str, Any]:
        with self._condition:
            payload = self._last_payload
            progress = copy.deepcopy(self._progress)
            completed_generation = self._completed_generation
            requested_generation = self._generation
            config = copy.deepcopy(self._target_config)
        if payload is not None:
            summary = dict(payload.get("summary", {}))
        else:
            summary = {
                "total_samples": len(self.dataset.sample_ids),
                "completed_samples": 0,
                "anomaly_count": 0,
                "failed_sample_count": 0,
                "selected_algorithms": enabled_algorithms(config),
                "affected_algorithms": [],
            }
        summary["result_generation"] = completed_generation
        summary["requested_generation"] = requested_generation
        summary["progress"] = progress
        return summary

    def status_response(self) -> dict[str, Any]:
        started = time.perf_counter()
        with self._condition:
            budget_ms = float(self._target_config["interaction"]["latency_budget_ms"])
            payload = self._last_payload
            completed_generation = self._completed_generation
            requested_generation = self._generation
        response = {
            "status": "pending" if requested_generation > completed_generation else "ok",
            "code": E_BUDGET_PENDING if requested_generation > completed_generation else None,
            "message": "current workbench status",
            "summary": self.status_summary(),
            "result": payload,
            "interaction": {
                "elapsed_ms": quantize_score((time.perf_counter() - started) * 1000.0),
                "budget_ms": quantize_score(budget_ms),
                "over_budget": False,
                "pending": requested_generation > completed_generation,
                "pending_code": E_BUDGET_PENDING if requested_generation > completed_generation else None,
                "completed_generation": completed_generation,
                "requested_generation": requested_generation,
            },
            "recovery": self.recovery,
        }
        self._record_response("status", requested_generation, response)
        return response

    def result_response(self) -> dict[str, Any]:
        started = time.perf_counter()
        with self._condition:
            budget_ms = float(self._target_config["interaction"]["latency_budget_ms"])
            payload = copy.deepcopy(self._last_payload)
            completed_generation = self._completed_generation
            requested_generation = self._generation
        elapsed = quantize_score((time.perf_counter() - started) * 1000.0)
        response = {
            "status": "ok" if payload is not None else "pending",
            "code": None if payload is not None else E_BUDGET_PENDING,
            "message": "latest completed full result",
            "summary": self.status_summary(),
            "result": payload,
            "interaction": {
                "elapsed_ms": elapsed,
                "budget_ms": quantize_score(budget_ms),
                "over_budget": elapsed > budget_ms,
                "pending": requested_generation > completed_generation,
                "pending_code": E_BUDGET_PENDING if requested_generation > completed_generation else None,
                "completed_generation": completed_generation,
                "requested_generation": requested_generation,
            },
            "recovery": self.recovery,
        }
        self._record_response("result", requested_generation, response)
        return response

    def wait_current(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._condition:
            requested = self._generation
            while self._completed_generation < requested and not self._graceful_stopped:
                if deadline is None:
                    self._condition.wait(timeout=0.05)
                else:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        return False
                    self._condition.wait(timeout=remaining)
            return self._completed_generation >= requested

    def settle_response(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.wait_current()
        with self._condition:
            budget_ms = float(self._target_config["interaction"]["latency_budget_ms"])
        response = self._build_response(
            requested_generation=self.generation,
            budget_ms=budget_ms,
            started=started,
            affected=[],
            pending=False,
            include_result=True,
            code=None,
        )
        response["message"] = "settled by explicit diagnostic command"
        self._record_response("settle", self.generation, response)
        return response

    def compare_full(self) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.wait_current():
            return {
                "status": "pending",
                "code": E_BUDGET_PENDING,
                "message": "incremental generation did not settle",
                "elapsed_ms": quantize_score((time.perf_counter() - started) * 1000.0),
            }
        with self._condition:
            config = copy.deepcopy(self._target_config)
            payload = copy.deepcopy(self._last_payload)
            generation = self._completed_generation
        if payload is None:
            return {"status": "error", "message": "no completed incremental result to compare"}
        full = run_full(self.dataset, config, injection_items=self.injection_items)
        if full.payload is None:
            return {
                "status": "error",
                "message": "full serial run did not produce a result",
                "error": full.error,
            }
        incremental_canonical = canonical_result_payload(payload)
        full_canonical = canonical_result_payload(full.payload)
        differences = _first_differences(incremental_canonical, full_canonical)
        result = {
            "status": "ok",
            "equal": not differences,
            "differences": differences,
            "compared_fields": [
                "samples",
                "ranking",
                "anomalies",
                "algorithm_rankings",
                "score_table",
                "failures",
                "summary",
            ],
            "incremental_generation": generation,
            "elapsed_ms": quantize_score((time.perf_counter() - started) * 1000.0),
            "incremental_cache": self._last_diagnostics.get("cache") if self._last_diagnostics else None,
            "full_run": {
                "cache": full.diagnostics.get("cache"),
                "recalculated_algorithms": full.diagnostics.get("recalculated_algorithms"),
            },
        }
        self._last_consistency = result
        return result

    def report_payload(self) -> dict[str, Any]:
        with self._condition:
            config = copy.deepcopy(self._target_config)
            diagnostics = copy.deepcopy(self._last_diagnostics) or {}
            payload = copy.deepcopy(self._last_payload)
            history = copy.deepcopy(self._history)
            generation = self._completed_generation
            requested = self._generation
            error = self._last_error
        storage = config["storage"]
        report = {
            "schema_version": 1,
            "generated_at_unix": round(time.time(), 6),
            "config_hash": config_hash(config),
            "data_fingerprint": self._data_fingerprint,
            "completed_generation": generation,
            "requested_generation": requested,
            "recovery": self.recovery,
            "latest_run": diagnostics,
            "latest_summary": payload.get("summary") if payload else None,
            "algorithm_score_distributions": diagnostics.get("distributions"),
            "cache_hit_and_invalidation": diagnostics.get("cache"),
            "reused_algorithms": diagnostics.get("reused_algorithms"),
            "recalculated_algorithms": diagnostics.get("recalculated_algorithms"),
            "failures": payload.get("failures") if payload else [],
            "injection": diagnostics.get("injection"),
            "injection_report_path": str(self._injection_report_path()),
            "command_responses": history,
            "consistency_check": self._last_consistency,
            "error": error,
        }
        return report

    def _injection_report_path(self) -> Path:
        with self._condition:
            config = copy.deepcopy(self._target_config)
        value = self.injection_labels.get("report_path") or config["storage"].get(
            "injection_report_path", ".workbench/injection_report.json"
        )
        path = Path(str(value))
        if path.is_absolute():
            return path
        root = Path(str(self.injection_labels.get("root"))) if self.injection_labels.get("root") else self.root
        return (root / path).resolve()

    def write_report(self) -> dict[str, Any]:
        with self._condition:
            config = copy.deepcopy(self._target_config)
        report = self.report_payload()
        report_path = self._resolve(config["storage"]["report_path"])
        atomic_write_json(report_path, report)
        injection_path = self._injection_report_path()
        if self.injection_items:
            injection_report = {
                "schema_version": 1,
                "generated_at_unix": report["generated_at_unix"],
                "config_hash": report["config_hash"],
                "data_fingerprint": report["data_fingerprint"],
                "injection": report.get("injection"),
                "failures": report.get("failures"),
                "mapped_business_codes": sorted(
                    {
                        item.get("business_code")
                        for item in (report.get("injection") or {}).get("items", [])
                        if item.get("business_code")
                    }
                ),
            }
            atomic_write_json(injection_path, injection_report)
        return {"report_path": str(report_path), "injection_report_path": str(injection_path)}

    def report_response(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            paths = self.write_report()
            status = "ok"
            error = None
        except OSError as exc:
            paths = {}
            status = "error"
            error = str(exc)
        with self._condition:
            budget_ms = float(self._target_config["interaction"]["latency_budget_ms"])
        elapsed = quantize_score((time.perf_counter() - started) * 1000.0)
        response = {
            "status": status,
            "code": None if status == "ok" else "E_REPORT_WRITE",
            "message": "diagnostic report written" if status == "ok" else error,
            "paths": paths,
            "summary": self.status_summary(),
            "interaction": {
                "elapsed_ms": elapsed,
                "budget_ms": quantize_score(budget_ms),
                "over_budget": elapsed > budget_ms,
                "pending": False,
                "pending_code": None,
            },
        }
        self._record_response("report", self.generation, response)
        return response

    def recover_response(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.recovery = self.snapshot_manager.recovery_summary(self._data_fingerprint)
        with self._condition:
            budget_ms = float(self._target_config["interaction"]["latency_budget_ms"])
        elapsed = quantize_score((time.perf_counter() - started) * 1000.0)
        response = {
            "status": "ok",
            "code": None,
            "message": "recovery point information",
            "recovery": self.recovery,
            "summary": self.status_summary(),
            "interaction": {
                "elapsed_ms": elapsed,
                "budget_ms": quantize_score(budget_ms),
                "over_budget": elapsed > budget_ms,
                "pending": False,
                "pending_code": None,
            },
        }
        self._record_response("recover", self.generation, response)
        return response

    def request_shutdown(self) -> None:
        with self._condition:
            self._stop_event.set()
            self._condition.notify_all()

    def join_worker(self, timeout: float = 5.0) -> bool:
        with self._condition:
            worker = self._worker
        if worker is None:
            return True
        worker.join(timeout=timeout)
        return not worker.is_alive()

    def shutdown(self) -> None:
        self.request_shutdown()
        self.join_worker(timeout=5.0)
        try:
            self.write_report()
        except OSError:
            pass
