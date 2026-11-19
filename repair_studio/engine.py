from __future__ import annotations

import copy
import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import (
    RenderParams,
    apply_command,
    canonical_json,
    command_affected_stages,
)
from .errors import (
    E_CONFIG_INVALID,
    E_RENDER_BUDGET_PENDING,
    E_RENDER_FAILED,
    StudioError,
    failure_item,
)
from .imaging import LoadedSample, load_sample_set
from .rendering import RenderOutcome, Renderer
from .snapshot import SnapshotStore


OK = "OK"


@dataclass
class Job:
    job_id: int
    command_index: int
    params: RenderParams
    expected_invalidated_stages: list[str]


@dataclass
class Completion:
    job: Job
    outcome: RenderOutcome


class InteractiveRenderService:
    def __init__(
        self,
        renderer: Renderer,
        sample: LoadedSample,
        initial_outcome: RenderOutcome | None = None,
    ) -> None:
        self.renderer = renderer
        self.sample = sample
        self._condition = threading.Condition(threading.RLock())
        self._pending: Job | None = None
        self._active: Job | None = None
        self._completed: Completion | None = (
            Completion(
                job=Job(
                    job_id=0,
                    command_index=0,
                    params=initial_outcome.params,
                    expected_invalidated_stages=[],
                ),
                outcome=initial_outcome,
            )
            if initial_outcome is not None
            else None
        )
        self._error: tuple[int, StudioError] | None = None
        self._progress: dict[str, Any] | None = (
            {
                "job_id": 0,
                "command_index": 0,
                "params_fingerprint": initial_outcome.params_fingerprint,
                "stage": "complete",
                "stage_index": 4,
                "total_stages": 4,
                "completed_rows": sample.config.height,
                "total_rows": sample.config.height,
                "progress": 1.0,
                "message": "bootstrap complete",
            }
            if initial_outcome is not None
            else None
        )
        self._next_job_id = 1
        self._stop_requested = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name=f"render-service-{sample.sample_id}",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        command_index: int,
        params: RenderParams,
        expected_invalidated_stages: list[str],
    ) -> Job:
        with self._condition:
            job = Job(
                job_id=self._next_job_id,
                command_index=command_index,
                params=params,
                expected_invalidated_stages=list(expected_invalidated_stages),
            )
            self._next_job_id += 1
            self._pending = job
            self._error = None
            self._condition.notify_all()
            return job

    def wait(self, job: Job, budget_ms: int, started_at: float) -> tuple[Completion | None, dict[str, Any] | None]:
        safety_seconds = min(0.003, max(0.00005, budget_ms / 1000.0 * 0.2))
        deadline = started_at + max(0.0, budget_ms / 1000.0 - safety_seconds)
        with self._condition:
            while True:
                if self._completed is not None:
                    if self._completed.job.job_id == job.job_id:
                        return self._completed, copy.deepcopy(self._progress)
                    if self._completed.outcome.params_fingerprint == job.params.fingerprint():
                        return self._completed, copy.deepcopy(self._progress)
                if self._completed is not None and self._completed.job.job_id > job.job_id:
                    return self._completed, copy.deepcopy(self._progress)
                if self._error is not None and self._error[0] >= job.job_id:
                    raise self._error[1]
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None, copy.deepcopy(self._progress)
                self._condition.wait(timeout=min(remaining, 0.05))

    def wait_until_complete(self, job: Job) -> Completion:
        with self._condition:
            while True:
                if self._completed is not None and self._completed.job.job_id == job.job_id:
                    return self._completed
                if self._completed is not None and self._completed.outcome.params_fingerprint == job.params.fingerprint():
                    return self._completed
                if self._error is not None and self._error[0] >= job.job_id:
                    raise self._error[1]
                self._condition.wait(timeout=0.1)

    def latest_completion(self) -> Completion | None:
        with self._condition:
            return self._completed

    def progress_snapshot(self) -> dict[str, Any] | None:
        with self._condition:
            return copy.deepcopy(self._progress)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop_requested = True
            self._condition.notify_all()
        self._thread.join(timeout=30.0)

    def _worker(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop_requested:
                    self._condition.wait(timeout=0.2)
                if self._stop_requested and self._pending is None:
                    return
                job = self._pending
                self._pending = None
                self._active = job
                self._progress = {
                    "job_id": job.job_id,
                    "command_index": job.command_index,
                    "params_fingerprint": job.params.fingerprint(),
                    "stage": "queued",
                    "stage_index": 0,
                    "total_stages": 4,
                    "completed_rows": 0,
                    "total_rows": self.sample.config.height,
                    "progress": 0.0,
                    "message": "render queued",
                }

            def progress_callback(event: dict[str, Any]) -> None:
                with self._condition:
                    self._progress = {"job_id": job.job_id, **event}
                    self._condition.notify_all()

            try:
                outcome = self.renderer.render(
                    self.sample,
                    job.params,
                    command_index=job.command_index,
                    expected_invalidated_stages=job.expected_invalidated_stages,
                    progress_callback=progress_callback,
                )
            except StudioError as error:
                with self._condition:
                    self._error = (job.job_id, error)
                    self._active = None
                    self._condition.notify_all()
                continue
            except Exception as error:
                wrapped = StudioError(E_RENDER_FAILED, f"render worker failed: {error}")
                with self._condition:
                    self._error = (job.job_id, wrapped)
                    self._active = None
                    self._condition.notify_all()
                continue
            with self._condition:
                self._completed = Completion(job=job, outcome=outcome)
                self._active = None
                self._progress = {
                    "job_id": job.job_id,
                    "command_index": job.command_index,
                    "params_fingerprint": outcome.params_fingerprint,
                    "stage": "complete",
                    "stage_index": 4,
                    "total_stages": 4,
                    "completed_rows": self.sample.config.height,
                    "total_rows": self.sample.config.height,
                    "progress": 1.0,
                    "message": "render complete",
                }
                self._condition.notify_all()


class SampleRunner:
    def __init__(
        self,
        sample: LoadedSample,
        output_root: Path,
        renderer: Renderer,
        snapshot_store: SnapshotStore,
        mode: str,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.sample = sample
        self.output_root = output_root
        self.renderer = renderer
        self.snapshots = snapshot_store
        self.mode = mode
        self.stop_event = stop_event or threading.Event()
        self.current_params = sample.config.params
        self.completed_cursor = 0
        self.last_snapshot_cursor = -1
        self.latest_outcome: RenderOutcome | None = None
        self.latest_command_index = 0
        self.last_applied_command_index = 0
        self.records: list[dict[str, Any]] = []
        self.snapshot_records: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self._service: InteractiveRenderService | None = None
        self._current_output = output_root / "current" / sample.sample_id
        self._current_output.mkdir(parents=True, exist_ok=True)

    def bootstrap(self) -> dict[str, Any]:
        started = time.perf_counter()
        outcome = self.renderer.render(self.sample, self.current_params, command_index=0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.latest_outcome = outcome
        self._write_current(outcome)
        snapshot = self._commit_snapshot(force=True)
        return {
            "elapsed_ms": round(elapsed_ms, 3),
            "budget_ms": self.sample.config.latency_budget_ms,
            "budget_met": elapsed_ms <= self.sample.config.latency_budget_ms,
            "preview": outcome.preview_summary(),
            "snapshot": snapshot,
            "business_code": OK,
            "config_sha256": self.sample.config.fingerprint(),
        }

    def run(self, commands: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.mode == "interactive":
            self._service = InteractiveRenderService(
                self.renderer,
                self.sample,
                initial_outcome=self.latest_outcome,
            )
        try:
            for index, command in enumerate(commands, start=1):
                if self.stop_event.is_set():
                    break
                response = self._run_command(index, command)
                self.records.append(response)
                if response.get("stop_requested"):
                    break
        finally:
            if self._service is not None:
                try:
                    self._finish_interactive()
                finally:
                    self._service.close()
        self._commit_snapshot(force=True)
        return self.records

    def _run_command(self, command_index: int, command: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        if self._service is not None:
            latest = self._service.latest_completion()
            if latest is not None and latest.job.command_index > self.latest_command_index:
                self._accept_completion(latest.job.command_index, latest.outcome)
        operation = str(command.get("op", command.get("type", ""))) if isinstance(command, Mapping) else ""
        if operation in {"exit", "request_exit", "shutdown"}:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return self._base_response(command_index, operation, started, elapsed_ms)
        if operation == "snapshot":
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            snapshot = self._commit_snapshot(force=True)
            response = self._base_response(command_index, operation, started, elapsed_ms)
            response["snapshot"] = snapshot
            return response

        previous_params = self.current_params
        try:
            next_params = apply_command(
                previous_params,
                self.sample.config.width,
                self.sample.config.height,
                command,
            )
        except StudioError as error:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            response = self._base_response(command_index, operation, started, elapsed_ms)
            response["business_code"] = error.code
            response["failure"] = error.as_failure()
            self.failures.append(
                failure_item(self.sample.sample_id, error)
                | {"command_index": command_index, "op": operation}
            )
            return response

        self.current_params = next_params
        self.last_applied_command_index = command_index
        expected_stages = [] if next_params == previous_params else command_affected_stages(operation)
        if self.mode == "sync":
            return self._run_sync(command_index, operation, next_params, expected_stages, started)
        return self._run_interactive(command_index, operation, next_params, expected_stages, started)

    def _run_sync(
        self,
        command_index: int,
        operation: str,
        params: RenderParams,
        expected_stages: list[str],
        started: float,
    ) -> dict[str, Any]:
        try:
            outcome = self.renderer.render(
                self.sample,
                params,
                command_index=command_index,
                expected_invalidated_stages=expected_stages,
            )
        except StudioError as error:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.failures.append(
                failure_item(self.sample.sample_id, error)
                | {"command_index": command_index, "op": operation}
            )
            response = self._base_response(command_index, operation, started, elapsed_ms, params)
            response["business_code"] = error.code
            response["failure"] = error.as_failure()
            return response
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._accept_completion(command_index, outcome)
        response = self._ready_response(command_index, operation, outcome, started, elapsed_ms)
        response["mode"] = "sync"
        response["budget_met"] = elapsed_ms <= self.sample.config.latency_budget_ms
        response["result_is_latest"] = True
        return response

    def _run_interactive(
        self,
        command_index: int,
        operation: str,
        params: RenderParams,
        expected_stages: list[str],
        started: float,
    ) -> dict[str, Any]:
        assert self._service is not None
        job = self._service.submit(command_index, params, expected_stages)
        try:
            completion, progress = self._service.wait(job, self.sample.config.latency_budget_ms, started)
        except StudioError as error:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.failures.append(
                failure_item(self.sample.sample_id, error)
                | {"command_index": command_index, "op": operation}
            )
            response = self._base_response(command_index, operation, started, elapsed_ms, params)
            response["business_code"] = error.code
            response["failure"] = error.as_failure()
            return response
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if completion is not None:
            outcome = completion.outcome
            if completion.job.command_index == command_index:
                self._accept_completion(command_index, outcome)
            response = self._ready_response(command_index, operation, outcome, started, elapsed_ms)
            response["budget_met"] = elapsed_ms <= self.sample.config.latency_budget_ms
            response["result_is_latest"] = outcome.params_fingerprint == params.fingerprint()
            return response

        response = self._base_response(command_index, operation, started, elapsed_ms, params)
        response["business_code"] = E_RENDER_BUDGET_PENDING
        response["budget_met"] = True
        response["result_is_latest"] = False
        response["affected_stages"] = expected_stages
        latest = self._service.latest_completion()
        if latest is None:
            response["latest_completed"] = None
            response["reused_count"] = 0
            response["invalidated_count"] = 0
        else:
            preview = latest.outcome.preview_summary()
            preview["command_index"] = latest.job.command_index
            preview["params_match_requested"] = (
                latest.outcome.params_fingerprint == params.fingerprint()
            )
            preview["completed_chunk_rows"] = self.sample.config.height
            preview["audit_source"] = "latest_completed"
            response["latest_completed"] = preview
            response["reused_count"] = len(latest.outcome.reused_stages)
            response["invalidated_count"] = len(latest.outcome.invalidated_stages)
        current_progress = progress or self._service.progress_snapshot()
        if current_progress is None:
            current_progress = {
                "stage": "queued",
                "stage_index": 0,
                "total_stages": 4,
                "completed_rows": 0,
                "total_rows": self.sample.config.height,
                "progress": 0.0,
                "params_fingerprint": params.fingerprint(),
            }
        current_progress["params_match_requested"] = (
            current_progress.get("params_fingerprint") == params.fingerprint()
        )
        response["progress"] = current_progress
        response["progress"]["affected_region"] = self._affected_region(operation, params)
        return response

    def _finish_interactive(self) -> None:
        if self._service is None:
            return
        final_job = self._service.submit(
            self.completed_cursor + 1,
            self.current_params,
            [],
        )
        completion = self._service.wait_until_complete(final_job)
        self._accept_completion(self.last_applied_command_index, completion.outcome, finish=True)

    def _accept_completion(self, command_index: int, outcome: RenderOutcome, finish: bool = False) -> None:
        self.latest_outcome = outcome
        self.latest_command_index = command_index
        self.completed_cursor = max(self.completed_cursor, command_index)
        self._write_current(outcome)
        interval = self.sample.config.snapshot_interval
        if finish or command_index % interval == 0:
            self._commit_snapshot(force=finish)

    def _write_current(self, outcome: RenderOutcome) -> None:
        temporary = self._current_output / ".final.png.tmp"
        temporary.write_bytes(outcome.final_png_bytes)
        os.replace(temporary, self._current_output / "final.png")
        metadata = {
            "sample_id": self.sample.sample_id,
            "command_index": self.latest_command_index,
            "params": outcome.params.as_dict(),
            "params_fingerprint": outcome.params_fingerprint,
            "stage_summaries": outcome.stage_summaries,
            "audit": outcome.audit(),
            "final_png_sha256": outcome.final_png_sha256,
        }
        temporary_meta = self._current_output / ".result.json.tmp"
        temporary_meta.write_bytes(canonical_json(metadata))
        os.replace(temporary_meta, self._current_output / "result.json")

    def _commit_snapshot(self, force: bool) -> dict[str, Any] | None:
        if self.latest_outcome is None:
            return None
        interval = self.sample.config.snapshot_interval
        if not force and self.completed_cursor - self.last_snapshot_cursor < interval:
            return None
        result = self.snapshots.commit(
            self.sample.sample_id,
            self.sample.config.fingerprint(),
            self.completed_cursor,
            self.latest_outcome,
        )
        result["completed_command_cursor"] = self.completed_cursor
        self.snapshot_records.append(result)
        if result.get("status") in {"committed", "unchanged"}:
            self.last_snapshot_cursor = self.completed_cursor
        return result

    def _base_response(
        self,
        command_index: int,
        operation: str,
        started: float,
        elapsed_ms: float,
        params: RenderParams | None = None,
    ) -> dict[str, Any]:
        requested = params or self.current_params
        return {
            "command_index": command_index,
            "op": operation,
            "business_code": OK,
            "elapsed_ms": round(elapsed_ms, 3),
            "budget_ms": self.sample.config.latency_budget_ms,
            "budget_met": elapsed_ms <= self.sample.config.latency_budget_ms,
            "requested_params_fingerprint": requested.fingerprint(),
            "requested_params": requested.as_dict(),
            "affected_region": self._affected_region(operation, requested),
            "reused_count": 0,
            "invalidated_count": 0,
            "stop_requested": operation in {"exit", "request_exit", "shutdown"},
        }

    def _ready_response(
        self,
        command_index: int,
        operation: str,
        outcome: RenderOutcome,
        started: float,
        elapsed_ms: float,
    ) -> dict[str, Any]:
        response = self._base_response(command_index, operation, started, elapsed_ms, outcome.params)
        preview = outcome.preview_summary()
        preview["command_index"] = command_index
        preview["params_match_requested"] = True
        preview["completed_chunk_rows"] = self.sample.config.height
        preview["audit_source"] = "latest_completed"
        response.update(
            {
                "business_code": OK,
                "result_is_latest": True,
                "latest_completed": preview,
                "reused_stages": outcome.reused_stages,
                "invalidated_stages": outcome.invalidated_stages,
                "expected_invalidated_stages": outcome.expected_invalidated_stages,
                "reused_count": len(outcome.reused_stages),
                "invalidated_count": len(outcome.invalidated_stages),
                "stage_summaries": outcome.stage_summaries,
                "final_png_sha256": outcome.final_png_sha256,
                "final_png_path": str(self._current_output / "final.png"),
                "max_workers_configured": outcome.max_workers_configured,
                "max_workers_observed": outcome.max_workers_observed,
                "stage_observed_workers": outcome.stage_observed_workers,
                "progress": {
                    "stage": "complete",
                    "stage_index": 4,
                    "total_stages": 4,
                    "completed_rows": self.sample.config.height,
                    "total_rows": self.sample.config.height,
                    "progress": 1.0,
                    "params_match_requested": True,
                    "affected_region": self._affected_region(operation, outcome.params),
                },
            }
        )
        return response

    def _affected_region(self, operation: str, params: RenderParams) -> dict[str, Any]:
        if operation in {
            "set_radius",
            "set_repair_radius",
            "radius",
            "set_iterations",
            "set_repair_iterations",
        }:
            return {"kind": "mask", **params.mask.as_dict()}
        if operation in {
            "set_style_weights",
            "style_weights",
            "set_quant_levels",
            "quant_levels",
        }:
            return {
                "kind": "full_image",
                "x": 0,
                "y": 0,
                "width": self.sample.config.width,
                "height": self.sample.config.height,
            }
        return {"kind": "mask", **params.mask.as_dict()}


class RunController:
    def __init__(
        self,
        sample_root: Path,
        output_root: Path,
        mode: str = "interactive",
    ) -> None:
        if mode not in {"interactive", "sync"}:
            raise StudioError(E_CONFIG_INVALID, "mode must be interactive or sync")
        self.sample_root = sample_root
        self.output_root = output_root
        self.mode = mode
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.renderer = Renderer(output_root / "cache")
        self.snapshots = SnapshotStore(output_root)
        self.snapshots.cleanup_temporary_directories()
        self.stop_event = threading.Event()
        self.started_at = time.perf_counter()
        self.run_id = (
            datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S")
            + f"-{os.getpid()}-{int(self.started_at * 1000) % 100000:05d}"
        )

    def run(self) -> dict[str, Any]:
        samples, load_failures = load_sample_set(self.sample_root)
        report_samples: list[dict[str, Any]] = []
        all_records: list[dict[str, Any]] = []
        for sample in samples:
            runner = SampleRunner(
                sample,
                self.output_root,
                self.renderer,
                self.snapshots,
                self.mode,
                self.stop_event,
            )
            sample_record: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "status": "ok",
                "config_sha256": sample.config.fingerprint(),
                "sample_fingerprint": sample.sample_fingerprint(),
            }
            try:
                sample_record["bootstrap"] = runner.bootstrap()
                records = runner.run(sample.commands)
                sample_record["commands"] = records
                sample_record["snapshots"] = runner.snapshot_records
                sample_record["completed_command_cursor"] = runner.completed_cursor
                sample_record["final"] = (
                    runner.latest_outcome.preview_summary()
                    if runner.latest_outcome is not None
                    else None
                )
                sample_record["final_png_path"] = str(
                    self.output_root / "current" / sample.sample_id / "final.png"
                )
                sample_record["failures"] = runner.failures
                all_records.extend(records)
            except StudioError as error:
                sample_record["status"] = "failed"
                sample_record["failure"] = error.as_failure()
                load_failures.append(failure_item(sample.sample_id, error))
            except Exception as error:
                sample_record["status"] = "failed"
                sample_record["failure"] = {
                    "code": E_RENDER_FAILED,
                    "message": f"unexpected render failure: {error}",
                    "details": {},
                }
                load_failures.append(
                    failure_item(
                        sample.sample_id,
                        StudioError(E_RENDER_FAILED, f"unexpected render failure: {error}"),
                    )
                )
            report_samples.append(sample_record)

        duration_ms = (time.perf_counter() - self.started_at) * 1000.0
        business_codes: dict[str, int] = {}
        for record in all_records:
            code = str(record.get("business_code", OK))
            business_codes[code] = business_codes.get(code, 0) + 1
        load_failures.sort(key=lambda item: (item.get("sample_id", ""), item.get("code", ""), item.get("message", "")))
        report = {
            "schema_version": 1,
            "run_id": self.run_id,
            "mode": self.mode,
            "sample_root": str(self.sample_root.resolve()),
            "output_root": str(self.output_root.resolve()),
            "duration_ms": round(duration_ms, 3),
            "samples": report_samples,
            "failures": load_failures,
            "summary": {
                "samples_loaded": len(samples),
                "samples_failed": len(report_samples) - sum(1 for item in report_samples if item["status"] == "ok"),
                "commands_executed": len(all_records),
                "business_codes": business_codes,
                "run_failure_count": len(load_failures),
            },
        }
        report_path = self.output_root / "reports" / f"diagnostics-{self.run_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_bytes(canonical_json(report))
        os.replace(temporary, report_path)
        report["report_path"] = str(report_path)
        self.renderer.close()
        return report


def install_graceful_signal_handlers(stop_event: threading.Event) -> dict[int, Any]:
    original: dict[int, Any] = {}

    def handle_signal(signum: int, frame: Any) -> None:
        del signum, frame
        stop_event.set()

    for signal_name in ("SIGINT", "SIGTERM"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is None:
            continue
        try:
            original[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, handle_signal)
        except (ValueError, OSError):
            continue
    return original


def restore_signal_handlers(original: Mapping[int, Any]) -> None:
    for signal_number, handler in original.items():
        try:
            signal.signal(signal_number, handler)
        except (ValueError, OSError):
            continue


def recover_output(output_root: Path) -> dict[str, Any]:
    snapshots = SnapshotStore(output_root)
    snapshot_root = output_root / "snapshots"
    sample_ids: list[str] = []
    if snapshot_root.is_dir():
        sample_ids = sorted(path.name for path in snapshot_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    recovered: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        recovered.append(
            snapshots.recover(sample_id, output_root / "recovered" / sample_id)
        )
    report = {
        "schema_version": 1,
        "status": "recovered" if any(item["status"] == "recovered" for item in recovered) else "no_valid_snapshot",
        "recovered_versions": [
            {"sample_id": item["sample_id"], "recovered_version": item["recovered_version"]}
            for item in recovered
            if item["status"] == "recovered"
        ],
        "samples": recovered,
        "exit_report_written": False,
    }
    report_path = output_root / "recovered" / "recovery-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_bytes(canonical_json(report))
    os.replace(temporary, report_path)
    report["report_path"] = str(report_path)
    return report


def render_full_reference(sample: LoadedSample, cache_root: Path, params: RenderParams) -> RenderOutcome:
    renderer = Renderer(cache_root)
    try:
        return renderer.render(sample, params, command_index=999999)
    finally:
        renderer.close()
