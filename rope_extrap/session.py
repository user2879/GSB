from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .engine import DiagnosticEngine, ExecutionPlan, run_full_diagnosis
from .errors import E_COMMAND_INVALID, E_INTERACT_BUDGET_PENDING, Failure
from .loader import load_commands, load_position_config, load_samples
from .models import AttentionSample, InteractionCommand, PositionEncodingConfig, STAGES, config_summary
from .snapshot import SnapshotStore


@dataclass
class _RunRequest:
    generation: int
    command_index: int
    config: PositionEncodingConfig
    samples: dict[str, AttentionSample]
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    started: bool = False
    complete: bool = False
    plan: ExecutionPlan | None = None
    engine: DiagnosticEngine | None = None
    error: str | None = None


class InteractiveSession:
    """Latency-budgeted interactive diagnostic session.

    A single persistent worker advances the latest content-addressed run.
    The command thread only waits for its share of the budget and returns the
    actual partial/completed summary; it never estimates unfinished values.
    A subsequent poll continues the same request, while a mutating command
    supersedes it.  Exact-key results already produced by a superseded request
    remain reusable from the shared cache.
    """

    def __init__(
        self,
        config: PositionEncodingConfig,
        samples: Mapping[str, AttentionSample] | Iterable[AttentionSample],
        commands: Iterable[InteractionCommand] = (),
        *,
        loader_failures: Iterable[Failure] = (),
        command_failures: Iterable[Failure] = (),
        snapshot_store: SnapshotStore | None = None,
        stage_hooks: Mapping[str, Callable[[AttentionSample], None]] | None = None,
    ) -> None:
        self.initial_config = config
        self.config = config
        self.commands = list(commands)
        self.loader_failures = list(loader_failures)
        self.command_failures = list(command_failures)
        self.engine = DiagnosticEngine(
            config,
            samples,
            loader_failures=loader_failures,
            stage_hooks=stage_hooks,
        )
        self.snapshot_store = snapshot_store
        self.command_index = 0
        self.snapshot_version = 0
        self.last_completed_report: dict[str, Any] | None = None
        self.last_completed_report_config_fingerprint: str | None = None
        self.last_completed_summary: dict[str, Any] | None = None
        self.last_completed_config_fingerprint: str | None = None
        self.last_report: dict[str, Any] | None = None
        self.last_report_config_fingerprint: str | None = None
        self.responses: list[dict[str, Any]] = []
        self.audit_history: list[dict[str, Any]] = []

        self._condition = threading.Condition(threading.RLock())
        self._generation = 0
        self._latest_request: _RunRequest | None = None
        self._stop = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="rope-extrap-interactive-worker",
            daemon=True,
        )
        self._worker.start()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            request = self._latest_request
            if request is not None:
                request.cancel.set()
            self._condition.notify_all()
        self._worker.join(timeout=5.0)

    def __enter__(self) -> "InteractiveSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._stop:
                    request = self._latest_request
                    if request is not None and not request.started and not request.cancel.is_set():
                        request.started = True
                        break
                    self._condition.wait()
                if self._stop:
                    return
                generation = self._generation
            assert request is not None
            if request.generation != generation or request.cancel.is_set():
                request.done.set()
                continue
            # Yield the GIL once so a micro-budget command can return its
            # pending acknowledgement before this worker starts hashing keys.
            time.sleep(0.002)
            with self._condition:
                if request.cancel.is_set() or request.generation != self._generation:
                    request.done.set()
                    self._condition.notify_all()
                    continue
            worker_engine = DiagnosticEngine(
                request.config,
                request.samples,
                loader_failures=self.loader_failures,
                cache=self.engine.cache,
                stage_hooks=self.engine.stage_hooks,
                stage_failures=self.engine.stage_failures,
            )
            try:
                plan = worker_engine.build_plan(yield_callback=lambda: time.sleep(0))
                with self._condition:
                    request.plan = plan
                    request.engine = worker_engine
                complete = worker_engine.execute_plan(
                    plan,
                    cancel_check=lambda: request.cancel.is_set() or request.generation != self._generation,
                )
                request.complete = bool(complete and not request.cancel.is_set())
            except Exception as exc:  # keep the session alive, expose a diagnostic failure
                request.error = repr(exc)
                request.complete = False
            finally:
                with self._condition:
                    request.done.set()
                    self._condition.notify_all()

    def _apply_command(self, command: InteractionCommand) -> Failure | None:
        if command.sample is not None:
            if command.sample.sample_id not in self.engine.samples:
                return Failure(
                    E_COMMAND_INVALID,
                    f"cannot replace unknown sample_id: {command.sample.sample_id}",
                    stage="load",
                    sample_id=command.sample.sample_id,
                    line=command.index,
                    scope="command",
                )
            self.engine.replace_sample(command.sample)
        if command.changes:
            changed = self.config.changed(**command.changes)
            if changed.training_context_length >= changed.extrapolation_target_length:
                return Failure(
                    E_COMMAND_INVALID,
                    "training_context_length must remain below extrapolation_target_length",
                    stage="load",
                    line=command.index,
                    scope="command",
                )
            self.config = changed
            self.engine.set_config(changed)
        return None

    def _new_request(self, command: InteractionCommand) -> _RunRequest:
        with self._condition:
            self._generation += 1
            if self._latest_request is not None:
                self._latest_request.cancel.set()
            request = _RunRequest(
                generation=self._generation,
                command_index=command.index,
                config=self.config,
                samples=dict(self.engine.samples),
            )
            self._latest_request = request
            self._condition.notify_all()
            return request

    def _request_for_poll(self, command: InteractionCommand) -> _RunRequest:
        with self._condition:
            request = self._latest_request
            if request is not None and request.generation == self._generation:
                return request
        return self._new_request(command)

    def apply_command(self, command: InteractionCommand) -> dict[str, Any]:
        started = time.monotonic()
        self.command_index = command.index
        command_failure = self._apply_command(command)
        if command_failure is not None:
            response = self._error_response(command, command_failure, started)
            self.responses.append(response)
            return response

        budget_ms = command.budget_ms or self.config.interaction_budget_ms
        if command.command_type in {"poll", "advance", "resume"} and command.changes == {} and command.sample is None:
            request = self._request_for_poll(command)
        else:
            request = self._new_request(command)

        budget_seconds = budget_ms / 1000.0
        # Keep headroom for response assembly and Python scheduler handoff.
        reserve_seconds = min(0.005, budget_seconds * 0.50)
        deadline = started + max(0.0, budget_seconds - reserve_seconds)
        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 0.0 and not request.done.is_set():
            request.done.wait(remaining)
        done = request.done.is_set() and request.generation == self._generation

        fast_complete = False
        report: dict[str, Any] | None = None
        if done and request.complete:
            fast_complete = True
            # Building a full report is intentionally avoided for micro-budgets.
            # The returned summary is still computed from real cached results.
            if budget_ms >= 50 and request.engine is not None and request.plan is not None:
                report = request.engine.build_report(
                    plan=request.plan,
                    execution_audit=request.plan.audit_dict(),
                    run_metadata={
                        "mode": "interactive",
                        "command_index": command.index,
                        "command_id": command.command_id,
                        "budget_ms": budget_ms,
                        "elapsed_ms": (time.monotonic() - started) * 1000.0,
                        "complete_after_command": True,
                    },
                )
                self.last_report = report
                self.last_report_config_fingerprint = request.config.config_fingerprint()
                self.last_completed_report = report
                self.last_completed_report_config_fingerprint = request.config.config_fingerprint()
                summary = copy.deepcopy(report["summary"])
            else:
                summary = self._fast_summary(request)
        elif done:
            summary = self._fast_summary(request)
        else:
            summary = self._fast_summary(request)

        is_complete = bool(fast_complete and summary.get("is_complete"))
        if is_complete:
            self.last_completed_summary = copy.deepcopy(summary)
            self.last_completed_config_fingerprint = request.config.config_fingerprint()
            business_codes = ["OK"]
        else:
            business_codes = [E_INTERACT_BUDGET_PENDING]

        audit = request.plan.audit_dict() if request.plan is not None else {
            "reused": [],
            "recomputed": [],
            "failed": [],
            "skipped": [],
            "summary": {
                "reused_stage_count": 0,
                "recomputed_stage_count": 0,
                "failed_stage_count": 0,
                "skipped_stage_count": 0,
                "reused_samples": [],
                "recomputed_samples": [],
                "failed_samples": [],
            },
        }
        audit["command_index"] = command.index
        audit["command_id"] = command.command_id
        audit["complete_after_command"] = is_complete
        self.audit_history.append(copy.deepcopy(audit))
        elapsed_ms = (time.monotonic() - started) * 1000.0
        response = {
            "command_index": command.index,
            "command_id": command.command_id,
            "command_type": command.command_type,
            "budget_ms": budget_ms,
            "elapsed_ms": elapsed_ms,
            "within_budget": elapsed_ms <= budget_ms,
            "pending": not is_complete,
            "business_codes": business_codes,
            "summary": copy.deepcopy(summary),
            "last_completed_summary": copy.deepcopy(self.last_completed_summary),
            "config": config_summary(request.config),
            "pending_queue": self._pending_queue(request),
            "reuse_and_recompute": audit,
            "failures": self._request_failures(request),
        }
        if report is not None:
            response["report_summary"] = copy.deepcopy(report["summary"])
        self.responses.append(response)
        if report is not None:
            self._maybe_commit_snapshot(report)
        return response

    def _error_response(
        self,
        command: InteractionCommand,
        failure: Failure,
        started: float,
    ) -> dict[str, Any]:
        return {
            "command_index": command.index,
            "command_id": command.command_id,
            "command_type": command.command_type,
            "budget_ms": command.budget_ms or self.config.interaction_budget_ms,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "pending": False,
            "business_codes": [failure.code],
            "summary": copy.deepcopy(self.last_completed_summary),
            "last_completed_summary": copy.deepcopy(self.last_completed_summary),
            "config": config_summary(self.config),
            "pending_queue": None,
            "reuse_and_recompute": None,
            "failures": [failure.to_dict()],
        }

    def _fast_summary(self, request: _RunRequest) -> dict[str, Any]:
        plan = request.plan
        completed = 0
        failed = 0
        violations = 0
        incomplete = 0
        if plan is None:
            incomplete = len(request.samples)
        else:
            for sample_id in sorted(request.samples):
                keys = plan.stage_keys_by_sample.get(sample_id)
                if keys is None:
                    incomplete += 1
                    continue
                key_map = keys.as_dict()
                stage_failure = None
                for stage in STAGES:
                    failure = self.engine.stage_failures.get((sample_id, stage, key_map[stage]))
                    if failure is not None:
                        stage_failure = failure
                        break
                if stage_failure is not None:
                    failed += 1
                    continue
                entries = [self.engine.cache.get(stage, key_map[stage]) for stage in STAGES]
                if all(entry is not None for entry in entries):
                    completed += 1
                    trace_entry = entries[2]
                    if trace_entry is not None and trace_entry.result.violation is not None:
                        violations += 1
                else:
                    incomplete += 1
        total = len(request.samples)
        resolved = completed + failed
        return {
            "completed_samples": completed,
            "failed_samples": failed,
            "resolved_samples": resolved,
            "total_valid_samples": total,
            "incomplete_samples": incomplete,
            "progress": (float(resolved) / total) if total else 1.0,
            "current_scheme": request.config.scheme,
            "extrapolation_over_tolerance_samples": violations,
            "is_complete": bool(request.done.is_set() and request.complete and incomplete == 0),
        }

    def _request_failures(self, request: _RunRequest) -> list[dict[str, Any]]:
        failures = list(self.loader_failures)
        if request.plan is not None:
            for sample_id in sorted(request.plan.stage_keys_by_sample):
                key_map = request.plan.stage_keys_by_sample[sample_id].as_dict()
                for stage in STAGES:
                    failure = self.engine.stage_failures.get((sample_id, stage, key_map[stage]))
                    if failure is not None:
                        failures.append(failure)
                        break
        failures.extend(request.plan.failed if request.plan is not None else [])
        unique: dict[tuple[object, ...], Failure] = {}
        for failure in failures:
            key = (failure.code, failure.stage, failure.sample_id, failure.line, failure.scope, failure.message)
            unique[key] = failure
        return [failure.to_dict() for failure in sorted(unique.values(), key=lambda item: item.sort_key())]

    def _pending_queue(self, request: _RunRequest) -> dict[str, Any]:
        plan = request.plan
        if plan is None:
            return {
                "status": "queued",
                "remaining_stage_tasks": None,
                "remaining_by_stage": None,
                "next_stable_keys": [],
                "blocked_samples": [],
            }
        remaining = plan.remaining_jobs()
        per_stage = {stage: 0 for stage in STAGES}
        for job in remaining:
            per_stage[job.stage] += 1
        return {
            "status": "running" if not request.done.is_set() else "idle",
            "remaining_stage_tasks": len(remaining),
            "remaining_by_stage": per_stage,
            "next_stable_keys": [job.stable_sort_key for job in remaining[:8]],
            "blocked_samples": sorted(plan.blocked_samples),
        }

    def _maybe_commit_snapshot(self, report: dict[str, Any]) -> None:
        if self.snapshot_store is None:
            return
        interval = self.config.snapshot_commit_interval
        if self.command_index % interval != 0:
            return
        self.snapshot_version += 1
        completed = [row["sample_id"] for row in report["samples"]]
        failed = sorted({item["sample_id"] for item in report["failures"] if item.get("sample_id")})
        self.snapshot_store.commit(
            self.engine,
            global_version=self.snapshot_version,
            completed_samples=completed,
            failed_samples=failed,
            metadata={
                "command_index": self.command_index,
                "pending": not report["summary"]["is_complete"],
            },
        )

    def run_all(self) -> list[dict[str, Any]]:
        for command in self.commands:
            self.apply_command(command)
        return list(self.responses)

    def drain(self, *, budget_ms: int = 120_000, max_polls: int = 100) -> dict[str, Any] | None:
        last_response: dict[str, Any] | None = None
        for _ in range(max_polls):
            if (
                self.last_completed_summary is not None
                and self.last_completed_config_fingerprint == self.config.config_fingerprint()
                and self.last_completed_summary.get("is_complete")
            ):
                return last_response
            poll = InteractionCommand(
                index=self.command_index + 1,
                command_id=f"drain-{self.command_index + 1}",
                command_type="poll",
                changes={},
                budget_ms=budget_ms,
            )
            last_response = self.apply_command(poll)
            if not last_response["pending"]:
                return last_response
        return last_response

    def final_report(self) -> dict[str, Any]:
        current_fingerprint = self.config.config_fingerprint()
        if (
            self.last_completed_report is not None
            and self.last_completed_report_config_fingerprint == current_fingerprint
        ):
            report = copy.deepcopy(self.last_completed_report)
        else:
            request = self._latest_request
            if request is not None and request.engine is not None and request.plan is not None:
                report = request.engine.build_report(
                    plan=request.plan,
                    execution_audit=request.plan.audit_dict(),
                    run_metadata={"mode": "interactive", "partial": True},
                )
            elif self.last_report is not None and self.last_report_config_fingerprint == current_fingerprint:
                report = copy.deepcopy(self.last_report)
            else:
                report = self.engine.build_report(run_metadata={"mode": "interactive", "no_commands": True})
        report["command_audit_history"] = copy.deepcopy(self.audit_history)
        report["loader_failures"] = [failure.to_dict() for failure in self.loader_failures]
        report["command_load_failures"] = [failure.to_dict() for failure in self.command_failures]
        interactive_counts: dict[str, int] = {}
        for response in self.responses:
            for code in response.get("business_codes", []):
                if code.startswith("E_"):
                    interactive_counts[code] = interactive_counts.get(code, 0) + 1
        report["interactive_business_code_counts"] = dict(sorted(interactive_counts.items()))
        return report


def run_command_sequence(
    config_path: str | Path,
    samples_path: str | Path,
    commands_path: str | Path,
    *,
    snapshot_dir: str | Path | None = None,
    drain_pending: bool = True,
    drain_budget_ms: int = 120_000,
) -> dict[str, Any]:
    config, config_failures = load_position_config(config_path)
    if config is None:
        return {
            "responses": [],
            "final_report": None,
            "setup_failures": [failure.to_dict() for failure in config_failures],
            "business_codes": sorted({failure.code for failure in config_failures}),
        }
    samples, sample_failures = load_samples(
        samples_path,
        dimension=config.dimension,
        target_length=config.extrapolation_target_length,
    )
    commands, command_failures = load_commands(commands_path, config=config)
    snapshot_store = SnapshotStore(snapshot_dir) if snapshot_dir is not None else None
    session = InteractiveSession(
        config,
        samples,
        commands,
        loader_failures=sample_failures,
        command_failures=command_failures,
        snapshot_store=snapshot_store,
    )
    responses: list[dict[str, Any]] = []
    try:
        responses = session.run_all()
        if drain_pending:
            drain_response = session.drain(budget_ms=drain_budget_ms)
            if drain_response is not None:
                responses.append(drain_response)
        report = session.final_report()
        business_codes = sorted(
            {
                code
                for response in responses
                for code in response.get("business_codes", [])
                if code.startswith("E_")
            }
        )
        if report.get("failures"):
            business_codes.extend(
                sorted({failure["code"] for failure in report["failures"] if failure["code"] not in business_codes})
            )
        return {
            "responses": responses,
            "final_report": report,
            "setup_failures": [failure.to_dict() for failure in config_failures + sample_failures + command_failures],
            "business_codes": sorted(set(business_codes)),
            "current_config": config_summary(session.config),
            "current_samples": {
                sample_id: {
                    **sample.to_input_dict(),
                    "feature_line": sample.feature_line,
                    "source_ref": sample.source_ref,
                }
                for sample_id, sample in sorted(session.engine.samples.items())
            },
        }
    finally:
        session.close()
