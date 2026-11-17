from __future__ import annotations

import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cache import CallState, active_failures
from .compute import ExecutionReport, create_lane_executor, execute_dirty, recompute_all
from .errors import BusinessError, BusinessFailure
from .jsonutil import sanitize_json
from .loader import coerce_float, tensor_fingerprint
from .models import CallSpec, CommandSpec, LoadedRun, RunConfig
from .reference import compute_reference_arrays
from .snapshots import SnapshotStore, write_run_outputs
from .summary import build_summary


@dataclass
class CommandRecord:
    index: int
    command_id: str
    command_type: str
    status: str = "queued"
    initial_status: str = "queued"
    business_codes: list[str] = field(default_factory=list)
    response_latency_ms: float = 0.0
    over_budget: bool = False
    completion_latency_ms: float | None = None
    reused_lanes: int = 0
    recomputed_lanes: int = 0
    reused_rows: int = 0
    recomputed_rows: int = 0
    affected_rows: list[int] = field(default_factory=list)
    snapshot_version: int | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_index": self.index,
            "command_id": self.command_id,
            "command_type": self.command_type,
            "status": self.status,
            "initial_status": self.initial_status,
            "business_codes": list(self.business_codes),
            "response_latency_ms": self.response_latency_ms,
            "over_budget": self.over_budget,
            "completion_latency_ms": self.completion_latency_ms,
            "reused_lanes": self.reused_lanes,
            "recomputed_lanes": self.recomputed_lanes,
            "reused_rows": self.reused_rows,
            "recomputed_rows": self.recomputed_rows,
            "affected_rows": list(self.affected_rows),
            "snapshot_version": self.snapshot_version,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CommandRecord":
        return cls(
            index=int(raw["command_index"]),
            command_id=str(raw["command_id"]),
            command_type=str(raw.get("command_type", "set_params")),
            status=str(raw.get("status", "ok")),
            initial_status=str(raw.get("initial_status", raw.get("status", "ok"))),
            business_codes=list(raw.get("business_codes", [])),
            response_latency_ms=float(raw.get("response_latency_ms", 0.0)),
            over_budget=bool(raw.get("over_budget", False)),
            completion_latency_ms=raw.get("completion_latency_ms"),
            reused_lanes=int(raw.get("reused_lanes", 0)),
            recomputed_lanes=int(raw.get("recomputed_lanes", 0)),
            reused_rows=int(raw.get("reused_rows", 0)),
            recomputed_rows=int(raw.get("recomputed_rows", 0)),
            affected_rows=[int(item) for item in raw.get("affected_rows", [])],
            snapshot_version=raw.get("snapshot_version"),
            error_message=raw.get("error_message"),
        )


@dataclass
class FullExecution:
    calls: list[CallSpec]
    tensor: np.ndarray
    states: list[CallState]
    execution: ExecutionReport
    failures: list[BusinessFailure]
    summary: dict[str, Any]

    def call_state(self, call_id: str) -> CallState:
        return next(state for state in self.states if state.spec.call_id == call_id)


def _command_failure(
    code: str,
    message: str,
    *,
    command: CommandSpec | None = None,
    details: dict[str, Any] | None = None,
) -> BusinessFailure:
    return BusinessFailure(
        code=code,
        message=message,
        stage="command",
        command_index=command.index if command else None,
        command_id=command.command_id if command else None,
        details=details or {},
    )


def _replacement_values(raw: dict[str, Any], count: int, cols: int, command: CommandSpec) -> np.ndarray:
    try:
        values = raw.get("values")
        if values is None and isinstance(raw.get("replacement"), dict):
            replacement = raw["replacement"]
            seed = replacement.get("seed", raw.get("seed"))
            distribution = replacement.get("distribution", "normal")
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError("replacement.seed must be an integer")
            rng = np.random.Generator(np.random.PCG64(seed))
            if distribution == "normal":
                loc = np.float32(replacement.get("loc", 0.0))
                scale = np.float32(replacement.get("scale", 1.0))
                values = rng.standard_normal((count, cols), dtype=np.float32) * scale + loc
            elif distribution == "uniform":
                low = replacement.get("low", -1.0)
                high = replacement.get("high", 1.0)
                values = rng.uniform(low, high, size=(count, cols)).astype(np.float32)
            elif distribution == "constant":
                values = np.full((count, cols), np.float32(coerce_float(replacement.get("value", 0.0))))
            else:
                raise ValueError("replacement.distribution must be normal, uniform, or constant")
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"values must contain exactly {count} rows")
        matrix = np.empty((count, cols), dtype=np.float32)
        for row_index, row in enumerate(values):
            if not isinstance(row, list) or len(row) != cols:
                raise ValueError(f"values[{row_index}] must contain exactly {cols} values")
            for col_index, item in enumerate(row):
                matrix[row_index, col_index] = np.float32(coerce_float(item))
        return matrix
    except (TypeError, ValueError, OverflowError) as exc:
        raise BusinessError(
            _command_failure(
                "E_SHAPE_MISMATCH",
                f"row replacement values are invalid: {exc}",
                command=command,
            )
        ) from exc


class Engine:
    """Serial command state machine with an isolated lane thread pool."""

    def __init__(
        self,
        loaded: LoadedRun,
        commands: list[CommandSpec],
        commands_digest: str,
        *,
        output_dir: str | Path | None = None,
        fresh: bool = False,
    ) -> None:
        self.loaded = loaded
        self.config = loaded.config
        self.commands = commands
        self.commands_digest = commands_digest
        self.output_dir = Path(output_dir or self.config.snapshot_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store = SnapshotStore(self.output_dir)
        self._lock = threading.RLock()
        self._commit_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._finalized = False
        self._exit_reason = "completed"
        self._lane_executor = create_lane_executor(self.config.thread_pool_workers)
        self._command_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="softmax-ln-command"
        )
        self.command_records: dict[int, CommandRecord] = {}
        self.command_errors: list[BusinessFailure] = []
        self.reuse_recompute_history: list[dict[str, Any]] = []
        self.event_business_code_counts: Counter[str] = Counter()
        self.last_affected_rows: list[int] = []
        self.snapshot_version = 0
        self.global_version = 0
        self.next_command_index = 0
        self.reference_comparison: dict[str, Any] = {}
        self.report_paths: tuple[Path, Path] | None = None

        resume_data = None
        if not fresh and self.config.resume:
            resume_data = self.store.load_current(
                config_hash=loaded.config_hash,
                source_input_fingerprint=loaded.source_input_fingerprint,
                commands_digest=commands_digest,
            )

        if resume_data is not None:
            self.tensor = np.asarray(resume_data.tensor, dtype=np.float32).copy()
            self.states = resume_data.call_states
            self.global_version = resume_data.global_version
            self.next_command_index = resume_data.next_command_index
            self.snapshot_version = resume_data.snapshot_version
            state = resume_data.state
            self.command_records = {
                int(item["command_index"]): CommandRecord.from_dict(item)
                for item in state.get("command_records", [])
            }
            self.command_errors = [
                BusinessFailure.from_dict(item) for item in state.get("command_errors", [])
            ]
            self.reuse_recompute_history = list(state.get("reuse_recompute_history", []))
            self.event_business_code_counts = Counter(
                {str(key): int(value) for key, value in state.get("event_business_code_counts", {}).items()}
            )
            self.last_affected_rows = [int(item) for item in state.get("last_affected_rows", [])]
            self.reference_comparison = dict(state.get("reference_comparison", {}))
        else:
            self.tensor = np.asarray(loaded.tensor, dtype=np.float32).copy()
            self.states = [CallState(spec=call) for call in loaded.calls]
            self.last_affected_rows = sorted({row for call in loaded.calls for row in range(call.rows)})
            execution = execute_dirty(self.states, self.tensor, self._lane_executor)
            self.reference_comparison = {}
            for failure in active_failures(self.states):
                self.event_business_code_counts[failure.code] += 1
            self.latest_summary = build_summary(self.states, self.last_affected_rows)
            self._commit_snapshot_locked_state()

        self.latest_summary = build_summary(self.states, self.last_affected_rows)
        if self.next_command_index > len(self.commands):
            raise BusinessError(
                BusinessFailure(
                    code="E_CONFIG_MISMATCH",
                    message="snapshot command cursor is beyond the supplied command stream",
                    stage="input",
                    details={
                        "next_command_index": self.next_command_index,
                        "commands": len(self.commands),
                    },
                )
            )

    def _state_payload_locked(self) -> dict[str, Any]:
        failures = active_failures(self.states)
        active_counts = Counter(failure.code for failure in failures)
        return {
            "current_input_fingerprint": tensor_fingerprint(self.tensor),
            "next_command_index": self.next_command_index,
            "latest_summary": copy.deepcopy(self.latest_summary),
            "command_records": [
                self.command_records[index].to_dict() for index in sorted(self.command_records)
            ],
            "command_errors": [failure.to_dict() for failure in self.command_errors],
            "reuse_recompute_history": copy.deepcopy(self.reuse_recompute_history),
            "event_business_code_counts": dict(sorted(self.event_business_code_counts.items())),
            "active_business_code_counts": dict(sorted(active_counts.items())),
            "last_affected_rows": list(self.last_affected_rows),
            "reference_comparison": copy.deepcopy(self.reference_comparison),
        }

    def _commit_snapshot_locked_state(self, reason: str = "checkpoint") -> int:
        # Build the payload under the state lock, then release it during filesystem I/O so
        # an interactive reader never waits on a snapshot commit.
        with self._lock:
            state_payload = self._state_payload_locked()
            failures = active_failures(self.states)
            states_snapshot = self.states
            tensor_snapshot = self.tensor
            reuse_payload = {
                "latest": copy.deepcopy(self.reuse_recompute_history[-1])
                if self.reuse_recompute_history
                else None,
                "history": copy.deepcopy(self.reuse_recompute_history),
            }
            failures_payload = [failure.to_dict() for failure in failures]

        reference_arrays, reference_comparison = compute_reference_arrays(
            states_snapshot, tensor_snapshot
        )
        with self._lock:
            self.reference_comparison = reference_comparison
            state_payload["reference_comparison"] = copy.deepcopy(reference_comparison)

        with self._commit_lock:
            snapshot_version, _ = self.store.commit(
                tensor=tensor_snapshot,
                states=states_snapshot,
                global_version=self.global_version,
                next_command_index=self.next_command_index,
                config_hash=self.loaded.config_hash,
                source_input_fingerprint=self.loaded.source_input_fingerprint,
                commands_digest=self.commands_digest,
                state_payload=state_payload,
                reuse_recompute_payload=reuse_payload,
                failures_payload=failures_payload,
                reference_arrays=reference_arrays,
            )
        with self._lock:
            self.snapshot_version = snapshot_version
        return snapshot_version

    def _targets(self, raw: dict[str, Any]) -> list[CallSpec]:
        raw_targets = raw.get("call_ids")
        if raw_targets is None and raw.get("call_id") is not None:
            raw_targets = [raw["call_id"]]
        if raw_targets is None:
            return [state.spec for state in self.states]
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        if not isinstance(raw_targets, list) or not all(isinstance(item, str) for item in raw_targets):
            raise BusinessError(_command_failure("E_CONFIG_INVALID", "call_ids must be a list of strings"))
        by_id = {state.spec.call_id: state.spec for state in self.states}
        missing = [item for item in raw_targets if item not in by_id]
        if missing:
            raise BusinessError(
                _command_failure(
                    "E_CONFIG_INVALID",
                    f"unknown call ids: {missing}",
                    details={"missing_call_ids": missing},
                )
            )
        return [by_id[item] for item in raw_targets]

    def _preview_affected_rows(self, command: CommandSpec) -> list[int]:
        raw = command.raw
        command_type = command.command_type
        try:
            targets = self._targets(raw)
        except BusinessError:
            targets = [state.spec for state in self.states]
        target_ids = {call.call_id for call in targets}
        rows: set[int] = set()
        if command_type == "replace_rows":
            start = raw.get("start", raw.get("start_row"))
            end = raw.get("end", raw.get("end_row"))
            if isinstance(start, int) and isinstance(end, int):
                for state in self.states:
                    selected_by_data = True
                    call = state.spec
                    if call.softmax_axis == 1 and call.layernorm_axis == 1 and selected_by_data:
                        rows.update(range(max(0, start), min(call.rows, end)))
                    else:
                        rows.update(range(call.rows))
            return sorted(rows)
        for state in self.states:
            if state.spec.call_id in target_ids:
                rows.update(range(state.spec.rows))
        return sorted(rows)

    def _apply_command(self, command: CommandSpec) -> None:
        raw = command.raw
        command_type = command.command_type
        targets = self._targets(raw)
        target_ids = {call.call_id for call in targets}

        if command_type == "replace_rows":
            start = raw.get("start", raw.get("start_row"))
            end = raw.get("end", raw.get("end_row"))
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or end > self.tensor.shape[0]
            ):
                raise BusinessError(
                    _command_failure(
                        "E_SHAPE_MISMATCH",
                        "replace_rows requires 0 <= start < end <= rows",
                        command=command,
                        details={"start": start, "end": end, "rows": self.tensor.shape[0]},
                    )
                )
            matrix = _replacement_values(raw, end - start, self.tensor.shape[1], command)
            self.tensor[start:end, :] = matrix
            return

        updates: dict[str, Any] = {}
        if command_type == "set_epsilon":
            updates["epsilon"] = raw.get(
                "epsilon", raw.get("new_epsilon", raw.get("value"))
            )
        elif command_type == "set_softmax_axis":
            updates["softmax_axis"] = raw.get(
                "softmax_axis", raw.get("axis", raw.get("value"))
            )
        elif command_type == "set_layernorm_axis":
            updates["layernorm_axis"] = raw.get(
                "layernorm_axis", raw.get("axis", raw.get("value"))
            )
        elif command_type == "set_stabilize":
            updates["stabilize"] = raw.get("enabled", raw.get("stabilize", raw.get("value")))
        elif command_type == "set_params":
            for key in ("epsilon", "softmax_axis", "layernorm_axis", "stabilize", "layernorm_scale"):
                if key in raw:
                    updates[key] = raw[key]
        else:
            raise BusinessError(
                _command_failure(
                    "E_CONFIG_INVALID", f"unsupported command type: {command_type}", command=command
                )
            )

        if not updates:
            raise BusinessError(
                _command_failure("E_CONFIG_INVALID", "command does not contain a parameter update", command=command)
            )

        if "epsilon" in updates:
            epsilon = updates["epsilon"]
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(float(epsilon))
                or float(epsilon) <= 0.0
            ):
                raise BusinessError(
                    _command_failure(
                        "E_EPS_INVALID",
                        "epsilon must be finite and positive",
                        command=command,
                        details={"epsilon": epsilon},
                    )
                )
            updates["epsilon"] = float(epsilon)

        for axis_key in ("softmax_axis", "layernorm_axis"):
            if axis_key in updates:
                axis = updates[axis_key]
                if (
                    not isinstance(axis, int)
                    or isinstance(axis, bool)
                    or axis < -2
                    or axis > 1
                ):
                    raise BusinessError(
                        _command_failure(
                            "E_AXIS_OUT_OF_RANGE",
                            f"{axis_key} must be in [-2, 1]",
                            command=command,
                            details={axis_key: axis},
                        )
                    )
                updates[axis_key] = axis % 2

        if "stabilize" in updates and not isinstance(updates["stabilize"], bool):
            raise BusinessError(
                _command_failure("E_CONFIG_INVALID", "stabilize must be a boolean", command=command)
            )

        if "layernorm_scale" in updates:
            scale = updates["layernorm_scale"]
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
                or float(scale) <= 0.0
            ):
                raise BusinessError(
                    _command_failure(
                        "E_CONFIG_INVALID", "layernorm_scale must be finite and positive", command=command
                    )
                )
            updates["layernorm_scale"] = float(scale)

        for state in self.states:
            if state.spec.call_id in target_ids:
                state.spec = replace(state.spec, **updates)
                state.ensure_storage()

    def _record_failure_event(self, failure: BusinessFailure) -> None:
        self.event_business_code_counts[failure.code] += 1

    def _execute_command(self, command: CommandSpec, done_event: threading.Event) -> None:
        start = time.monotonic()
        with self._lock:
            record = self.command_records.setdefault(
                command.index,
                CommandRecord(
                    index=command.index,
                    command_id=command.command_id,
                    command_type=command.command_type,
                ),
            )
            record.status = "running"

        command_error: BusinessFailure | None = None
        execution: ExecutionReport | None = None
        try:
            self._apply_command(command)
            execution = execute_dirty(self.states, self.tensor, self._lane_executor)
        except BusinessError as exc:
            command_error = exc.failures[0]
        except Exception as exc:  # pragma: no cover - defensive command boundary
            command_error = _command_failure(
                "E_INTERNAL",
                f"command execution failed: {type(exc).__name__}: {exc}",
                command=command,
            )

        completion_ms = (time.monotonic() - start) * 1000.0
        with self._lock:
            self.global_version += 1
            self.next_command_index = max(self.next_command_index, command.index + 1)
            if command_error is not None:
                record.status = "error"
                record.business_codes = [command_error.code]
                record.error_message = command_error.message
                record.affected_rows = self._preview_affected_rows(command)
                self.command_errors.append(command_error)
                self._record_failure_event(command_error)
            else:
                assert execution is not None
                record.status = "ok"
                record.business_codes = []
                record.reused_lanes = execution.total_reused()
                record.recomputed_lanes = execution.total_recomputed()
                reused_rows: set[int] = set()
                recomputed_rows: set[int] = set()
                for stages in execution.calls.values():
                    for stage_execution in stages.values():
                        reused_rows.update(stage_execution.reused_rows)
                        recomputed_rows.update(stage_execution.recomputed_rows)
                record.reused_rows = len(reused_rows)
                record.recomputed_rows = len(recomputed_rows)
                record.affected_rows = sorted(recomputed_rows)
                self.last_affected_rows = sorted(recomputed_rows)
                self.latest_summary = build_summary(self.states, self.last_affected_rows)
                history_item = {
                    "command_index": command.index,
                    "command_id": command.command_id,
                    "command_type": command.command_type,
                    **execution.to_dict(),
                }
                self.reuse_recompute_history.append(history_item)
                for stages in execution.calls.values():
                    for stage_execution in stages.values():
                        for failure in stage_execution.failures:
                            self._record_failure_event(failure)
            record.completion_latency_ms = completion_ms
            done_event.set()

        should_commit = (
            (command.index + 1) % self.config.snapshot_interval_commands == 0
        )
        if should_commit:
            try:
                snapshot_version = self._commit_snapshot_locked_state()
                with self._lock:
                    record.snapshot_version = snapshot_version
            except Exception as exc:  # pragma: no cover - filesystem failures are recorded
                failure = _command_failure(
                    "E_INTERNAL", f"snapshot commit failed: {type(exc).__name__}: {exc}", command=command
                )
                with self._lock:
                    record.status = "error"
                    record.business_codes = [failure.code]
                    record.error_message = failure.message
                    self.command_errors.append(failure)
                    self._record_failure_event(failure)

    def handle_command(self, command: CommandSpec) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("engine is finalized")
        if command.index < 0 or command.index >= len(self.commands):
            raise BusinessError(
                _command_failure("E_CONFIG_INVALID", "command index is out of range", command=command)
            )
        start = time.monotonic()
        done_event = threading.Event()
        with self._lock:
            if command.index in self.command_records:
                raise BusinessError(
                    _command_failure("E_CONFIG_INVALID", "command index was already submitted", command=command)
                )
            self.command_records[command.index] = CommandRecord(
                index=command.index,
                command_id=command.command_id,
                command_type=command.command_type,
            )
        self._command_executor.submit(self._execute_command, command, done_event)

        budget_ms = self.config.interaction_budget_ms
        deadline = start + budget_ms / 1000.0
        while not done_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            done_event.wait(remaining)
        completed_within_observation = done_event.is_set()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        over_budget = elapsed_ms > budget_ms

        with self._lock:
            record = self.command_records[command.index]
            record.response_latency_ms = elapsed_ms
            record.over_budget = over_budget
            if completed_within_observation:
                record.initial_status = record.status
                status = record.status
                business_codes = list(record.business_codes)
            else:
                record.initial_status = "pending"
                status = "pending"
                business_codes = ["E_INTERACT_BUDGET_PENDING"]
                self._record_failure_event(
                    _command_failure(
                        "E_INTERACT_BUDGET_PENDING",
                        "latest parameters are still being evaluated",
                        command=command,
                    )
                )
            latest_summary = copy.deepcopy(self.latest_summary)
            pending_queue = [
                item.command_id
                for item in sorted(self.command_records.values(), key=lambda item: item.index)
                if item.status in {"queued", "running"}
            ]
            preview_rows = record.affected_rows or self._preview_affected_rows(command)
            snapshot_version = self.snapshot_version

        return {
            "command_index": command.index,
            "command_id": command.command_id,
            "status": status,
            "business_codes": business_codes,
            "response_latency_ms": elapsed_ms,
            "over_budget": over_budget,
            "latest_completed_summary": latest_summary,
            "affected_rows": preview_rows,
            "pending_queue": pending_queue,
            "progress": latest_summary.get("progress", 0.0),
            "pending_progress": {
                "queued_commands": len(pending_queue),
                "completed_rows": latest_summary.get("completed_rows", 0),
            },
            "snapshot_version": snapshot_version,
        }

    def request_stop(self, reason: str = "signal") -> None:
        self._exit_reason = reason
        self._stop_event.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def finalize(self) -> dict[str, Any]:
        with self._lock:
            if self.report_paths is not None:
                return self._last_report
        self._command_executor.shutdown(wait=True)
        self._lane_executor.shutdown(wait=True)
        snapshot_version = self._commit_snapshot_locked_state(reason="final")
        with self._lock:
            failures = active_failures(self.states)
            active_counts = Counter(failure.code for failure in failures)
            combined_failures = sorted(
                failures + self.command_errors, key=lambda item: item.order_key()
            )
            report = {
                "format_version": 1,
                "mode": "incremental",
                "config_hash": self.loaded.config_hash,
                "source_input_fingerprint": self.loaded.source_input_fingerprint,
                "current_input_fingerprint": tensor_fingerprint(self.tensor),
                "input_tensor_fingerprint": tensor_fingerprint(self.tensor),
                "thread_pool_workers": self.config.thread_pool_workers,
                "interaction_budget_ms": self.config.interaction_budget_ms,
                "snapshot_interval_commands": self.config.snapshot_interval_commands,
                "snapshot_version": snapshot_version,
                "global_version": self.global_version,
                "next_command_index": self.next_command_index,
                "total_commands": len(self.commands),
                "exit_reason": "signal" if self._stop_event.is_set() else self._exit_reason,
                "summary": copy.deepcopy(self.latest_summary),
                "commands": [
                    self.command_records[index].to_dict()
                    for index in sorted(self.command_records)
                ],
                "business_code_counts": dict(sorted(self.event_business_code_counts.items())),
                "active_business_code_counts": dict(sorted(active_counts.items())),
                "failures": [failure.to_dict() for failure in failures],
                "failure_list": [failure.to_dict() for failure in combined_failures],
                "command_errors": [failure.to_dict() for failure in self.command_errors],
                "reuse_recompute": copy.deepcopy(self.reuse_recompute_history),
                "reference_comparison": copy.deepcopy(self.reference_comparison),
                "report_path": str(self.output_dir / "reports" / "latest.json"),
                "failure_path": str(self.output_dir / "failures" / "latest.json"),
            }
            report = sanitize_json(report)
            report_path, failure_path = write_run_outputs(
                self.output_dir, report, [failure.to_dict() for failure in combined_failures]
            )
            self.report_paths = (report_path, failure_path)
            self._last_report = report
            return report

    def export_final(self) -> tuple[list[CallSpec], np.ndarray]:
        with self._lock:
            return [state.spec for state in self.states], self.tensor.copy()


def compute_from_scratch(
    calls: list[CallSpec],
    tensor: np.ndarray,
    config: RunConfig,
    *,
    affected_rows: list[int] | None = None,
) -> FullExecution:
    states = [CallState(spec=call) for call in sorted(calls, key=lambda item: item.sequence)]
    executor = create_lane_executor(config.thread_pool_workers)
    try:
        execution = recompute_all(states, tensor, executor)
    finally:
        executor.shutdown(wait=True)
    failures = active_failures(states)
    affected = (
        sorted(set(affected_rows))
        if affected_rows is not None
        else sorted({row for call in calls for row in range(call.rows)})
    )
    return FullExecution(
        calls=list(calls),
        tensor=np.asarray(tensor, dtype=np.float32).copy(),
        states=states,
        execution=execution,
        failures=failures,
        summary=build_summary(states, affected),
    )
