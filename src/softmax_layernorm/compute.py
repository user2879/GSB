from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cache import (
    CallState,
    Stage,
    apply_lane_result,
    compute_lane_keys,
    dirty_lanes,
    lane_axis,
    lane_count,
    lane_row_label,
    lane_row_number,
    lane_values,
    set_stage_keys,
)
from .errors import BusinessFailure
from .ops import layer_norm_lane, softmax_lane


STAGE_ORDER = {"softmax": 1, "layernorm": 2}


@dataclass(frozen=True)
class StageTask:
    call_sequence: int
    stage_order: int
    lane: int
    call_id: str
    stage: Stage
    key: str
    values: np.ndarray

    @property
    def stable_key(self) -> tuple[int, int, int, str]:
        return (self.call_sequence, self.lane, self.stage_order, self.call_id)


@dataclass
class StageLaneResult:
    task: StageTask
    values: np.ndarray
    failure: BusinessFailure | None
    stats: dict[str, Any] | None


@dataclass
class StageExecution:
    reused: list[int] = field(default_factory=list)
    recomputed: list[int] = field(default_factory=list)
    reused_rows: list[int] = field(default_factory=list)
    recomputed_rows: list[int] = field(default_factory=list)
    failures: list[BusinessFailure] = field(default_factory=list)


@dataclass
class ExecutionReport:
    calls: dict[str, dict[str, StageExecution]] = field(default_factory=dict)

    def total_reused(self) -> int:
        return sum(
            len(stage.reused)
            for call in self.calls.values()
            for stage in call.values()
        )

    def total_recomputed(self) -> int:
        return sum(
            len(stage.recomputed)
            for call in self.calls.values()
            for stage in call.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": {
                call_id: {
                    stage: {
                        "reused_lanes": list(execution.reused),
                        "recomputed_lanes": list(execution.recomputed),
                        "reused_rows": list(execution.reused_rows),
                        "recomputed_rows": list(execution.recomputed_rows),
                    }
                    for stage, execution in stages.items()
                }
                for call_id, stages in self.calls.items()
            },
            "total_reused_lanes": self.total_reused(),
            "total_recomputed_lanes": self.total_recomputed(),
        }


def _append_execution_metadata(execution: StageExecution, state: CallState, stage: Stage) -> None:
    axis = lane_axis(state.spec, stage)
    if axis == 1:
        execution.reused_rows = list(execution.reused)
        execution.recomputed_rows = list(execution.recomputed)
    else:
        all_rows = list(range(state.spec.rows))
        execution.reused_rows = all_rows if execution.reused else []
        execution.recomputed_rows = all_rows if execution.recomputed else []


def _run_task(task: StageTask, state: CallState) -> StageLaneResult:
    call = state.spec
    axis = lane_axis(call, task.stage)
    row_label = lane_row_label(axis)
    row = lane_row_number(axis, task.lane)
    try:
        if task.stage == "softmax":
            result = softmax_lane(
                task.values,
                stabilize=call.stabilize,
                call_id=call.call_id,
                call_sequence=call.sequence,
                lane=task.lane,
                row=row,
                row_label=row_label,
            )
        else:
            result = layer_norm_lane(
                task.values,
                epsilon=call.epsilon,
                scale=call.layernorm_scale,
                call_id=call.call_id,
                call_sequence=call.sequence,
                lane=task.lane,
                row=row,
                row_label=row_label,
            )
        return StageLaneResult(
            task=task,
            values=result.values,
            failure=result.failure,
            stats=result.stats,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary for worker failures
        failure = BusinessFailure(
            code="E_INTERNAL",
            message=f"lane execution failed: {type(exc).__name__}: {exc}",
            stage=task.stage,
            call_id=call.call_id,
            call_sequence=call.sequence,
            lane=task.lane,
            row=row,
        )
        placeholder_shape = lane_values(np.zeros(call.shape, dtype=np.float32), axis, task.lane).shape
        return StageLaneResult(
            task=task,
            values=np.full(placeholder_shape, np.nan, dtype=np.float32),
            failure=failure,
            stats={
                "mean": np.float32(np.nan),
                "variance": np.float32(np.nan),
                "epsilon_applied": False,
            },
        )


def execute_dirty(
    states: list[CallState],
    tensor: np.ndarray,
    executor: Executor,
) -> ExecutionReport:
    """Execute all cache-miss lanes, then apply lane results in stable task order."""

    report = ExecutionReport()
    tasks: list[StageTask] = []
    task_states: dict[tuple[int, int, int, str], CallState] = {}
    for state in sorted(states, key=lambda item: item.spec.sequence):
        state.ensure_storage()
        call_report: dict[str, StageExecution] = {}
        for stage in ("softmax", "layernorm"):
            dirty, reused = dirty_lanes(state, tensor, stage)
            current_keys = compute_lane_keys(state, tensor, stage)
            stored_keys = state.softmax_keys if stage == "softmax" else state.layernorm_keys
            for lane in range(lane_count(state.spec, stage)):
                if stored_keys[lane] == current_keys[lane]:
                    continue
                axis = lane_axis(state.spec, stage)
                task = StageTask(
                    call_sequence=state.spec.sequence,
                    stage_order=STAGE_ORDER[stage],
                    lane=lane,
                    call_id=state.spec.call_id,
                    stage=stage,
                    key=current_keys[lane],
                    values=lane_values(tensor, axis, lane),
                )
                tasks.append(task)
                task_states[task.stable_key] = state
            execution = StageExecution(reused=list(reused), recomputed=list(dirty))
            _append_execution_metadata(execution, state, stage)
            call_report[stage] = execution
        report.calls[state.spec.call_id] = call_report

    futures = {executor.submit(_run_task, task, task_states[task.stable_key]): task for task in tasks}
    results: list[StageLaneResult] = []
    for future in as_completed(futures):
        results.append(future.result())

    for result in sorted(results, key=lambda item: item.task.stable_key):
        state = task_states[result.task.stable_key]
        apply_lane_result(
            state=state,
            stage=result.task.stage,
            lane=result.task.lane,
            values=result.values,
            key=result.task.key,
            failure=result.failure,
            stats=result.stats,
        )
        if result.failure is not None:
            report.calls[result.task.call_id][result.task.stage].failures.append(result.failure)

    for stages in report.calls.values():
        for execution in stages.values():
            execution.failures.sort(key=lambda item: item.order_key())
    return report


def recompute_all(states: list[CallState], tensor: np.ndarray, executor: Executor) -> ExecutionReport:
    for state in states:
        state.ensure_storage()
        state.softmax_keys = [None] * len(state.softmax_keys)
        state.layernorm_keys = [None] * len(state.layernorm_keys)
        state.softmax_failures.clear()
        state.layernorm_failures.clear()
    return execute_dirty(states, tensor, executor)


def create_lane_executor(max_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="softmax-ln-lane")
