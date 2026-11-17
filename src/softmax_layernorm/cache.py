from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .errors import BusinessFailure
from .jsonutil import sha256_json
from .models import CallSpec


Stage = Literal["softmax", "layernorm"]


def lane_count(call: CallSpec, stage: Stage) -> int:
    axis = call.softmax_axis if stage == "softmax" else call.layernorm_axis
    return call.rows if axis == 1 else call.cols


def lane_axis(call: CallSpec, stage: Stage) -> int:
    return call.softmax_axis if stage == "softmax" else call.layernorm_axis


def lane_values(tensor: np.ndarray, axis: int, lane: int) -> np.ndarray:
    if axis == 1:
        return np.ascontiguousarray(tensor[lane, :], dtype=np.float32)
    return np.ascontiguousarray(tensor[:, lane], dtype=np.float32)


def lane_row_label(axis: int) -> str:
    return "row" if axis == 1 else "column"


def lane_row_number(axis: int, lane: int) -> int | None:
    return lane if axis == 1 else None


def lane_content_fingerprint(tensor: np.ndarray, axis: int, lane: int) -> str:
    values = np.ascontiguousarray(lane_values(tensor, axis, lane), dtype="<f4")
    digest = hashlib.sha256()
    digest.update(b"float32-le-lane")
    digest.update(struct.pack("<I", int(values.size)))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def lane_to_rows(call: CallSpec, axis: int, lane: int) -> list[int]:
    if axis == 1:
        return [lane]
    return list(range(call.rows))


@dataclass
class CallState:
    spec: CallSpec
    softmax_result: np.ndarray | None = None
    layernorm_result: np.ndarray | None = None
    layernorm_mean: np.ndarray | None = None
    layernorm_variance: np.ndarray | None = None
    layernorm_epsilon_applied: np.ndarray | None = None
    softmax_keys: list[str | None] = field(default_factory=list)
    layernorm_keys: list[str | None] = field(default_factory=list)
    softmax_failures: dict[int, BusinessFailure] = field(default_factory=dict)
    layernorm_failures: dict[int, BusinessFailure] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ensure_storage()

    def ensure_storage(self) -> None:
        shape = self.spec.shape
        if self.softmax_result is None:
            self.softmax_result = np.zeros(shape, dtype=np.float32)
        if self.layernorm_result is None:
            self.layernorm_result = np.zeros(shape, dtype=np.float32)
        if self.layernorm_mean is None:
            self.layernorm_mean = np.zeros(shape, dtype=np.float32)
        if self.layernorm_variance is None:
            self.layernorm_variance = np.zeros(shape, dtype=np.float32)
        if self.layernorm_epsilon_applied is None:
            self.layernorm_epsilon_applied = np.zeros(shape, dtype=bool)
        for stage in ("softmax", "layernorm"):
            count = lane_count(self.spec, stage)  # type: ignore[arg-type]
            keys = self.softmax_keys if stage == "softmax" else self.layernorm_keys
            if len(keys) < count:
                keys.extend([None] * (count - len(keys)))
            elif len(keys) > count:
                del keys[count:]

    @property
    def softmax_completed(self) -> bool:
        return all(key is not None for key in self.softmax_keys)

    @property
    def layernorm_completed(self) -> bool:
        return all(key is not None for key in self.layernorm_keys)


def compute_lane_keys(
    state: CallState,
    tensor: np.ndarray,
    stage: Stage,
) -> list[str]:
    call = state.spec
    axis = lane_axis(call, stage)
    count = lane_count(call, stage)
    keys: list[str] = []
    for lane in range(count):
        content = lane_content_fingerprint(tensor, axis, lane)
        payload: dict[str, Any] = {
            "format_version": 1,
            "call_id": call.call_id,
            "call_sequence": call.sequence,
            "stage": stage,
            "lane": lane,
            "content": content,
            "axis": axis,
        }
        if stage == "softmax":
            payload["stabilize"] = call.stabilize
        else:
            payload["epsilon"] = float(call.epsilon).hex()
            payload["scale"] = float(call.layernorm_scale).hex()
        keys.append(sha256_json(payload))
    return keys


def dirty_lanes(state: CallState, tensor: np.ndarray, stage: Stage) -> tuple[list[int], list[int]]:
    current_keys = compute_lane_keys(state, tensor, stage)
    stored_keys = state.softmax_keys if stage == "softmax" else state.layernorm_keys
    dirty: list[int] = []
    reused: list[int] = []
    for lane, key in enumerate(current_keys):
        if stored_keys[lane] == key:
            reused.append(lane)
        else:
            dirty.append(lane)
    dirty.sort()
    reused.sort()
    return dirty, reused


def set_stage_keys(state: CallState, stage: Stage, keys: list[str]) -> None:
    if stage == "softmax":
        state.softmax_keys = list(keys)
    else:
        state.layernorm_keys = list(keys)


def apply_lane_result(
    state: CallState,
    stage: Stage,
    lane: int,
    values: np.ndarray,
    key: str,
    failure: BusinessFailure | None,
    stats: dict[str, Any] | None,
) -> None:
    call = state.spec
    axis = lane_axis(call, stage)
    state.ensure_storage()
    if stage == "softmax":
        assert state.softmax_result is not None
        if axis == 1:
            state.softmax_result[lane, :] = values
        else:
            state.softmax_result[:, lane] = values
        state.softmax_keys[lane] = key
        if failure is None:
            state.softmax_failures.pop(lane, None)
        else:
            state.softmax_failures[lane] = failure
        return

    assert state.layernorm_result is not None
    assert state.layernorm_mean is not None
    assert state.layernorm_variance is not None
    assert state.layernorm_epsilon_applied is not None
    assert stats is not None
    mean = np.float32(stats["mean"])
    variance = np.float32(stats["variance"])
    epsilon_applied = bool(stats["epsilon_applied"])
    if axis == 1:
        state.layernorm_result[lane, :] = values
        state.layernorm_mean[lane, :] = mean
        state.layernorm_variance[lane, :] = variance
        state.layernorm_epsilon_applied[lane, :] = epsilon_applied
    else:
        state.layernorm_result[:, lane] = values
        state.layernorm_mean[:, lane] = mean
        state.layernorm_variance[:, lane] = variance
        state.layernorm_epsilon_applied[:, lane] = epsilon_applied
    state.layernorm_keys[lane] = key
    if failure is None:
        state.layernorm_failures.pop(lane, None)
    else:
        state.layernorm_failures[lane] = failure


def active_failures(states: list[CallState]) -> list[BusinessFailure]:
    failures: list[BusinessFailure] = []
    for state in sorted(states, key=lambda item: item.spec.sequence):
        failures.extend(state.softmax_failures.values())
        failures.extend(state.layernorm_failures.values())
    return sorted(failures, key=lambda item: item.order_key())


def stage_completion(state: CallState, stage: Stage) -> tuple[int, int]:
    keys = state.softmax_keys if stage == "softmax" else state.layernorm_keys
    completed = sum(key is not None for key in keys)
    return completed, len(keys)


def affected_rows(states: list[CallState], stage: Stage, lanes: list[int], call_id: str) -> list[int]:
    state = next(item for item in states if item.spec.call_id == call_id)
    axis = lane_axis(state.spec, stage)
    rows: set[int] = set()
    for lane in lanes:
        rows.update(lane_to_rows(state.spec, axis, lane))
    return sorted(rows)
