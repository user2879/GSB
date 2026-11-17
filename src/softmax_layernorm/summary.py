from __future__ import annotations

from typing import Any

from .cache import CallState


def _stage_values(state: CallState, stage: str) -> tuple[int, int]:
    keys = state.softmax_keys if stage == "softmax" else state.layernorm_keys
    return sum(key is not None for key in keys), len(keys)


def completed_output_rows(state: CallState) -> int:
    softmax_axis = state.spec.softmax_axis
    layernorm_axis = state.spec.layernorm_axis
    softmax_done, softmax_total = _stage_values(state, "softmax")
    layernorm_done, layernorm_total = _stage_values(state, "layernorm")
    softmax_rows = (
        softmax_done
        if softmax_axis == 1
        else (state.spec.rows if softmax_done == softmax_total else 0)
    )
    layernorm_rows = (
        layernorm_done
        if layernorm_axis == 1
        else (state.spec.rows if layernorm_done == layernorm_total else 0)
    )
    return min(softmax_rows, layernorm_rows)


def build_summary(states: list[CallState], last_affected_rows: list[int]) -> dict[str, Any]:
    ordered = sorted(states, key=lambda item: item.spec.sequence)
    calls: list[dict[str, Any]] = []
    total_completed_lanes = 0
    total_lanes = 0
    completed_rows = 0
    for state in ordered:
        softmax_done, softmax_total = _stage_values(state, "softmax")
        layernorm_done, layernorm_total = _stage_values(state, "layernorm")
        total_completed_lanes += softmax_done + layernorm_done
        total_lanes += softmax_total + layernorm_total
        rows_completed = completed_output_rows(state)
        completed_rows += rows_completed
        calls.append(
            {
                "call_id": state.spec.call_id,
                "sequence": state.spec.sequence,
                "rows": state.spec.rows,
                "cols": state.spec.cols,
                "softmax_axis": state.spec.softmax_axis,
                "layernorm_axis": state.spec.layernorm_axis,
                "epsilon": state.spec.epsilon,
                "layernorm_scale": state.spec.layernorm_scale,
                "softmax_max_subtraction": state.spec.stabilize,
                "completed_rows": rows_completed,
                "completed_lanes": {
                    "softmax": softmax_done,
                    "layernorm": layernorm_done,
                },
                "total_lanes": {
                    "softmax": softmax_total,
                    "layernorm": layernorm_total,
                },
                "progress": {
                    "softmax": (softmax_done / softmax_total) if softmax_total else 1.0,
                    "layernorm": (layernorm_done / layernorm_total) if layernorm_total else 1.0,
                },
            }
        )
    return {
        "completed_rows": completed_rows,
        "completed_lanes": total_completed_lanes,
        "total_lanes": total_lanes,
        "progress": (total_completed_lanes / total_lanes) if total_lanes else 1.0,
        "current_max_subtraction": {
            state.spec.call_id: state.spec.stabilize for state in ordered
        },
        "affected_rows": sorted(set(last_affected_rows)),
        "calls": calls,
    }
