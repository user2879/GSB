"""Per-keypoint, stage-cached deterministic solving."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .budget import ResidualCounter
from .cache import StageCache, make_stage_key
from .config import SolverConfig
from .errors import (
    E_CFG_INVALID,
    E_INLIER_SHORTAGE,
    E_POINT_QUOTA_EXHAUSTED,
    Failure,
    SolverFailure,
    BudgetExceeded,
)
from .geometry import compute_residuals, triangulate_dlt, triangulation_stage
from .hashing import sha256_json
from .injections import InjectionItem, InjectionSet
from .models import Camera, InputBundle, Observation, PointSource, observations_fingerprint
from .quantize import quantize_point, quantize_residuals


@dataclass
class PointSolveResult:
    keypoint_id: str
    record: dict[str, Any]
    stage_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_additions: dict[str, dict[str, Any]] = field(default_factory=dict)
    residual_evaluations: int = 0
    status: str = "error"
    injection_hits: dict[str, str] = field(default_factory=dict)


def _failure_dict(
    code: str,
    keypoint_id: str | None,
    stage: str,
    message: str,
    *,
    view_index: int | None = None,
    observation_index: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return Failure(
        code=code,
        keypoint_id=keypoint_id,
        stage=stage,
        message=message,
        view_index=view_index,
        observation_index=observation_index,
        details=details or {},
    ).to_dict()


def _obs_from_dicts(items: list[dict[str, Any]]) -> list[Observation]:
    return sorted((Observation.from_dict(item) for item in items), key=lambda obs: obs.view_index)


def _sort_failure_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            item.get("keypoint_id") or "",
            item.get("code") or "",
            item.get("stage") or "",
            item.get("view_index", -1),
            item.get("observation_index", -1),
            item.get("message") or "",
        )

    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        unique[sha256_json(item)] = item
    return sorted(unique.values(), key=key)


def _worst_residual(
    residuals: list[dict[str, Any]], *, quantized: bool = False
) -> dict[str, Any]:
    field_name = "residual_quantized_px" if quantized else "residual_px"
    return min(
        residuals,
        key=lambda item: (-float(item[field_name]), int(item["view_index"])),
    )


def _state_record(
    point: np.ndarray,
    observations: list[Observation],
    residuals: list[dict[str, Any]],
    *,
    quantized: bool = False,
) -> dict[str, Any]:
    worst = _worst_residual(residuals, quantized=quantized)
    return {
        "point": np.asarray(point, dtype=np.float64).tolist(),
        "views": [obs.view_index for obs in observations],
        "residuals": copy.deepcopy(residuals),
        "max_residual_px": float(
            worst["residual_quantized_px"] if quantized else worst["residual_px"]
        ),
        "max_residual_view": int(worst["view_index"]),
    }


def _quantize_residual_records(
    residuals: list[dict[str, Any]], fractional_bits: int
) -> list[dict[str, Any]]:
    quantized = quantize_residuals(residuals, fractional_bits)
    output: list[dict[str, Any]] = []
    for original, record in zip(residuals, quantized["records"], strict=True):
        merged = copy.deepcopy(original)
        merged.update(
            {
                "residual_int": record["residual_int"],
                "residual_quantized_px": record["residual_quantized_px"],
                "quantization_error_px": record["quantization_error_px"],
            }
        )
        output.append(merged)
    return output


def _quantized_triangulation(
    observations: list[Observation],
    cameras: dict[int, Camera],
    cfg: SolverConfig,
    fractional_bits: int,
    keypoint_id: str,
) -> dict[str, Any]:
    raw = triangulate_dlt(
        observations, cameras, cfg, keypoint_id=keypoint_id, stage="quantization"
    )
    point_q = quantize_point(raw["point"], fractional_bits)
    return {
        "unquantized_point": raw["point"].tolist(),
        "point": np.asarray(point_q["quantized"], dtype=np.float64),
        "point_quantization": point_q,
        "singular_values": raw["singular_values"],
        "participating_views": raw["participating_views"],
    }


def _rejection_failure_result(
    *,
    keypoint_id: str,
    failure: Failure,
    trace: list[dict[str, Any]],
    removed_views: list[int],
    current_point: np.ndarray | None,
    current_observations: list[Observation],
    current_residuals: list[dict[str, Any]],
    counter: ResidualCounter,
    stop_reason: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "failure": failure.to_dict(),
        "trace": trace,
        "removed_views": removed_views,
        "rounds_executed": len(removed_views),
        "stop_reason": stop_reason,
        "current_point": None
        if current_point is None
        else np.asarray(current_point, dtype=np.float64).tolist(),
        "current_observations": [obs.to_dict() for obs in current_observations],
        "current_residuals": copy.deepcopy(current_residuals),
        "last_round_residual": float(
            _worst_residual(current_residuals)["residual_px"]
        )
        if current_residuals
        else None,
        "final_point": None,
        "final_observations": None,
        "final_residuals": None,
        "final_observation_fingerprint": None,
        "residual_evaluations": counter.used,
    }


def run_rejection_chain(
    *,
    initial_observations: list[Observation],
    initial_point: np.ndarray,
    initial_residuals: list[dict[str, Any]],
    cameras: dict[int, Camera],
    cfg: SolverConfig,
    keypoint_id: str,
    counter: ResidualCounter,
) -> dict[str, Any]:
    """Run the unquantized robust rejection chain.

    ``initial_residuals`` have already been charged by the triangulation stage.
    Every subsequent residual evaluation is charged to ``counter`` before it is
    computed.
    """

    current_observations = sorted(initial_observations, key=lambda obs: obs.view_index)
    current_point = np.asarray(initial_point, dtype=np.float64)
    current_residuals = copy.deepcopy(initial_residuals)
    trace: list[dict[str, Any]] = []
    removed_views: list[int] = []

    while True:
        worst = _worst_residual(current_residuals)
        worst_value = float(worst["residual_px"])
        if worst_value <= cfg.residual_threshold_px:
            stop_reason = "threshold"
            break
        if len(removed_views) >= cfg.max_rounds:
            stop_reason = "round_limit"
            break
        if len(current_observations) - 1 < cfg.min_inliers:
            failure = Failure(
                code=E_INLIER_SHORTAGE,
                keypoint_id=keypoint_id,
                stage="rejection",
                message="removing the worst observation would fall below min_inliers",
                view_index=int(worst["view_index"]),
                details={
                    "remaining_after_removal": len(current_observations) - 1,
                    "min_inliers": cfg.min_inliers,
                    "max_residual_px": worst_value,
                },
            )
            trace.append(
                {
                    "round": len(removed_views) + 1,
                    "decision": "inlier_shortage",
                    "state_before": _state_record(
                        current_point, current_observations, current_residuals
                    ),
                    "worst_view": int(worst["view_index"]),
                    "worst_residual_px": worst_value,
                    "would_remove_view": int(worst["view_index"]),
                    "remaining_after_removal": len(current_observations) - 1,
                    "min_inliers": cfg.min_inliers,
                    "state_after": None,
                }
            )
            return _rejection_failure_result(
                keypoint_id=keypoint_id,
                failure=failure,
                trace=trace,
                removed_views=removed_views,
                current_point=current_point,
                current_observations=current_observations,
                current_residuals=current_residuals,
                counter=counter,
                stop_reason="inlier_shortage",
            )

        removed_view = int(worst["view_index"])
        candidate_observations = [
            obs for obs in current_observations if obs.view_index != removed_view
        ]
        trace_entry: dict[str, Any] = {
            "round": len(removed_views) + 1,
            "decision": "remove",
            "state_before": _state_record(
                current_point, current_observations, current_residuals
            ),
            "worst_view": removed_view,
            "worst_residual_px": worst_value,
            "removed_view": removed_view,
            "threshold_px": cfg.residual_threshold_px,
        }
        try:
            candidate_triangulation = triangulate_dlt(
                candidate_observations,
                cameras,
                cfg,
                keypoint_id=keypoint_id,
                stage="rejection",
            )
        except SolverFailure as exc:
            trace_entry["decision"] = "triangulation_degenerate_after_removal"
            trace_entry["failure"] = exc.failure.to_dict()
            trace_entry["state_after"] = None
            trace.append(trace_entry)
            removed_views.append(removed_view)
            return _rejection_failure_result(
                keypoint_id=keypoint_id,
                failure=exc.failure,
                trace=trace,
                removed_views=removed_views,
                current_point=current_point,
                current_observations=current_observations,
                current_residuals=current_residuals,
                counter=counter,
                stop_reason="triangulation_degenerate",
            )
        candidate_point = np.asarray(candidate_triangulation["point"], dtype=np.float64)
        try:
            counter.charge(
                len(candidate_observations),
                stage="rejection",
                details={"round": len(removed_views) + 1, "removed_view": removed_view},
            )
        except BudgetExceeded as exc:
            trace_entry["decision"] = "quota_exhausted"
            trace_entry["failure"] = exc.failure.to_dict()
            trace_entry["state_after"] = None
            trace.append(trace_entry)
            removed_views.append(removed_view)
            return _rejection_failure_result(
                keypoint_id=keypoint_id,
                failure=exc.failure,
                trace=trace,
                removed_views=removed_views,
                current_point=current_point,
                current_observations=current_observations,
                current_residuals=current_residuals,
                counter=counter,
                stop_reason="quota_exhausted",
            )
        candidate_residuals = compute_residuals(
            candidate_point, candidate_observations, cameras
        )
        trace_entry["state_after"] = _state_record(
            candidate_point, candidate_observations, candidate_residuals
        )
        trace_entry["charged_residual_evaluations"] = len(candidate_observations)
        trace.append(trace_entry)
        removed_views.append(removed_view)
        current_observations = candidate_observations
        current_point = candidate_point
        current_residuals = candidate_residuals

    final_obs_fingerprint = observations_fingerprint(current_observations)
    return {
        "status": "ok",
        "failure": None,
        "trace": trace,
        "removed_views": removed_views,
        "rounds_executed": len(removed_views),
        "stop_reason": stop_reason,
        "final_point": current_point.tolist(),
        "final_observations": [obs.to_dict() for obs in current_observations],
        "final_residuals": current_residuals,
        "final_observation_fingerprint": final_obs_fingerprint,
        "last_round_residual": float(_worst_residual(current_residuals)["residual_px"]),
        "max_residual_px": float(_worst_residual(current_residuals)["residual_px"]),
        "residual_evaluations": counter.used,
    }


def run_quantized_chain(
    *,
    initial_observations: list[Observation],
    cameras: dict[int, Camera],
    cfg: SolverConfig,
    fractional_bits: int,
    keypoint_id: str,
    counter: ResidualCounter,
) -> dict[str, Any]:
    """Run a complete DLT -> quantize -> residual -> rejection chain."""

    current_observations = sorted(initial_observations, key=lambda obs: obs.view_index)
    trace: list[dict[str, Any]] = []
    removed_views: list[int] = []
    try:
        initial_triangulation = _quantized_triangulation(
            current_observations, cameras, cfg, fractional_bits, keypoint_id
        )
    except SolverFailure as exc:
        return {
            "status": "error",
            "fractional_bits": fractional_bits,
            "failure": exc.failure.to_dict(),
            "trace": trace,
            "removed_views": removed_views,
            "rounds_executed": 0,
            "initial_point": None,
            "initial_residuals": None,
            "final_point": None,
            "final_residuals": None,
            "residual_evaluations": counter.used,
        }
    initial_point = initial_triangulation["point"]
    try:
        counter.charge(
            len(current_observations),
            stage="quantization",
            details={"fractional_bits": fractional_bits},
        )
    except BudgetExceeded as exc:
        return {
            "status": "error",
            "fractional_bits": fractional_bits,
            "failure": exc.failure.to_dict(),
            "trace": trace,
            "removed_views": removed_views,
            "rounds_executed": 0,
            "initial_point": initial_triangulation["unquantized_point"],
            "initial_residuals": None,
            "final_point": None,
            "final_residuals": None,
            "residual_evaluations": counter.used,
        }
    base_residuals = compute_residuals(initial_point, current_observations, cameras)
    current_residuals = _quantize_residual_records(base_residuals, fractional_bits)
    current_point = initial_point
    initial_state = _state_record(
        current_point, current_observations, current_residuals, quantized=True
    )
    initial_state["point_quantization"] = initial_triangulation["point_quantization"]

    while True:
        worst = _worst_residual(current_residuals, quantized=True)
        worst_value = float(worst["residual_quantized_px"])
        if worst_value <= cfg.residual_threshold_px:
            stop_reason = "threshold"
            break
        if len(removed_views) >= cfg.max_rounds:
            stop_reason = "round_limit"
            break
        if len(current_observations) - 1 < cfg.min_inliers:
            failure = Failure(
                code=E_INLIER_SHORTAGE,
                keypoint_id=keypoint_id,
                stage="quantization",
                message="quantized chain would fall below min_inliers",
                view_index=int(worst["view_index"]),
                details={
                    "fractional_bits": fractional_bits,
                    "remaining_after_removal": len(current_observations) - 1,
                    "min_inliers": cfg.min_inliers,
                },
            )
            trace.append(
                {
                    "round": len(removed_views) + 1,
                    "decision": "inlier_shortage",
                    "state_before": _state_record(
                        current_point, current_observations, current_residuals, quantized=True
                    ),
                    "worst_view": int(worst["view_index"]),
                    "worst_residual_quantized_px": worst_value,
                    "would_remove_view": int(worst["view_index"]),
                    "state_after": None,
                }
            )
            return {
                "status": "error",
                "fractional_bits": fractional_bits,
                "failure": failure.to_dict(),
                "trace": trace,
                "removed_views": removed_views,
                "rounds_executed": len(removed_views),
                "initial_point": initial_point.tolist(),
                "initial_residuals": initial_state["residuals"],
                "final_point": None,
                "final_residuals": None,
                "residual_evaluations": counter.used,
                "stop_reason": "inlier_shortage",
            }

        removed_view = int(worst["view_index"])
        candidate_observations = [
            obs for obs in current_observations if obs.view_index != removed_view
        ]
        trace_entry: dict[str, Any] = {
            "round": len(removed_views) + 1,
            "decision": "remove",
            "state_before": _state_record(
                current_point, current_observations, current_residuals, quantized=True
            ),
            "worst_view": removed_view,
            "worst_residual_quantized_px": worst_value,
            "removed_view": removed_view,
            "threshold_px": cfg.residual_threshold_px,
            "fractional_bits": fractional_bits,
        }
        try:
            candidate_triangulation = _quantized_triangulation(
                candidate_observations, cameras, cfg, fractional_bits, keypoint_id
            )
        except SolverFailure as exc:
            trace_entry["decision"] = "triangulation_degenerate_after_removal"
            trace_entry["failure"] = exc.failure.to_dict()
            trace_entry["state_after"] = None
            trace.append(trace_entry)
            removed_views.append(removed_view)
            return {
                "status": "error",
                "fractional_bits": fractional_bits,
                "failure": exc.failure.to_dict(),
                "trace": trace,
                "removed_views": removed_views,
                "rounds_executed": len(removed_views),
                "initial_point": initial_point.tolist(),
                "initial_residuals": initial_state["residuals"],
                "final_point": None,
                "final_residuals": None,
                "residual_evaluations": counter.used,
                "stop_reason": "triangulation_degenerate",
            }
        candidate_point = candidate_triangulation["point"]
        try:
            counter.charge(
                len(candidate_observations),
                stage="quantization",
                details={
                    "fractional_bits": fractional_bits,
                    "round": len(removed_views) + 1,
                    "removed_view": removed_view,
                },
            )
        except BudgetExceeded as exc:
            trace_entry["decision"] = "quota_exhausted"
            trace_entry["failure"] = exc.failure.to_dict()
            trace_entry["state_after"] = None
            trace.append(trace_entry)
            removed_views.append(removed_view)
            return {
                "status": "error",
                "fractional_bits": fractional_bits,
                "failure": exc.failure.to_dict(),
                "trace": trace,
                "removed_views": removed_views,
                "rounds_executed": len(removed_views),
                "initial_point": initial_point.tolist(),
                "initial_residuals": initial_state["residuals"],
                "final_point": None,
                "final_residuals": None,
                "residual_evaluations": counter.used,
                "stop_reason": "quota_exhausted",
            }
        candidate_residuals = _quantize_residual_records(
            compute_residuals(candidate_point, candidate_observations, cameras),
            fractional_bits,
        )
        trace_entry["state_after"] = _state_record(
            candidate_point, candidate_observations, candidate_residuals, quantized=True
        )
        trace_entry["charged_residual_evaluations"] = len(candidate_observations)
        trace.append(trace_entry)
        removed_views.append(removed_view)
        current_observations = candidate_observations
        current_point = candidate_point
        current_residuals = candidate_residuals

    final_fingerprint = observations_fingerprint(current_observations)
    final_point_quantization = quantize_point(current_point, fractional_bits)
    return {
        "status": "ok",
        "fractional_bits": fractional_bits,
        "failure": None,
        "trace": trace,
        "removed_views": removed_views,
        "rounds_executed": len(removed_views),
        "stop_reason": stop_reason,
        "initial_point": initial_point.tolist(),
        "initial_residuals": initial_state["residuals"],
        "final_point": current_point.tolist(),
        "final_point_quantization": final_point_quantization,
        "final_observations": [obs.to_dict() for obs in current_observations],
        "final_residuals": current_residuals,
        "final_observation_fingerprint": final_fingerprint,
        "last_round_residual": float(
            _worst_residual(current_residuals, quantized=True)["residual_quantized_px"]
        ),
        "residual_evaluations": counter.used,
    }


def _residuals_by_view(residuals: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if not residuals:
        return {}
    return {int(item["view_index"]): item for item in residuals}


def _compare_state(
    stage: str,
    base_point: Any,
    base_residuals: list[dict[str, Any]] | None,
    quant_point: Any,
    quant_residuals: list[dict[str, Any]] | None,
    *,
    quantized_field: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"stage": stage}
    if base_point is None or quant_point is None:
        return {
            **record,
            "point_delta_vector": None,
            "point_delta_l2": None,
            "residual_delta_max_abs": None,
            "residual_delta_rms": None,
            "residuals": [],
            "available": False,
        }
    base_point_array = np.asarray(base_point, dtype=np.float64)
    quant_point_array = np.asarray(quant_point, dtype=np.float64)
    point_delta = quant_point_array - base_point_array
    base_map = _residuals_by_view(base_residuals)
    quant_map = _residuals_by_view(quant_residuals)
    views = sorted(set(base_map) | set(quant_map))
    residual_records: list[dict[str, Any]] = []
    deltas: list[float] = []
    for view in views:
        base_item = base_map.get(view)
        quant_item = quant_map.get(view)
        base_value = float(base_item["residual_px"]) if base_item else None
        quant_value = None
        if quant_item is not None:
            quant_value = float(
                quant_item.get(quantized_field or "residual_px", quant_item["residual_px"])
            )
        if base_value is not None and quant_value is not None:
            delta = quant_value - base_value
            deltas.append(abs(delta))
        else:
            delta = None
        residual_records.append(
            {
                "view_index": view,
                "base_residual_px": base_value,
                "quantized_residual_px": quant_value,
                "delta_px": delta,
            }
        )
    if deltas:
        residual_max = max(deltas)
        residual_rms = math.sqrt(sum(value * value for value in deltas) / len(deltas))
    else:
        residual_max = None
        residual_rms = None
    return {
        **record,
        "point_delta_vector": point_delta.tolist(),
        "point_delta_l2": float(np.linalg.norm(point_delta)),
        "residual_delta_max_abs": residual_max,
        "residual_delta_rms": residual_rms,
        "residuals": residual_records,
        "available": True,
    }


def _error_propagation(
    base_triangulation: dict[str, Any],
    base_rejection: dict[str, Any],
    quantized_chain: dict[str, Any],
    tolerance: float,
    quantized_field: str,
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    stages.append(
        _compare_state(
            "triangulation",
            base_triangulation.get("point"),
            base_triangulation.get("residuals"),
            quantized_chain.get("initial_point"),
            quantized_chain.get("initial_residuals"),
            quantized_field=quantized_field,
        )
    )
    base_trace = base_rejection.get("trace", [])
    quant_trace = quantized_chain.get("trace", [])
    for index in range(max(len(base_trace), len(quant_trace))):
        base_entry = base_trace[index] if index < len(base_trace) else {}
        quant_entry = quant_trace[index] if index < len(quant_trace) else {}
        base_state = base_entry.get("state_after")
        quant_state = quant_entry.get("state_after")
        stage = _compare_state(
            f"rejection_round_{index + 1}",
            None if base_state is None else base_state.get("point"),
            None if base_state is None else base_state.get("residuals"),
            None if quant_state is None else quant_state.get("point"),
            None if quant_state is None else quant_state.get("residuals"),
            quantized_field=quantized_field,
        )
        if base_entry.get("removed_view") is not None or quant_entry.get("removed_view") is not None:
            stage["base_removed_view"] = base_entry.get("removed_view")
            stage["quantized_removed_view"] = quant_entry.get("removed_view")
        stages.append(stage)
    stages.append(
        _compare_state(
            "finalization",
            base_rejection.get("final_point"),
            base_rejection.get("final_residuals"),
            quantized_chain.get("final_point"),
            quantized_chain.get("final_residuals"),
            quantized_field=quantized_field,
        )
    )
    first_exceeded = None
    for stage in stages:
        point_delta = stage.get("point_delta_l2")
        residual_delta = stage.get("residual_delta_max_abs")
        if point_delta is not None and point_delta > tolerance:
            first_exceeded = stage["stage"]
            break
        if residual_delta is not None and residual_delta > tolerance:
            first_exceeded = stage["stage"]
            break
    return {
        "tolerance": tolerance,
        "stages": stages,
        "first_tolerance_exceeded_stage": first_exceeded,
        "all_within_tolerance": first_exceeded is None,
    }


def _classify_removal_flip(
    *,
    base_rejection: dict[str, Any],
    low_chain: dict[str, Any],
    high_chain: dict[str, Any],
    primary_bits: int,
    high_bits: int,
) -> dict[str, Any]:
    base_removed = list(base_rejection.get("removed_views") or [])
    low_removed = list(low_chain.get("removed_views") or [])
    high_removed = list(high_chain.get("removed_views") or [])
    detected = low_removed != high_removed
    result: dict[str, Any] = {
        "detected": detected,
        "origin": None,
        "base_removed_views": base_removed,
        "low_removed_views": low_removed,
        "high_removed_views": high_removed,
        "primary_bits": primary_bits,
        "high_bits": high_bits,
    }
    if not detected:
        return result
    common = min(len(low_removed), len(high_removed))
    first_difference = common
    for index in range(common):
        if low_removed[index] != high_removed[index]:
            first_difference = index
            break
    last_low = low_removed[-1] if low_removed else None
    last_high = high_removed[-1] if high_removed else None
    last_differs = last_low != last_high
    prefixes_agree_until_last = (
        bool(low_removed)
        and bool(high_removed)
        and len(low_removed) == len(high_removed)
        and low_removed[:-1] == high_removed[:-1]
        and last_differs
    )
    one_chain_has_no_removal = bool(low_removed) != bool(high_removed)
    if prefixes_agree_until_last or one_chain_has_no_removal:
        origin = "quantization_truncation"
    elif first_difference < common or len(low_removed) != len(high_removed):
        origin = "residual_accumulation_rounding"
    else:
        origin = "tie_break_rounding"
    result.update(
        {
            "origin": origin,
            "first_difference_round": first_difference + 1,
            "last_removed_low": last_low,
            "last_removed_high": last_high,
            "prefixes_agree_until_last": prefixes_agree_until_last,
            "one_chain_has_no_removal": one_chain_has_no_removal,
            "explanation": {
                "quantization_truncation": (
                    "The removed-view sequences agree until the final removal, "
                    "where a residual quantized at the lower width changes the "
                    "decisive maximum."
                ),
                "residual_accumulation_rounding": (
                    "The removed-view sequences diverge before the final removal, "
                    "so earlier quantized point/residual updates accumulate into "
                    "a different rejection path."
                ),
                "tie_break_rounding": (
                    "The sequences differ only at the end without an earlier "
                    "path change; the final tie/ordering is resolved by view index."
                ),
            }[origin],
        }
    )
    if low_chain.get("status") == "ok" and high_chain.get("status") == "ok":
        low_last = low_chain.get("final_residuals") or []
        high_last = high_chain.get("final_residuals") or []
        if low_last and high_last:
            low_worst = _worst_residual(low_last, quantized=True)
            high_worst = _worst_residual(high_last, quantized=True)
            result["evidence"] = {
                "low_max_residual_px": float(low_worst["residual_quantized_px"]),
                "low_max_residual_view": int(low_worst["view_index"]),
                "high_max_residual_px": float(high_worst["residual_quantized_px"]),
                "high_max_residual_view": int(high_worst["view_index"]),
            }
    return result


def _stage_material_for_injections(
    injections: InjectionSet, keypoint_id: str, stage: str
) -> dict[str, Any]:
    matching = [item for item in injections.for_keypoint(keypoint_id) if item.stage == stage]
    return {
        "injections": [
            {"id": item.id, "fingerprint": item.fingerprint} for item in matching
        ]
    }


def _find_injection(
    injections: InjectionSet, keypoint_id: str, stage: str, fault: str
) -> InjectionItem | None:
    for item in injections.for_keypoint(keypoint_id):
        if item.stage == stage and item.fault == fault:
            return item
    return None


def solve_point(
    point: PointSource,
    bundle: InputBundle,
    cfg: SolverConfig,
    injections: InjectionSet,
    cache: StageCache,
) -> PointSolveResult:
    """Solve one keypoint through four independently cached stages."""

    cache_additions: dict[str, dict[str, Any]] = {}
    reused_stages: list[str] = []
    recomputed_stages: list[str] = []
    stage_keys: dict[str, str] = {}
    stage_payloads: dict[str, dict[str, Any]] = {}
    all_failures: list[dict[str, Any]] = [failure.to_dict() for failure in point.failures]
    injection_hits = dict(point.injection_hits)

    def cached_or_compute(
        stage: str,
        key: str,
        compute_fn,
    ) -> tuple[dict[str, Any], bool]:
        stage_keys[stage] = key
        cached = cache.get(key)
        if cached is not None:
            reused_stages.append(stage)
            stage_payloads[stage] = copy.deepcopy(cached)
            return cached, True
        recomputed_stages.append(stage)
        payload = compute_fn()
        stage_payloads[stage] = copy.deepcopy(payload)
        cache_additions[key] = {
            "schema": "pose3d-stage-cache/1",
            "stage": stage,
            "payload": copy.deepcopy(payload),
        }
        return payload, False

    load_key = make_stage_key(
        "load",
        None,
        {
            "keypoint_id": point.keypoint_id,
            "raw_observation_fingerprint": point.raw_fingerprint,
            "camera_fingerprint": point.camera_fingerprint(bundle.cameras),
            "min_views": cfg.min_views,
            "injections": _stage_material_for_injections(
                injections, point.keypoint_id, "load"
            ),
        },
    )

    def compute_load() -> dict[str, Any]:
        if point.load_error is not None:
            return {
                "status": "error",
                "observations": [obs.to_dict() for obs in point.observations],
                "failures": [failure.to_dict() for failure in point.failures],
                "load_error": point.load_error,
                "observation_fingerprint": point.observations_fingerprint,
                "camera_fingerprint": point.camera_fingerprint(bundle.cameras),
            }
        return {
            "status": "ok",
            "observations": [obs.to_dict() for obs in point.observations],
            "failures": [failure.to_dict() for failure in point.failures],
            "load_error": None,
            "observation_fingerprint": point.observations_fingerprint,
            "camera_fingerprint": point.camera_fingerprint(bundle.cameras),
        }

    load_payload, load_reused = cached_or_compute("load", load_key, compute_load)
    all_failures.extend(load_payload.get("failures", []))
    if load_payload.get("load_error"):
        all_failures.append(load_payload["load_error"])
    for item in injections.for_keypoint(point.keypoint_id):
        if item.fault == "nonorthogonal_rotation" and item.id in point.injection_hits:
            injection_hits[item.id] = item.business_code

    base_record: dict[str, Any] = {
        "keypoint_id": point.keypoint_id,
        "line_number": point.line_number,
        "base_name": point.base_name,
        "instance_suffix": point.instance_suffix,
        "status": "error",
        "error_codes": [],
        "failures": [],
        "observation_count_loaded": len(point.raw_observations),
        "observation_count_valid": len(point.observations),
        "observation_count_final": 0,
        "observations": [obs.to_dict() for obs in point.observations],
        "triangulation": None,
        "disparity": None,
        "outlier_rejection": None,
        "final": None,
        "quantization": None,
        "cache": {},
        "residual_evaluations": 0,
        "injection_hits": injection_hits,
    }

    def finish(status: str, residual_evaluations: int) -> PointSolveResult:
        base_record["status"] = status
        failures_sorted = _sort_failure_dicts(all_failures)
        base_record["failures"] = failures_sorted
        base_record["error_codes"] = sorted({item["code"] for item in failures_sorted})
        base_record["cache"] = {
            "stage_keys": stage_keys,
            "reused_stages": reused_stages,
            "recomputed_stages": recomputed_stages,
            "fully_reused": len(recomputed_stages) == 0,
        }
        base_record["residual_evaluations"] = int(residual_evaluations)
        base_record["injection_hits"] = injection_hits
        return PointSolveResult(
            keypoint_id=point.keypoint_id,
            record=base_record,
            stage_payloads=stage_payloads,
            cache_additions=cache_additions,
            residual_evaluations=int(residual_evaluations),
            status=status,
            injection_hits=injection_hits,
        )

    if load_payload.get("status") != "ok":
        return finish("error", 0)

    effective_cfg = cfg
    disparity_injection = _find_injection(
        injections, point.keypoint_id, "triangulation", "min_disparity_above_limit"
    )
    if disparity_injection is not None:
        injection_hits[disparity_injection.id] = disparity_injection.business_code
        raw_value = disparity_injection.value
        try:
            injected_value = float(raw_value)
        except (TypeError, ValueError):
            injected_value = math.inf
        if (
            not math.isfinite(injected_value)
            or injected_value < 0.0
            or injected_value > cfg.legal_min_disparity_max_deg
        ):
            failure = _failure_dict(
                E_CFG_INVALID,
                point.keypoint_id,
                "triangulation",
                "injected minimum disparity exceeds the legal upper bound",
                details={
                    "injection_id": disparity_injection.id,
                    "injected_min_disparity_deg": raw_value,
                    "legal_min_disparity_max_deg": cfg.legal_min_disparity_max_deg,
                },
            )
            all_failures.append(failure)
            triangulation_key = make_stage_key(
                "triangulation",
                load_key,
                {
                    "config": effective_cfg.triangulation_fingerprint,
                    "injection": _stage_material_for_injections(
                        injections, point.keypoint_id, "triangulation"
                    ),
                    "status": "config_invalid",
                },
            )
            stage_keys["triangulation"] = triangulation_key
            payload = {
                "status": "error",
                "failure": failure,
                "residual_evaluations": 0,
            }
            stage_payloads["triangulation"] = payload
            recomputed_stages.append("triangulation")
            cache_additions[triangulation_key] = {
                "schema": "pose3d-stage-cache/1",
                "stage": "triangulation",
                "payload": copy.deepcopy(payload),
            }
            return finish("error", 0)
        effective_cfg = cfg.replace_for_point(min_disparity_deg=injected_value)

    observations = _obs_from_dicts(load_payload["observations"])
    counter = ResidualCounter(
        cfg.per_keypoint_quota, keypoint_id=point.keypoint_id, start_used=0
    )
    triangulation_key = make_stage_key(
        "triangulation",
        load_key,
        {
            "config": effective_cfg.triangulation_fingerprint,
            "camera_fingerprint": load_payload.get("camera_fingerprint"),
            "per_keypoint_quota": cfg.per_keypoint_quota,
            "injection": _stage_material_for_injections(
                injections, point.keypoint_id, "triangulation"
            ),
        },
    )

    def compute_triangulation() -> dict[str, Any]:
        try:
            return triangulation_stage(
                observations,
                bundle.cameras,
                effective_cfg,
                keypoint_id=point.keypoint_id,
                residual_counter=counter,
            )
        except SolverFailure as exc:
            return {
                "status": "error",
                "failure": exc.failure.to_dict(),
                "partial": exc.partial,
                "residual_evaluations": counter.used,
            }

    triangulation_payload, _ = cached_or_compute(
        "triangulation", triangulation_key, compute_triangulation
    )
    triangulation_cost = int(triangulation_payload.get("residual_evaluations", 0))
    counter.used = triangulation_cost
    if triangulation_payload.get("status") != "ok":
        if triangulation_payload.get("failure"):
            all_failures.append(triangulation_payload["failure"])
        base_record["triangulation"] = triangulation_payload
        base_record["disparity"] = (triangulation_payload.get("partial") or {}).get(
            "disparity"
        ) or (triangulation_payload.get("failure", {}).get("details") or {}).get(
            "disparity"
        )
        return finish("error", triangulation_cost)

    base_record["triangulation"] = triangulation_payload
    base_record["disparity"] = triangulation_payload.get("disparity")
    accepted_observations = _obs_from_dicts(triangulation_payload["accepted_observations"])
    initial_point = np.asarray(triangulation_payload["point"], dtype=np.float64)
    initial_residuals = triangulation_payload["residuals"]

    threshold_injection = _find_injection(
        injections, point.keypoint_id, "rejection", "zero_outlier_threshold"
    )
    rejection_key = make_stage_key(
        "rejection",
        triangulation_key,
        {
            "config": effective_cfg.rejection_fingerprint,
            "per_keypoint_quota": cfg.per_keypoint_quota,
            "triangulation_cost": triangulation_cost,
            "injection": _stage_material_for_injections(
                injections, point.keypoint_id, "rejection"
            ),
        },
    )
    if threshold_injection is not None:
        injection_hits[threshold_injection.id] = threshold_injection.business_code
        failure = _failure_dict(
            E_CFG_INVALID,
            point.keypoint_id,
            "rejection",
            "injected outlier threshold is zero",
            details={
                "injection_id": threshold_injection.id,
                "threshold_px": 0.0,
            },
        )
        all_failures.append(failure)
        payload = {
            "status": "error",
            "failure": failure,
            "residual_evaluations": counter.used,
        }
        stage_keys["rejection"] = rejection_key
        stage_payloads["rejection"] = payload
        recomputed_stages.append("rejection")
        cache_additions[rejection_key] = {
            "schema": "pose3d-stage-cache/1",
            "stage": "rejection",
            "payload": copy.deepcopy(payload),
        }
        return finish("error", counter.used)

    def compute_rejection() -> dict[str, Any]:
        try:
            return run_rejection_chain(
                initial_observations=accepted_observations,
                initial_point=initial_point,
                initial_residuals=initial_residuals,
                cameras=bundle.cameras,
                cfg=effective_cfg,
                keypoint_id=point.keypoint_id,
                counter=counter,
            )
        except BudgetExceeded as exc:
            return {
                "status": "error",
                "failure": exc.failure.to_dict(),
                "trace": [],
                "removed_views": [],
                "rounds_executed": 0,
                "stop_reason": "quota_exhausted",
                "final_point": None,
                "final_residuals": None,
                "residual_evaluations": counter.used,
            }

    rejection_payload, _ = cached_or_compute(
        "rejection", rejection_key, compute_rejection
    )
    rejection_cost = int(rejection_payload.get("residual_evaluations", counter.used))
    base_record["residual_evaluations"] = rejection_cost
    base_record["outlier_rejection"] = {
        key: copy.deepcopy(value)
        for key, value in rejection_payload.items()
        if key
        in {
            "status",
            "failure",
            "trace",
            "removed_views",
            "rounds_executed",
            "stop_reason",
            "last_round_residual",
            "max_residual_px",
            "final_point",
            "final_observations",
            "final_residuals",
            "final_observation_fingerprint",
            "residual_evaluations",
        }
    }
    if rejection_payload.get("status") != "ok":
        if rejection_payload.get("failure"):
            all_failures.append(rejection_payload["failure"])
        base_record["observation_count_final"] = len(
            rejection_payload.get("current_observations", [])
        )
        return finish("error", rejection_cost)

    final_point = rejection_payload["final_point"]
    final_residuals = rejection_payload["final_residuals"]
    final_observations = rejection_payload.get("final_observations", [])
    base_record["observation_count_final"] = len(final_observations)
    base_record["final"] = {
        "point": final_point,
        "observations": final_observations,
        "residuals": final_residuals,
        "observation_fingerprint": rejection_payload.get(
            "final_observation_fingerprint"
        ),
        "max_residual_px": rejection_payload.get("max_residual_px"),
    }

    finalization_key = make_stage_key(
        "finalization",
        rejection_key,
        {
            "quantization": effective_cfg.quantization_fingerprint,
            "per_keypoint_quota": cfg.per_keypoint_quota,
            "rejection_cost": rejection_cost,
        },
    )

    def compute_finalization() -> dict[str, Any]:
        quantized_counter = ResidualCounter(
            cfg.per_keypoint_quota,
            keypoint_id=point.keypoint_id,
            start_used=rejection_cost,
        )
        low_chain = run_quantized_chain(
            initial_observations=accepted_observations,
            cameras=bundle.cameras,
            cfg=effective_cfg,
            fractional_bits=effective_cfg.primary_bits,
            keypoint_id=point.keypoint_id,
            counter=quantized_counter,
        )
        if quantized_counter.used > cfg.per_keypoint_quota:
            return {
                "status": "error",
                "failure": _failure_dict(
                    E_POINT_QUOTA_EXHAUSTED,
                    point.keypoint_id,
                    "quantization",
                    "quantization diagnostics exceed the per-keypoint quota",
                    details={
                        "quota": cfg.per_keypoint_quota,
                        "used": quantized_counter.used,
                    },
                ),
                "residual_evaluations": quantized_counter.used,
                "low_chain": low_chain,
                "high_chain": None,
            }
        if (low_chain.get("failure") or {}).get("code") == E_POINT_QUOTA_EXHAUSTED:
            return {
                "status": "error",
                "failure": low_chain["failure"],
                "residual_evaluations": quantized_counter.used,
                "low_chain": low_chain,
                "high_chain": None,
            }
        high_chain = run_quantized_chain(
            initial_observations=accepted_observations,
            cameras=bundle.cameras,
            cfg=effective_cfg,
            fractional_bits=effective_cfg.high_bits,
            keypoint_id=point.keypoint_id,
            counter=quantized_counter,
        )
        if (high_chain.get("failure") or {}).get("code") == E_POINT_QUOTA_EXHAUSTED:
            return {
                "status": "error",
                "failure": high_chain["failure"],
                "residual_evaluations": quantized_counter.used,
                "low_chain": low_chain,
                "high_chain": high_chain,
            }
        static_point = quantize_point(final_point, effective_cfg.primary_bits)
        static_residuals = quantize_residuals(
            final_residuals, effective_cfg.primary_bits
        )
        propagation_low = _error_propagation(
            triangulation_payload,
            rejection_payload,
            low_chain,
            effective_cfg.error_tolerance,
            "residual_quantized_px",
        )
        propagation_high = _error_propagation(
            triangulation_payload,
            rejection_payload,
            high_chain,
            effective_cfg.error_tolerance,
            "residual_quantized_px",
        )
        flip = _classify_removal_flip(
            base_rejection=rejection_payload,
            low_chain=low_chain,
            high_chain=high_chain,
            primary_bits=effective_cfg.primary_bits,
            high_bits=effective_cfg.high_bits,
        )
        return {
            "status": "ok",
            "failure": None,
            "rounding": effective_cfg.rounding,
            "primary_bits": effective_cfg.primary_bits,
            "high_bits": effective_cfg.high_bits,
            "static_primary_point": static_point,
            "static_primary_residuals": static_residuals,
            "low_chain": low_chain,
            "high_chain": high_chain,
            "error_propagation_low": propagation_low,
            "error_propagation_high": propagation_high,
            "removal_flip": flip,
            "residual_evaluations": quantized_counter.used,
        }

    finalization_payload, _ = cached_or_compute(
        "finalization", finalization_key, compute_finalization
    )
    residual_evaluations = int(
        finalization_payload.get("residual_evaluations", rejection_cost)
    )
    base_record["residual_evaluations"] = residual_evaluations
    base_record["quantization"] = copy.deepcopy(finalization_payload)
    if finalization_payload.get("status") != "ok":
        if finalization_payload.get("failure"):
            all_failures.append(finalization_payload["failure"])
        return finish("error", residual_evaluations)
    low_chain = finalization_payload.get("low_chain") or {}
    high_chain = finalization_payload.get("high_chain") or {}
    for chain in (low_chain, high_chain):
        if chain.get("status") != "ok" and chain.get("failure"):
            all_failures.append(chain["failure"])
    return finish("ok", residual_evaluations)
