from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .quantization import QuantParams, dequantize, quantize


def aggregate_float64(
    arrays_by_origin: Mapping[int, np.ndarray], mode: str
) -> tuple[np.ndarray, tuple[int, ...]]:
    origins = tuple(sorted(arrays_by_origin))
    if not origins:
        raise ValueError("cannot aggregate an empty origin set")
    result = np.zeros_like(arrays_by_origin[origins[0]], dtype=np.float64)
    for origin in origins:
        result += np.asarray(arrays_by_origin[origin], dtype=np.float64)
    if mode == "mean":
        result /= len(origins)
    return result, origins


def aggregate_fixed_point(
    q_by_origin: Mapping[int, np.ndarray],
    params: QuantParams,
    mode: str,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    origins = tuple(sorted(q_by_origin))
    if not origins:
        raise ValueError("cannot aggregate an empty origin set")
    qsum = np.zeros_like(np.asarray(q_by_origin[origins[0]], dtype=np.int64), dtype=np.int64)
    for origin in origins:
        qsum += np.asarray(q_by_origin[origin], dtype=np.int64)
    zero_adjustment = params.zero_point * len(origins) if params.scheme == "asymmetric" else 0
    effective = qsum - np.int64(zero_adjustment)
    if mode == "mean":
        effective = np.rint(effective.astype(np.float64) / len(origins)).astype(np.int64)
    values = (effective.astype(np.float64) * params.scale).astype(np.float32)
    return values, origins, effective


def _metrics(values: np.ndarray, truth: np.ndarray, tolerance: float) -> dict[str, object]:
    error = np.abs(np.asarray(values, dtype=np.float64) - np.asarray(truth, dtype=np.float64))
    max_abs = float(np.max(error)) if error.size else 0.0
    denominator = np.maximum(np.abs(np.asarray(truth, dtype=np.float64)), 1e-12)
    max_rel = float(np.max(error / denominator)) if error.size else 0.0
    return {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "tolerance_exceeded": bool(max_abs > tolerance),
    }


def _stage_summary(
    values: np.ndarray,
    truth: np.ndarray,
    tolerance: float,
    stage: str,
) -> dict[str, object]:
    result = {"stage": stage}
    result.update(_metrics(values, truth, tolerance))
    return result


def _first_exceeded(stages: Iterable[dict[str, object]]) -> str | None:
    for stage in stages:
        if bool(stage["tolerance_exceeded"]):
            return str(stage["stage"])
    return None


def _source_for_stage(first_exceeded: str | None) -> str:
    if first_exceeded in {"pack", "unpack"}:
        return "quantization"
    if first_exceeded == "aggregate":
        return "fixed_point_aggregation"
    return "within_tolerance"


def track_tensor_errors(
    *,
    window: int,
    worker_id: int,
    tensor_name: str,
    originals_by_origin: Mapping[int, np.ndarray],
    source_q_by_origin: Mapping[int, np.ndarray],
    received_q_by_origin: Mapping[int, np.ndarray],
    params: QuantParams,
    mode: str,
    tolerance: float,
    lower_params: QuantParams | None = None,
    complete: bool = True,
) -> dict[str, object]:
    exact_pack, pack_origins = aggregate_float64(originals_by_origin, mode)
    packed_by_origin = {
        origin: dequantize(source_q_by_origin[origin], params)
        for origin in sorted(source_q_by_origin)
    }
    packed_values, _ = aggregate_float64(packed_by_origin, mode)

    if received_q_by_origin:
        received_originals = {
            origin: originals_by_origin[origin]
            for origin in received_q_by_origin
            if origin in originals_by_origin
        }
        exact_received, received_origins = aggregate_float64(received_originals, mode)
        unpacked_values = np.zeros_like(exact_received, dtype=np.float64)
        for origin in sorted(received_q_by_origin):
            unpacked_values += dequantize(received_q_by_origin[origin], params).astype(np.float64)
        if mode == "mean":
            unpacked_values /= len(received_q_by_origin)
        aggregate_values, aggregate_origins, _ = aggregate_fixed_point(
            received_q_by_origin, params, mode
        )
    else:
        exact_received = exact_pack
        received_origins = ()
        unpacked_values = exact_pack.astype(np.float64)
        aggregate_values = packed_values
        aggregate_origins = ()

    stages = [
        _stage_summary(packed_values, exact_pack, tolerance, "pack"),
        _stage_summary(unpacked_values.astype(np.float32), exact_received, tolerance, "unpack"),
        _stage_summary(aggregate_values, exact_received, tolerance, "aggregate"),
    ]
    first_exceeded = _first_exceeded(stages)

    boundary = None
    if lower_params is not None and lower_params.bits < params.bits:
        lower_source_q_by_origin = {
            origin: quantize(originals_by_origin[origin], lower_params)
            for origin in sorted(originals_by_origin)
        }
        lower_packed_by_origin = {
            origin: dequantize(lower_source_q_by_origin[origin], lower_params)
            for origin in sorted(lower_source_q_by_origin)
        }
        lower_pack, _ = aggregate_float64(lower_packed_by_origin, mode)
        lower_aggregate, _, _ = aggregate_fixed_point(
            lower_source_q_by_origin, lower_params, mode
        )
        lower_stages = [
            _stage_summary(lower_pack, exact_pack, tolerance, "pack"),
            _stage_summary(lower_pack, exact_pack, tolerance, "unpack"),
            _stage_summary(lower_aggregate, exact_pack, tolerance, "aggregate"),
        ]
        lower_first = _first_exceeded(lower_stages)
        current_exceeds = first_exceeded is not None
        lower_exceeds = lower_first is not None
        if current_exceeds != lower_exceeds:
            boundary = {
                "tensor": tensor_name,
                "worker_id": worker_id,
                "current_bits": params.bits,
                "lower_bits": lower_params.bits,
                "current_scheme": params.scheme,
                "lower_scheme": lower_params.scheme,
                "current_exceeds": current_exceeds,
                "lower_exceeds": lower_exceeds,
                "current_first_exceeded_stage": first_exceeded,
                "lower_first_exceeded_stage": lower_first,
                "current_error_source": _source_for_stage(first_exceeded),
                "lower_error_source": _source_for_stage(lower_first),
                "current_stages": stages,
                "lower_stages": lower_stages,
            }

    return {
        "window": window,
        "worker_id": worker_id,
        "tensor": tensor_name,
        "complete": bool(complete),
        "origins": list(pack_origins),
        "received_origins": list(received_origins),
        "aggregate_origins": list(aggregate_origins),
        "tolerance": tolerance,
        "stages": stages,
        "first_exceeded_stage": first_exceeded,
        "error_source": _source_for_stage(first_exceeded),
        "boundary": boundary,
    }
