"""Deterministic quantile calibration and scale/zero-point derivation."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .config import ResolvedTensorConfig
from .errors import QuantCalError
from .models import (
    CalibrationSample,
    TensorSpec,
    quant_bounds,
    scheme_saturation_max,
    symmetric_saturation_max,
)
from .util import as_finite_float, fingerprint, is_int, load_json, load_jsonl


QUANTILE_METHOD = "linear_type7"
TIE_RULE = "stable_sort;retain_duplicate_multiplicity;no_jitter"
ZERO_RANGE_GUARD_SCALE = 0.001
ZERO_SPAN_EXCHANGE_HALF_WIDTH = 0.001


def load_tensor_specs(path: str) -> list[TensorSpec]:
    document = load_json(path)
    if not isinstance(document, dict) or not isinstance(document.get("tensors"), list):
        raise QuantCalError(
            "E_FORMAT_INVALID", "tensor manifest must contain a tensors array", stage="parse"
        )
    return parse_tensor_specs(document)


def parse_tensor_specs(document: Any) -> list[TensorSpec]:
    if not isinstance(document, dict) or not isinstance(document.get("tensors"), list):
        raise QuantCalError(
            "E_FORMAT_INVALID", "tensor manifest must contain a tensors array", stage="parse"
        )
    result: list[TensorSpec] = []
    seen: set[str] = set()
    for index, row in enumerate(document["tensors"]):
        if not isinstance(row, dict):
            raise QuantCalError(
                "E_FORMAT_INVALID", f"tensor manifest row {index} must be an object", stage="parse"
            )
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise QuantCalError("E_FORMAT_INVALID", "tensor name must be non-empty", stage="parse")
        if name in seen:
            raise QuantCalError(
                "E_FORMAT_INVALID", "duplicate tensor name in manifest", tensor=name, stage="parse"
            )
        seen.add(name)
        shape_raw = row.get("shape")
        if not isinstance(shape_raw, list) or not shape_raw:
            raise QuantCalError(
                "E_FORMAT_INVALID", "tensor shape must be a non-empty array", tensor=name, stage="parse"
            )
        if any(not is_int(dim) or dim <= 0 for dim in shape_raw):
            raise QuantCalError(
                "E_FORMAT_INVALID", "tensor shape dimensions must be positive integers", tensor=name, stage="parse"
            )
        special_raw = row.get("special_positions", [])
        if not isinstance(special_raw, list) or any(not isinstance(item, dict) for item in special_raw):
            raise QuantCalError(
                "E_FORMAT_INVALID", "special_positions must be an array of objects", tensor=name, stage="parse"
            )
        result.append(
            TensorSpec(
                name=name,
                shape=tuple(int(dim) for dim in shape_raw),
                distribution=str(row.get("distribution", "unspecified")),
                special_positions=tuple(dict(item) for item in special_raw),
            )
        )
    if not result:
        raise QuantCalError("E_FORMAT_INVALID", "tensor manifest is empty", stage="parse")
    return result


def load_samples(path: str, tensor_specs: Iterable[TensorSpec]) -> list[CalibrationSample]:
    return parse_samples(load_jsonl(path), tensor_specs)


def parse_samples(rows: Any, tensor_specs: Iterable[TensorSpec]) -> list[CalibrationSample]:
    if not isinstance(rows, list):
        raise QuantCalError("E_FORMAT_INVALID", "calibration samples must be JSONL rows", stage="parse")
    specs = {spec.name: spec for spec in tensor_specs}
    samples: list[CalibrationSample] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise QuantCalError(
                "E_FORMAT_INVALID", f"sample row {index + 1} must be an object", stage="parse"
            )
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise QuantCalError(
                "E_FORMAT_INVALID", f"sample row {index + 1} has invalid sample_id", stage="parse"
            )
        if sample_id in seen:
            raise QuantCalError(
                "E_FORMAT_INVALID", "duplicate calibration sample_id", stage="parse", details={"sample_id": sample_id}
            )
        seen.add(sample_id)
        features = row.get("features")
        if not isinstance(features, dict):
            raise QuantCalError(
                "E_FORMAT_INVALID", f"sample {sample_id} features must be an object", stage="parse"
            )
        parsed: dict[str, tuple[float, ...]] = {}
        for tensor_name, raw_values in features.items():
            if tensor_name not in specs:
                raise QuantCalError(
                    "E_FORMAT_INVALID",
                    f"sample {sample_id} references unknown tensor",
                    tensor=tensor_name,
                    stage="parse",
                )
            if not isinstance(raw_values, list):
                raise QuantCalError(
                    "E_FORMAT_INVALID", "feature vector must be an array", tensor=tensor_name, stage="parse"
                )
            expected = math.prod(specs[tensor_name].shape)
            if len(raw_values) != expected:
                raise QuantCalError(
                    "E_FORMAT_INVALID",
                    f"feature length {len(raw_values)} does not match shape product {expected}",
                    tensor=tensor_name,
                    stage="parse",
                )
            parsed[tensor_name] = tuple(
                as_finite_float(value, "feature value", tensor_name) for value in raw_values
            )
        if not parsed:
            raise QuantCalError(
                "E_FORMAT_INVALID", f"sample {sample_id} has no features", stage="parse"
            )
        samples.append(CalibrationSample(sample_id=sample_id, features=parsed))
    if not samples:
        raise QuantCalError("E_FORMAT_INVALID", "no calibration samples found", stage="parse")
    samples.sort(key=lambda sample: sample.sample_id)
    return samples


def tensor_values_and_refs(
    tensor_name: str, samples: Iterable[CalibrationSample]
) -> tuple[list[float], list[str], list[dict[str, Any]]]:
    values: list[float] = []
    refs: list[str] = []
    layout: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        if tensor_name not in sample.features:
            continue
        refs.append(sample.sample_id)
        for element_index, value in enumerate(sample.features[tensor_name]):
            if not math.isfinite(value):
                raise QuantCalError(
                    "E_NON_FINITE_SAMPLE",
                    f"non-finite calibration value at sample {sample.sample_id}, element {element_index}",
                    tensor=tensor_name,
                    stage="calibrate",
                    details={
                        "sample_id": sample.sample_id,
                        "element_index": element_index,
                        "flat_index": len(values),
                        "value": str(value),
                    },
                )
            values.append(value)
            layout.append(
                {
                    "flat_index": len(values) - 1,
                    "sample_id": sample.sample_id,
                    "element_index": element_index,
                }
            )
    if not values:
        raise QuantCalError(
            "E_FORMAT_INVALID",
            "tensor has no calibration samples",
            tensor=tensor_name,
            stage="calibrate",
        )
    return values, refs, layout


def _quantile_params(resolved: ResolvedTensorConfig) -> dict[str, Any]:
    return {
        "lower_permille": resolved.quantile_lower_permille,
        "upper_permille": resolved.quantile_upper_permille,
        "method": QUANTILE_METHOD,
        "tie_rule": TIE_RULE,
        "numpy_version": np.__version__,
    }


def calibrate_quantiles(
    tensor_name: str,
    values: list[float],
    refs: list[str],
    layout: list[dict[str, Any]],
    resolved: ResolvedTensorConfig,
) -> dict[str, Any]:
    if any(not math.isfinite(value) for value in values):
        raise QuantCalError(
            "E_NON_FINITE_SAMPLE", "non-finite value reached quantile stage", tensor=tensor_name, stage="calibrate"
        )
    array = np.asarray(values, dtype=np.float64)
    sorted_array = np.sort(array, kind="mergesort")
    lower_probability = resolved.quantile_lower_permille / 1000.0
    upper_probability = resolved.quantile_upper_permille / 1000.0
    lower = float(np.quantile(sorted_array, lower_probability, method="linear"))
    upper = float(np.quantile(sorted_array, upper_probability, method="linear"))
    params = _quantile_params(resolved)
    payload = {
        "tensor": tensor_name,
        "point_count": len(values),
        "sample_ids": refs,
        "layout": layout,
        "values": values,
        "values_fingerprint": fingerprint(values),
        "quantile_lower": lower,
        "quantile_upper": upper,
        "quantile_params": params,
        "quantile_metadata": {
            "interpolation": QUANTILE_METHOD,
            "sort": "stable_ascending",
            "ties": TIE_RULE,
        },
    }
    return payload


def derive_scale_zero(
    tensor_name: str,
    quantiles: dict[str, Any],
    resolved: ResolvedTensorConfig,
) -> dict[str, Any]:
    lower = float(quantiles["quantile_lower"])
    upper = float(quantiles["quantile_upper"])
    bits = resolved.quantization_bits
    scheme = resolved.quantization_scheme
    qmin, qmax = quant_bounds(scheme, bits)
    guard_reason: str | None = None
    zero_point_rule = "fixed_zero"
    if scheme == "symmetric":
        max_abs = max(abs(lower), abs(upper))
        denominator = symmetric_saturation_max(bits)
        if max_abs == 0.0:
            scale = ZERO_RANGE_GUARD_SCALE
            guard_reason = "zero_abs_range;scale=0.001"
        else:
            scale = max_abs / denominator
        zero_point = 0
    else:
        span = upper - lower
        if span <= 0.0:
            absolute_level = max(abs(lower), abs(upper))
            if absolute_level > 0.0:
                scale = absolute_level / (qmax - qmin)
                guard_reason = "zero_span;scale_from_absolute_level"
            else:
                scale = ZERO_RANGE_GUARD_SCALE
                guard_reason = "zero_span;scale=0.001"
        else:
            scale = span / (qmax - qmin)
        raw_zero = qmin - lower / scale
        zero_point = int(np.rint(raw_zero))
        zero_point_rule = "nearest_even_then_clamp_qrange"
        clamped_zero = min(qmax, max(qmin, zero_point))
        if clamped_zero != zero_point:
            zero_point = clamped_zero
            guard_reason = (guard_reason + ";" if guard_reason else "") + "zero_point_clamped"
    if not math.isfinite(scale) or scale <= 0.0:
        scale = ZERO_RANGE_GUARD_SCALE
        guard_reason = "non_positive_scale;scale=0.001"
    return {
        "tensor": tensor_name,
        "quantization_bits": bits,
        "quantization_scheme": scheme,
        "quantization_bounds": [qmin, qmax],
        "scale": float(scale),
        "zero_point": int(zero_point),
        "scale_guard": guard_reason,
        "zero_point_rule": zero_point_rule,
        "derivation": {
            "symmetric_denominator": symmetric_saturation_max(bits) if scheme == "symmetric" else None,
            "asymmetric_integer_span": (qmax - qmin) if scheme == "asymmetric" else None,
            "zero_range_guard_scale": ZERO_RANGE_GUARD_SCALE,
            "tie_rounding": "nearest_even",
        },
    }


def export_quantile_range(lower: float, upper: float) -> tuple[float, float, bool]:
    """Return a strictly ordered v2 exchange range.

    Measured equal quantiles are retained in result.json; the exchange uses a
    fixed half-width guard window so the v2 format invariant remains strict.
    """

    if lower < upper:
        return lower, upper, False
    center = (lower + upper) / 2.0
    return (
        center - ZERO_SPAN_EXCHANGE_HALF_WIDTH,
        center + ZERO_SPAN_EXCHANGE_HALF_WIDTH,
        True,
    )


def exchange_scaled_result(
    scale_result: dict[str, Any], exchange_version: str
) -> dict[str, Any]:
    """Apply exchange representability before quantisation is measured."""

    scale = float(scale_result["scale"])
    zero_point = int(scale_result["zero_point"])
    scheme = str(scale_result["quantization_scheme"])
    payload = dict(scale_result)
    payload["calibrated_scale"] = scale
    payload["calibrated_zero_point"] = zero_point
    payload["calibrated_scheme"] = scheme
    if exchange_version == "v1":
        milli = int(math.floor(scale * 1000.0 + 0.5))
        if milli <= 0:
            milli = 1
            payload["exchange_scale_guard"] = "clamped_to_min_milli"
        scale = milli / 1000.0
        zero_point = 0
        scheme = "symmetric"
        payload["exchange_scale_milli"] = milli
        payload["exchange_loss"] = {
            "scale_quantized_to_milli": abs(scale * 1000.0 - milli) > 1e-12,
            "scale_milli_exact": scale * 1000.0,
            "scale_milli_rounded": milli,
            "scale_milli_residual": scale * 1000.0 - milli,
            "asymmetric_scheme_approximated_as_symmetric": (
                scale_result["quantization_scheme"] == "asymmetric"
            ),
            "zero_point_offset": int(scale_result["zero_point"]),
            "named_rounding_replaced_by": "nearest_even",
        }
    payload.update(
        {
            "scale": float(scale),
            "zero_point": int(zero_point),
            "quantization_scheme": scheme,
            "quantization_bounds": list(quant_bounds(scheme, int(scale_result["quantization_bits"]))),
            "saturation_max": scheme_saturation_max(scheme, int(scale_result["quantization_bits"])),
            "exchange_version": exchange_version,
        }
    )
    return payload
