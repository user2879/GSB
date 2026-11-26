from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import AuditError
from .quant import dequantize_int, fixed_exp_neg, fixed_sin_cos, qscale, quantize_with_trace, rounded_div


def angle_values(position: int, dimension: int, scheme: dict[str, Any]) -> list[float]:
    block_size = int(scheme["block_size"])
    correction_power = 1.0 + 0.125 * int(scheme["correction_level"])
    base = np.float64(scheme["base_freq"])
    scale = np.float64(scheme["angle_scale"])
    rows: list[float] = []
    for dim_index in range(dimension):
        block_index = dim_index // block_size
        exponent = np.float64(-(2.0 * block_index / dimension) * correction_power)
        inv_freq = np.power(base, exponent, dtype=np.float64)
        angle = np.float64(position) * scale * np.float64(inv_freq)
        rows.append(float(angle))
    return rows


def _build_angle_row_fixed(
    position: int,
    dimension: int,
    scheme: dict[str, Any],
    quant: dict[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise AuditError(code="E_POSITION_INVALID", message=f"{label} position must be a non-negative integer")
    raw_angles = angle_values(position, dimension, scheme)
    integers, trace = quantize_with_trace(raw_angles, quant, f"angle.{label}")
    bits = int(quant["effective_frac_bits"])
    rows: list[dict[str, Any]] = []
    for dim_index, (raw, angle_int, item_trace) in enumerate(zip(raw_angles, integers, trace)):
        sin_int, cos_int = fixed_sin_cos(angle_int, bits)
        rows.append(
            {
                "dim": dim_index,
                "angle_raw": raw,
                "angle_int": angle_int,
                "angle_q": dequantize_int(angle_int, qscale(quant)),
                "cos_int": cos_int,
                "sin_int": sin_int,
                "cos_q": dequantize_int(cos_int, qscale(quant)),
                "sin_q": dequantize_int(sin_int, qscale(quant)),
                "quant_trace": item_trace,
            }
        )
    return rows


def fixed_position_encoding(
    sample: dict[str, Any],
    scheme: dict[str, Any],
    quant: dict[str, Any],
    dimension: int,
) -> dict[str, Any]:
    query_position = sample["query_position"]
    key_positions = sample["key_positions"]
    query_row = _build_angle_row_fixed(query_position, dimension, scheme, quant, "query")
    key_rows = [
        _build_angle_row_fixed(position, dimension, scheme, quant, f"key[{index}]")
        for index, position in enumerate(key_positions)
    ]
    return {
        "interface": "rope_unified_v1",
        "scheme_name": scheme["name"],
        "sample_id": sample["id"],
        "dimension": dimension,
        "query_position": query_position,
        "key_positions": list(key_positions),
        "query_row": query_row,
        "key_rows": key_rows,
        "scale": qscale(quant),
        "effective_frac_bits": int(quant["effective_frac_bits"]),
        "quant_level": int(quant["quant_level"]),
    }


def position_vector_fixed(encoding: dict[str, Any]) -> list[float]:
    scale = int(encoding["scale"])
    values: list[float] = []
    for row in [encoding["query_row"], *encoding["key_rows"]]:
        for item in row:
            values.append(dequantize_int(item["cos_int"], scale))
            values.append(dequantize_int(item["sin_int"], scale))
    return values


def _quantized_rotation(
    vector: list[float],
    angle_row: list[dict[str, Any]],
    quant: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    if len(vector) != len(angle_row):
        raise AuditError(
            code="E_SHAPE_INVALID",
            message=f"{label} vector/angle shape mismatch: {len(vector)} != {len(angle_row)}",
        )
    scale = qscale(quant)
    input_int, trace = quantize_with_trace(vector, quant, label)
    output_int = [0] * len(vector)
    for dim_index in range(0, len(vector), 2):
        x0 = input_int[dim_index]
        x1 = input_int[dim_index + 1]
        cos_int = int(angle_row[dim_index]["cos_int"])
        sin_int = int(angle_row[dim_index]["sin_int"])
        output_int[dim_index] = rounded_div(x0 * cos_int - x1 * sin_int, scale)
        output_int[dim_index + 1] = rounded_div(x0 * sin_int + x1 * cos_int, scale)
    return {"input_int": input_int, "output_int": output_int, "quant_trace": trace}


def fixed_rotation(sample: dict[str, Any], encoding: dict[str, Any], quant: dict[str, Any]) -> dict[str, Any]:
    scale = qscale(quant)
    dimension = int(encoding["dimension"])
    query_vector = sample["query_vector"]
    key_vectors = sample["key_vectors"]
    if len(query_vector) != dimension or any(len(vector) != dimension for vector in key_vectors):
        raise AuditError(code="E_SHAPE_INVALID", message="feature vector length does not match declared dimension")
    query = _quantized_rotation(query_vector, encoding["query_row"], quant, "rotation.query")
    keys = [
        _quantized_rotation(vector, row, quant, f"rotation.key[{index}]")
        for index, (vector, row) in enumerate(zip(key_vectors, encoding["key_rows"]))
    ]
    return {"query": query, "keys": keys, "scale": scale}


def rotation_vector_fixed(rotation: dict[str, Any]) -> list[float]:
    scale = int(rotation["scale"])
    vectors = [rotation["query"]["output_int"], *[item["output_int"] for item in rotation["keys"]]]
    return [dequantize_int(value, scale) for vector in vectors for value in vector]


def fixed_dot_product(rotation: dict[str, Any]) -> dict[str, Any]:
    query = rotation["query"]["output_int"]
    scale = int(rotation["scale"])
    scores_int: list[int] = []
    accumulators: list[int] = []
    for key_rotation in rotation["keys"]:
        key_vector = key_rotation["output_int"]
        if len(key_vector) != len(query):
            raise AuditError(
                code="E_SHAPE_INVALID",
                message=f"dot product shape mismatch: query={len(query)}, key={len(key_vector)}",
            )
        accumulator = 0
        for dim_index in range(len(query)):
            accumulator += int(query[dim_index]) * int(key_vector[dim_index])
        accumulators.append(accumulator)
        scores_int.append(rounded_div(accumulator, scale))
    return {
        "accumulators_int": accumulators,
        "scores_int": scores_int,
        "scores": [dequantize_int(value, scale) for value in scores_int],
        "scale": scale,
        "dimension_order": "ascending",
        "key_order": "ascending_key_positions",
    }


def fixed_normalize(dot: dict[str, Any], force_zero_sum: bool = False) -> dict[str, Any]:
    scores = dot["scores_int"]
    scale = int(dot["scale"])
    if not scores:
        raise AuditError(code="E_EMPTY_SCORES", message="cannot normalize an empty score row")
    max_score = scores[0]
    for score in scores[1:]:
        if score > max_score:
            max_score = score
    shifted = [score - max_score for score in scores]
    exponentials = [fixed_exp_neg(-delta, scale) for delta in shifted]
    if force_zero_sum:
        exponentials = [0 for _ in exponentials]
    sum_exp = 0
    for value in exponentials:
        sum_exp += int(value)
    if sum_exp <= 0:
        raise AuditError(code="E_NORMALIZATION_INVALID", message="fixed-point softmax denominator is non-positive")
    probabilities_int = [rounded_div(value * scale, sum_exp) for value in exponentials]
    return {
        "max_score_int": max_score,
        "shifted_int": shifted,
        "exp_int": exponentials,
        "sum_exp_int": sum_exp,
        "probabilities_int": probabilities_int,
        "probabilities": [dequantize_int(value, scale) for value in probabilities_int],
        "scale": scale,
    }


def run_fixed_pipeline(
    sample: dict[str, Any],
    scheme: dict[str, Any],
    quant: dict[str, Any],
    dimension: int,
) -> dict[str, Any]:
    encoding = fixed_position_encoding(sample, scheme, quant, dimension)
    rotation = fixed_rotation(sample, encoding, quant)
    dot = fixed_dot_product(rotation)
    normalization = fixed_normalize(dot, bool(sample.get("_force_zero_norm_sum", False)))
    return {
        "position": encoding,
        "rotation": rotation,
        "dot": dot,
        "normalization": normalization,
        "position_vector": position_vector_fixed(encoding),
        "rotation_vector": rotation_vector_fixed(rotation),
    }


def _float_angle_trig(position: int, dimension: int, scheme: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    angles = angle_values(position, dimension, scheme)
    cosines = [float(np.cos(np.float64(angle), dtype=np.float64)) for angle in angles]
    sines = [float(np.sin(np.float64(angle), dtype=np.float64)) for angle in angles]
    return angles, cosines, sines


def _rotate_float_vector(vector: list[float], cosines: list[float], sines: list[float]) -> list[float]:
    if len(vector) != len(cosines):
        raise AuditError(code="E_SHAPE_INVALID", message="float rotation shape mismatch")
    result: list[np.float64] = []
    for dim_index in range(0, len(vector), 2):
        x0 = np.float64(vector[dim_index])
        x1 = np.float64(vector[dim_index + 1])
        cosine = np.float64(cosines[dim_index])
        sine = np.float64(sines[dim_index])
        result.append(x0 * cosine - x1 * sine)
        result.append(x0 * sine + x1 * cosine)
    return [float(value) for value in result]


def run_float_pipeline(sample: dict[str, Any], scheme: dict[str, Any], dimension: int) -> dict[str, Any]:
    query_angle, query_cos, query_sin = _float_angle_trig(sample["query_position"], dimension, scheme)
    key_trig = [_float_angle_trig(position, dimension, scheme) for position in sample["key_positions"]]
    position_vector: list[float] = []
    for cosines, sines in [(query_cos, query_sin), *[(item[1], item[2]) for item in key_trig]]:
        for cosine, sine in zip(cosines, sines):
            position_vector.extend([float(cosine), float(sine)])

    query_rotation = _rotate_float_vector(sample["query_vector"], query_cos, query_sin)
    key_rotations = [
        _rotate_float_vector(vector, trig[1], trig[2])
        for vector, trig in zip(sample["key_vectors"], key_trig)
    ]
    rotation_vector = list(query_rotation)
    for vector in key_rotations:
        rotation_vector.extend(vector)

    scores: list[float] = []
    for key_vector in key_rotations:
        accumulator = np.float64(0.0)
        for dim_index in range(dimension):
            accumulator = accumulator + np.float64(query_rotation[dim_index]) * np.float64(key_vector[dim_index])
        scores.append(float(accumulator))

    max_score = np.float64(scores[0])
    for score in scores[1:]:
        score64 = np.float64(score)
        if score64 > max_score:
            max_score = score64
    exponentials = [np.exp(np.float64(score) - max_score, dtype=np.float64) for score in scores]
    denominator = np.float64(0.0)
    for value in exponentials:
        denominator = denominator + value
    probabilities = [float(value / denominator) for value in exponentials]

    return {
        "angles": {"query": query_angle, "keys": [item[0] for item in key_trig]},
        "cos": {"query": query_cos, "keys": [item[1] for item in key_trig]},
        "sin": {"query": query_sin, "keys": [item[2] for item in key_trig]},
        "query_rotation": query_rotation,
        "key_rotations": key_rotations,
        "rotation_vector": rotation_vector,
        "scores": scores,
        "probabilities": probabilities,
        "position_vector": position_vector,
    }


def error_summary(fixed_values: list[float] | float, truth_values: list[float] | float) -> dict[str, Any]:
    fixed_list = [float(fixed_values)] if isinstance(fixed_values, (int, float)) else [float(v) for v in fixed_values]
    truth_list = [float(truth_values)] if isinstance(truth_values, (int, float)) else [float(v) for v in truth_values]
    if len(fixed_list) != len(truth_list):
        raise AuditError(code="E_ERROR_METRIC", message="error comparison lengths differ")

    max_abs = 0.0
    max_rel = 0.0
    zero_branch = False
    worst_abs: dict[str, Any] | None = None
    worst_rel: dict[str, Any] | None = None
    for index, (approx, truth) in enumerate(zip(fixed_list, truth_list)):
        absolute_error = abs(approx - truth)
        if truth == 0.0:
            relative_component = absolute_error
            zero_branch = True
            branch = "truth_zero_use_absolute"
        else:
            relative_component = absolute_error / abs(truth)
            branch = "relative_to_abs_truth"
        candidate_abs = {
            "index": index,
            "approx": approx,
            "truth": truth,
            "abs_error": absolute_error,
            "rel_error": relative_component,
            "branch": branch,
        }
        if absolute_error > max_abs:
            max_abs = absolute_error
            worst_abs = candidate_abs
        if relative_component > max_rel:
            max_rel = relative_component
            worst_rel = candidate_abs
    return {
        "abs_error": max_abs,
        "rel_error": max_rel,
        "relative_error_branch": "truth_zero_uses_absolute" if zero_branch else "relative_to_abs_truth",
        "zero_truth_branch": zero_branch,
        "count": len(fixed_list),
        "worst_abs": worst_abs,
        "worst_rel": worst_rel,
    }


def build_cell_errors(
    fixed: dict[str, Any],
    truth: dict[str, Any],
    sample: dict[str, Any],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    stage_errors = {
        "position": error_summary(fixed["position_vector"], truth["position_vector"]),
        "rotation": error_summary(fixed["rotation_vector"], truth["rotation_vector"]),
        "dot": error_summary(fixed["dot"]["scores"], truth["scores"]),
        "normalization": error_summary(fixed["normalization"]["probabilities"], truth["probabilities"]),
    }
    first_breach: dict[str, Any] | None = None
    for stage in ("position", "rotation", "dot", "normalization"):
        metric = stage_errors[stage]
        tolerance = float(tolerances[stage])
        exceeded = bool(metric["abs_error"] > tolerance)
        metric["tolerance"] = tolerance
        metric["exceeded"] = exceeded
        if exceeded and first_breach is None:
            first_breach = {
                "stage": stage,
                "code": "E_TOLERANCE_EXCEEDED",
                "metric": "abs_error",
                "value": metric["abs_error"],
                "tolerance": tolerance,
                "worst_index": metric["worst_abs"]["index"] if metric["worst_abs"] else None,
            }
    stage_errors["first_breach"] = first_breach

    key_metrics: list[dict[str, Any]] = []
    scale = int(fixed["position"]["scale"])
    for key_index, position in enumerate(sample["key_positions"]):
        pos_fixed: list[float] = []
        for item in fixed["position"]["key_rows"][key_index]:
            pos_fixed.extend([dequantize_int(item["cos_int"], scale), dequantize_int(item["sin_int"], scale)])
        pos_truth: list[float] = []
        for cosine, sine in zip(truth["cos"]["keys"][key_index], truth["sin"]["keys"][key_index]):
            pos_truth.extend([cosine, sine])
        relative_distance = int(sample["query_position"]) - int(position)
        stratum = "extrapolation" if relative_distance > int(sample["train_length"]) else "training"
        key_stage_errors = {
            "position": error_summary(pos_fixed, pos_truth),
            "rotation": error_summary(
                [dequantize_int(v, scale) for v in fixed["rotation"]["keys"][key_index]["output_int"]],
                truth["key_rotations"][key_index],
            ),
            "dot": error_summary(fixed["dot"]["scores"][key_index], truth["scores"][key_index]),
            "normalization": error_summary(
                fixed["normalization"]["probabilities"][key_index],
                truth["probabilities"][key_index],
            ),
        }
        for stage_name, metric in key_stage_errors.items():
            metric["tolerance"] = float(tolerances[stage_name])
            metric["exceeded"] = bool(metric["abs_error"] > float(tolerances[stage_name]))
        key_metrics.append(
            {
                "key_index": key_index,
                "key_position": int(position),
                "relative_distance": relative_distance,
                "stratum": stratum,
                "stages": key_stage_errors,
                "fixed_score": fixed["dot"]["scores"][key_index],
                "truth_score": truth["scores"][key_index],
                "fixed_probability": fixed["normalization"]["probabilities"][key_index],
                "truth_probability": truth["probabilities"][key_index],
            }
        )
    return {"stages": stage_errors, "keys": key_metrics}
