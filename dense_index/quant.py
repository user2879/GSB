from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import QUANT_LEVEL_BITS, ConfigBundle
from .errors import E_BOUNDARY_FLIP, E_VECTOR_NONFINITE, DiagnosticError


def assert_finite_vector(vector: Sequence[float], context: str, injection_id: Optional[str] = None) -> None:
    array = np.asarray(vector, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        bad = [index for index, value in enumerate(array) if not math.isfinite(float(value))]
        raise DiagnosticError(
            E_VECTOR_NONFINITE,
            f"non-finite vector component in {context}",
            injection_id=injection_id,
        )


def quantize_vector(
    vector: Sequence[float],
    config: ConfigBundle,
    effective_fractional_bits: Optional[int] = None,
    total_bits: Optional[int] = None,
    injection_id: Optional[str] = None,
) -> Dict[str, Any]:
    raw = np.asarray(vector, dtype=np.float64)
    assert_finite_vector(raw, "quantize_vector", injection_id)
    eff_bits = int(config.effective_fractional_bits if effective_fractional_bits is None else effective_fractional_bits)
    bits = int(QUANT_LEVEL_BITS[config.data["quant_level"]] if total_bits is None else total_bits)
    scale = float(2 ** eff_bits)
    max_value = float(2 ** (bits - 1) - 1)

    scaled = raw * scale
    signs = np.sign(scaled)
    absolute = np.abs(scaled)
    integers = signs * np.floor(absolute + 0.5)
    clipped = np.clip(integers, -max_value, max_value)
    fixed = clipped.astype(np.int64)
    step = 1.0 / scale
    residuals = (raw - fixed.astype(np.float64) * step).tolist()
    clipped_coordinates = [index for index in range(len(fixed)) if integers[index] != clipped[index]]
    return {
        "fixed_vector": fixed.tolist(),
        "scale": scale,
        "quant_step": step,
        "effective_fractional_bits": eff_bits,
        "total_bits": bits,
        "max_value": int(max_value),
        "residuals": residuals,
        "clipped_coordinates": clipped_coordinates,
    }


def bucket_key_from_vector(
    raw_vector: Sequence[float],
    fixed_vector: Sequence[int],
    config: ConfigBundle,
    alternative_tie_rule: Optional[str] = None,
) -> Dict[str, Any]:
    bits = int(config.data["bucket_bits"])
    tie_rule = alternative_tie_rule or str(config.data["sign_tie_rule"])
    step = 2.0 ** (-int(config.effective_fractional_bits))
    key = 0
    tie_breaks: List[Dict[str, Any]] = []

    for coordinate in range(bits):
        raw_value = float(raw_vector[coordinate])
        fixed_value = int(fixed_vector[coordinate])
        bit_position = bits - 1 - coordinate
        within_half_step = abs(raw_value) <= 0.5 * step + 1e-15
        if fixed_value == 0 or within_half_step:
            if raw_value == 0.0:
                tie_kind = "exact_zero"
            elif abs(raw_value) <= 0.5 * step - 1e-15:
                tie_kind = "quantized_zero"
            else:
                tie_kind = "half_step_boundary"
            chosen_negative = tie_rule == "negative"
            bit = 1 if chosen_negative else 0
            tie_breaks.append(
                {
                    "coordinate": coordinate,
                    "bit_position": bit_position,
                    "raw_value": raw_value,
                    "fixed_value": fixed_value,
                    "distance_to_zero": abs(raw_value),
                    "half_step": 0.5 * step,
                    "tie_kind": tie_kind,
                    "tie_rule": tie_rule,
                    "chosen_bit": bit,
                    "chosen_sign": "negative" if bit == 1 else "positive",
                }
            )
        else:
            bit = 1 if fixed_value < 0 else 0
        key |= bit << bit_position

    return {
        "bucket_key": key,
        "bucket_id": f"B-{key:04d}",
        "tie_breaks": tie_breaks,
        "tie_rule_used": tie_rule,
    }


def fixed_cosine(query: Sequence[int], block: Sequence[int]) -> float:
    q = np.asarray(query, dtype=np.float64)
    b = np.asarray(block, dtype=np.float64)
    norm_q = float(np.linalg.norm(q))
    norm_b = float(np.linalg.norm(b))
    if norm_q == 0.0 or norm_b == 0.0:
        return 0.0
    score = float(np.dot(q, b) / (norm_q * norm_b))
    return max(-1.0, min(1.0, score))


def half_step_score_band(query: Sequence[int], block: Sequence[int]) -> float:
    q = np.asarray(query, dtype=np.float64)
    b = np.asarray(block, dtype=np.float64)
    norm_q = float(np.linalg.norm(q)) or 1.0
    norm_b = float(np.linalg.norm(b)) or 1.0
    denominator = norm_q * norm_b
    return 0.5 * (float(np.abs(q).sum()) + float(np.abs(b).sum())) / denominator


def reference_similarity(
    query_raw: Sequence[float],
    block_raw: Sequence[float],
    config: ConfigBundle,
) -> Dict[str, Any]:
    query_wide = quantize_vector(query_raw, config, effective_fractional_bits=config.effective_fractional_bits + 1, total_bits=63)
    block_wide = quantize_vector(block_raw, config, effective_fractional_bits=config.effective_fractional_bits + 1, total_bits=63)
    score = fixed_cosine(query_wide["fixed_vector"], block_wide["fixed_vector"])
    return {
        "reference_score": score,
        "reference_effective_fractional_bits": config.effective_fractional_bits + 1,
        "reference_quant_step": query_wide["quant_step"],
    }


def diagnose_pair(
    query_raw: Sequence[float],
    query_fixed: Sequence[int],
    block_raw: Sequence[float],
    block_fixed: Sequence[int],
    block_id: str,
    config: ConfigBundle,
    threshold: Optional[float] = None,
    query_id: str = "query",
) -> Dict[str, Any]:
    threshold = float(config.data["similarity_threshold"] if threshold is None else threshold)
    score = fixed_cosine(query_fixed, block_fixed)
    wide = reference_similarity(query_raw, block_raw, config)
    band = half_step_score_band(query_fixed, block_fixed)
    difference = abs(wide["reference_score"] - score)
    exact_threshold = score == threshold
    threshold_gap = abs(score - threshold)
    boundary = exact_threshold or threshold_gap <= band + 1e-12 or difference <= band + 1e-12
    current_decision = bool(score >= threshold)
    reference_decision = bool(wide["reference_score"] >= threshold)
    flip = current_decision != reference_decision
    cause = None
    if flip:
        cause = "quantization_truncation"
    return {
        "query_id": query_id,
        "block_id": block_id,
        "score": score,
        "threshold": threshold,
        "threshold_gap": threshold_gap,
        "exact_threshold": exact_threshold,
        "half_step_band": band,
        "reference_score": wide["reference_score"],
        "reference_effective_fractional_bits": wide["reference_effective_fractional_bits"],
        "reference_difference": difference,
        "boundary": boundary,
        "current_decision": current_decision,
        "reference_decision": reference_decision,
        "decision": current_decision,
        "flip": flip,
        "code": E_BOUNDARY_FLIP if flip else None,
        "cause": cause,
        "lexicographic_key": block_id,
    }


def routing_tie_flip(
    query_bucket: Dict[str, Any],
    block_bucket: Dict[str, Any],
    query_alternative: Dict[str, Any],
    block_alternative: Dict[str, Any],
) -> Dict[str, Any]:
    current_shared = query_bucket["bucket_id"] == block_bucket["bucket_id"]
    alternative_shared = query_alternative["bucket_id"] == block_alternative["bucket_id"]
    flip = current_shared != alternative_shared
    return {
        "flip": flip,
        "code": E_BOUNDARY_FLIP if flip else None,
        "cause": "tie_rule" if flip else None,
        "current_query_bucket": query_bucket["bucket_id"],
        "current_block_bucket": block_bucket["bucket_id"],
        "alternative_query_bucket": query_alternative["bucket_id"],
        "alternative_block_bucket": block_alternative["bucket_id"],
        "current_shared_bucket": current_shared,
        "alternative_shared_bucket": alternative_shared,
    }
