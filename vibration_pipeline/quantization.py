from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .algorithms import RawScore
from .config import PipelineConfig, QuantizationConfig
from .errors import E_CONFIG_INVALID, PipelineError
from .util import hash_object, quantize_half_away, stable_float


@dataclass(frozen=True)
class QuantizedScore:
    algorithm: str
    raw_score: float
    normalized_score: float
    normalized_ticks: int
    cache_key: str
    raw_cache_key: str
    normalization_hash: str
    quantization_fingerprint: str
    details: dict[str, Any]


@dataclass(frozen=True)
class Decision:
    status: str
    score: float
    score_ticks: int
    threshold: float
    threshold_ticks: int
    normalized_total: float
    anomaly: bool
    boundary: bool
    boundary_reason: str | None
    cache_key: str
    algorithms_used: tuple[str, ...]
    reference: dict[str, Any]


def normalize_raw_score(raw_score: float, normalization: dict[str, Any]) -> float:
    method = normalization.get("method", "identity_clip")
    raw = float(raw_score)
    if method == "identity_clip":
        reference = float(normalization.get("reference", 1.0))
        if reference <= 0:
            raise PipelineError(E_CONFIG_INVALID, "normalization reference must be > 0")
        value = raw / reference
    elif method == "log1p_clip":
        reference = float(normalization.get("reference", 1.0))
        if reference <= 0:
            raise PipelineError(E_CONFIG_INVALID, "normalization reference must be > 0")
        value = math.log1p(max(raw, 0.0)) / math.log1p(reference)
    elif method == "minmax_clip":
        lower = float(normalization.get("min", 0.0))
        upper = float(normalization.get("max", 1.0))
        if lower >= upper:
            raise PipelineError(E_CONFIG_INVALID, "normalization min must be below max")
        value = (raw - lower) / (upper - lower)
    else:
        raise PipelineError(E_CONFIG_INVALID, f"unsupported normalization method {method}")
    return stable_float(min(1.0, max(0.0, value)))


def quantized_score_cache_key(
    raw_score: RawScore,
    normalization_hash: str,
    quantization: QuantizationConfig,
) -> str:
    return hash_object(
        {
            "raw_score_cache_key": raw_score.cache_key,
            "normalization_hash": normalization_hash,
            "quantization_fingerprint": quantization.fingerprint,
            "formula": "normalized-fixed-point-score-v1",
        },
        namespace="quantized-algorithm-score-key-v1",
    )


def quantize_score(
    raw_result: RawScore,
    config: PipelineConfig,
) -> QuantizedScore:
    spec = config.algorithms[raw_result.algorithm]
    normalized = normalize_raw_score(raw_result.raw_score, spec.normalization)
    ticks = quantize_half_away(normalized, config.quantization.scale, config.quantization.quantum)
    ticks = max(0, min(config.quantization.scale, ticks))
    key = quantized_score_cache_key(raw_result, spec.normalization_hash, config.quantization)
    return QuantizedScore(
        algorithm=raw_result.algorithm,
        raw_score=raw_result.raw_score,
        normalized_score=normalized,
        normalized_ticks=ticks,
        cache_key=key,
        raw_cache_key=raw_result.cache_key,
        normalization_hash=spec.normalization_hash,
        quantization_fingerprint=config.quantization.fingerprint,
        details={
            "raw_cache_key": raw_result.cache_key,
            "normalization": dict(spec.normalization),
            "weight": spec.weight,
            "decimal_places": config.quantization.decimal_places,
            "quantum": config.quantization.quantum,
        },
    )


def _ordered_positive_scores(scores: list[QuantizedScore], config: PipelineConfig) -> list[QuantizedScore]:
    useful = [score for score in scores if config.algorithms[score.algorithm].weight > 0]
    return sorted(useful, key=lambda score: score.algorithm)


def fused_score(scores: list[QuantizedScore], config: PipelineConfig) -> tuple[float, int, float, list[dict[str, Any]]]:
    ordered = _ordered_positive_scores(scores, config)
    if not ordered:
        raise PipelineError(E_CONFIG_INVALID, "no positive-weight algorithm score is available")
    total_weight = sum(float(config.algorithms[score.algorithm].weight) for score in ordered)
    scale = config.quantization.scale
    weighted_ticks = 0.0
    reference_scores: list[dict[str, Any]] = []
    for score in ordered:
        weight = float(config.algorithms[score.algorithm].weight)
        weighted_ticks += weight * (score.normalized_ticks / scale)
        reference_scores.append(
            {
                "algorithm": score.algorithm,
                "weight": weight,
                "normalized_score": score.normalized_score,
                "normalized_ticks": score.normalized_ticks,
                "raw_score": score.raw_score,
            }
        )
    normalized_total = stable_float(weighted_ticks / total_weight)
    raw_total = config.score_min + normalized_total * (config.score_max - config.score_min)
    raw_total = min(config.score_max, max(config.score_min, raw_total))
    total_ticks = quantize_half_away(raw_total, scale, config.quantization.quantum)
    min_ticks = quantize_half_away(config.score_min, scale, config.quantization.quantum)
    max_ticks = quantize_half_away(config.score_max, scale, config.quantization.quantum)
    total_ticks = max(min_ticks, min(max_ticks, total_ticks))
    quantized_total = total_ticks / scale
    return stable_float(quantized_total), int(total_ticks), normalized_total, reference_scores


def decision_cache_key(
    *,
    device_id: str,
    segment_index: int,
    waveform_fingerprint: str,
    preprocessing_fingerprint: str,
    quantized_scores: list[QuantizedScore],
    config: PipelineConfig,
    threshold_ticks: int,
) -> str:
    ordered = sorted(quantized_scores, key=lambda score: score.algorithm)
    return hash_object(
        {
            "formula": "segment-fusion-decision-v1",
            "device_id": device_id,
            "segment_index": segment_index,
            "waveform_fingerprint": waveform_fingerprint,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "quantized_score_keys": [score.cache_key for score in ordered],
            "weights": {score.algorithm: config.algorithms[score.algorithm].weight for score in ordered},
            "score_range": [config.score_min, config.score_max],
            "threshold_ticks": threshold_ticks,
            "quantization": {
                "decimal_places": config.quantization.decimal_places,
                "quantum": config.quantization.quantum,
                "tie_policy": config.quantization.tie_policy,
            },
        },
        namespace="segment-decision-key-v1",
    )


def make_decision(
    *,
    device_id: str,
    segment_index: int,
    waveform_fingerprint: str,
    preprocessing_fingerprint: str,
    quantized_scores: list[QuantizedScore],
    config: PipelineConfig,
    effective_threshold: float,
) -> Decision:
    threshold = min(config.score_max, max(config.score_min, float(effective_threshold)))
    threshold_ticks = quantize_half_away(threshold, config.quantization.scale, config.quantization.quantum)
    score, score_ticks, normalized_total, reference_scores = fused_score(quantized_scores, config)
    boundary = int(score_ticks) == int(threshold_ticks)
    if config.quantization.tie_policy == "anomaly":
        anomaly = score_ticks >= threshold_ticks
    else:
        anomaly = score_ticks > threshold_ticks
    boundary_reason = None
    if boundary:
        boundary_reason = f"score_ticks == threshold_ticks; tie_policy={config.quantization.tie_policy}"
    key = decision_cache_key(
        device_id=device_id,
        segment_index=segment_index,
        waveform_fingerprint=waveform_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        quantized_scores=quantized_scores,
        config=config,
        threshold_ticks=threshold_ticks,
    )
    algorithms = tuple(sorted(score.algorithm for score in quantized_scores if config.algorithms[score.algorithm].weight > 0))
    reference = {
        "score_ticks": int(score_ticks),
        "threshold_ticks": int(threshold_ticks),
        "threshold": stable_float(threshold),
        "normalized_total": normalized_total,
        "score_range": [config.score_min, config.score_max],
        "algorithms": reference_scores,
        "tie_policy": config.quantization.tie_policy,
        "quantization_fingerprint": config.quantization.fingerprint,
    }
    return Decision(
        status="ok",
        score=score,
        score_ticks=int(score_ticks),
        threshold=stable_float(threshold),
        threshold_ticks=int(threshold_ticks),
        normalized_total=normalized_total,
        anomaly=bool(anomaly),
        boundary=bool(boundary),
        boundary_reason=boundary_reason,
        cache_key=key,
        algorithms_used=algorithms,
        reference=reference,
    )
