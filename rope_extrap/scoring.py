from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .canonical import sha256_json
from .errors import E_COMPUTE_NONFINITE, BusinessError
from .models import AttentionSample, EncodedSample, KeyScore, PositionEncodingConfig, ScoreResult


def calculate_scores(
    config: PositionEncodingConfig,
    sample: AttentionSample,
    encoded: EncodedSample,
) -> ScoreResult:
    query = np.asarray(encoded.query_vector, dtype=np.float64)
    keys = np.asarray(encoded.key_vectors, dtype=np.float64)
    contributions = keys * query[np.newaxis, :]
    raw_scores = np.sum(contributions, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(raw_scores)) or not np.all(np.isfinite(contributions)):
        raise BusinessError(
            E_COMPUTE_NONFINITE,
            "attention score produced a non-finite value",
            stage="attention_score",
            sample_id=sample.sample_id,
        )
    shifted = raw_scores - float(np.max(raw_scores))
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials, dtype=np.float64))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise BusinessError(
            E_COMPUTE_NONFINITE,
            "softmax denominator is not a positive finite value",
            stage="attention_score",
            sample_id=sample.sample_id,
        )
    weights = exponentials / denominator

    entries: list[KeyScore] = []
    for index, key_position in enumerate(encoded.key_positions):
        distance = int(sample.query_position - key_position)
        interval = "in_training" if distance <= config.training_context_length else "extrapolation"
        row = contributions[index]
        dominant_dimension = int(np.argmax(np.abs(row))) if row.size else 0
        entries.append(
            KeyScore(
                key_position=int(key_position),
                relative_distance=distance,
                interval=interval,
                score=float(raw_scores[index]),
                attention_weight=float(weights[index]),
                dominant_dimension=dominant_dimension,
                dominant_contribution=float(row[dominant_dimension]) if row.size else 0.0,
                contributions=tuple(float(value) for value in row),
            )
        )
    has_extrapolation = any(entry.interval == "extrapolation" for entry in entries)
    all_abs = np.abs(contributions)
    if all_abs.size:
        dominant_flat = int(np.argmax(all_abs))
        dominant_dimension = dominant_flat % config.dimension
        dominant_contribution_abs = float(all_abs.flat[dominant_flat])
    else:
        dominant_dimension = 0
        dominant_contribution_abs = 0.0
    fingerprint = sha256_json(
        {
            "scheme": encoded.scheme,
            "position_fingerprint": encoded.position_fingerprint,
            "training_context_length": config.training_context_length,
            "scores": [entry.score for entry in entries],
            "weights": [entry.attention_weight for entry in entries],
        }
    )
    return ScoreResult(
        entries=tuple(entries),
        has_extrapolation=has_extrapolation,
        sample_interval="extrapolation" if has_extrapolation else "in_training",
        dominant_dimension=dominant_dimension,
        dominant_contribution_abs=dominant_contribution_abs,
        raw_scores=tuple(float(value) for value in raw_scores),
        attention_weights=tuple(float(value) for value in weights),
        score_fingerprint=fingerprint,
    )


def score_entry_dict(entry: KeyScore) -> dict[str, object]:
    return {
        "key_position": entry.key_position,
        "relative_distance": entry.relative_distance,
        "interval": entry.interval,
        "score": entry.score,
        "attention_weight": entry.attention_weight,
        "dominant_dimension": entry.dominant_dimension,
        "dominant_contribution": entry.dominant_contribution,
    }


def _range_stats(entries: Iterable[KeyScore], interval: str) -> dict[str, object]:
    selected = [entry for entry in entries if entry.interval == interval]
    if not selected:
        return {
            "interval": interval,
            "sample_count": 0,
            "key_count": 0,
            "mean_score": None,
            "max_score": None,
            "mean_attention_weight": None,
            "max_attention_weight": None,
            "sample_ids": [],
        }
    scores = [entry.score for entry in selected]
    weights = [entry.attention_weight for entry in selected]
    return {
        "interval": interval,
        "sample_count": 0,
        "key_count": len(selected),
        "mean_score": float(np.mean(scores, dtype=np.float64)),
        "max_score": float(np.max(scores)),
        "mean_attention_weight": float(np.mean(weights, dtype=np.float64)),
        "max_attention_weight": float(np.max(weights)),
        "sample_ids": [],
    }


def aggregate_score_statistics(score_results: dict[str, ScoreResult]) -> dict[str, object]:
    all_entries: list[tuple[str, KeyScore]] = []
    for sample_id in sorted(score_results):
        all_entries.extend((sample_id, entry) for entry in score_results[sample_id].entries)
    entries_only = [entry for _, entry in all_entries]

    def range_with_samples(interval: str) -> dict[str, object]:
        rows = [(sample_id, entry) for sample_id, entry in all_entries if entry.interval == interval]
        result = _range_stats([entry for _, entry in rows], interval)
        result["sample_count"] = len({sample_id for sample_id, _ in rows})
        result["sample_ids"] = sorted({sample_id for sample_id, _ in rows})
        return result

    grouped: dict[int, list[tuple[str, KeyScore]]] = defaultdict(list)
    for sample_id, entry in all_entries:
        grouped[entry.relative_distance].append((sample_id, entry))
    by_distance: list[dict[str, object]] = []
    for distance in sorted(grouped):
        rows = grouped[distance]
        interval = "in_training" if distance <= 0 else rows[0][1].interval
        scores = [entry.score for _, entry in rows]
        weights = [entry.attention_weight for _, entry in rows]
        by_distance.append(
            {
                "relative_distance": distance,
                "interval": interval,
                "sample_count": len({sample_id for sample_id, _ in rows}),
                "key_count": len(rows),
                "mean_score": float(np.mean(scores, dtype=np.float64)),
                "max_score": float(np.max(scores)),
                "mean_attention_weight": float(np.mean(weights, dtype=np.float64)),
                "max_attention_weight": float(np.max(weights)),
            }
        )

    sample_counts = {
        "in_training": sum(not result.has_extrapolation for result in score_results.values()),
        "extrapolation": sum(result.has_extrapolation for result in score_results.values()),
    }
    return {
        "in_training": range_with_samples("in_training"),
        "extrapolation": range_with_samples("extrapolation"),
        "by_relative_distance": by_distance,
        "sample_distribution": sample_counts,
        "key_count": len(entries_only),
    }


def aggregate_dimension_sensitivity(score_results: dict[str, ScoreResult], dimension: int) -> list[dict[str, object]]:
    totals = np.zeros(dimension, dtype=np.float64)
    external_totals = np.zeros(dimension, dtype=np.float64)
    for sample_id in sorted(score_results):
        for entry in score_results[sample_id].entries:
            contributions = np.abs(np.asarray(entry.contributions, dtype=np.float64))
            totals += contributions
            if entry.interval == "extrapolation":
                external_totals += contributions
    grand_total = float(np.sum(totals, dtype=np.float64))
    external_grand_total = float(np.sum(external_totals, dtype=np.float64))
    rows: list[dict[str, object]] = []
    for index in range(dimension):
        rows.append(
            {
                "dimension": index,
                "total_abs_contribution": float(totals[index]),
                "extrapolation_abs_contribution": float(external_totals[index]),
                "share": float(totals[index] / grand_total) if grand_total > 0.0 else 0.0,
                "extrapolation_share": (
                    float(external_totals[index] / external_grand_total) if external_grand_total > 0.0 else 0.0
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["extrapolation_abs_contribution"]),
            -float(row["total_abs_contribution"]),
            int(row["dimension"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
