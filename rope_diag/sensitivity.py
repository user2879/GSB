from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .attention import ScoreResult, score_attention
from .config import DiagnosticConfig
from .errors import EncodingBoundaryError
from .rope import (
    apply_input_precision_and_quantization,
    apply_rotation,
    apply_rotation_batch,
    generate_position_encoding,
)
from .samples import AttentionSample


@dataclass(frozen=True)
class Candidate:
    parameter: str
    value: float
    step: float
    offset: int


def candidate_values(base: float, scaling: float, config: DiagnosticConfig) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, float]] = set()
    radius = config.sensitivity.radius
    for offset in range(-radius, radius + 1):
        value = base + offset * config.sensitivity.base_step
        key = ("base", float(value))
        if key not in seen:
            seen.add(key)
            candidates.append(
                Candidate("base", float(value), config.sensitivity.base_step, offset)
            )
    for offset in range(-radius, radius + 1):
        value = scaling + offset * config.sensitivity.scale_step
        key = ("scaling_factor", float(value))
        if key not in seen:
            seen.add(key)
            candidates.append(
                Candidate("scaling_factor", float(value), config.sensitivity.scale_step, offset)
            )
    return candidates


def _candidate_scores(
    scheme: str,
    samples: Sequence[AttentionSample],
    config: DiagnosticConfig,
    dimensions: Sequence[int],
    chunk_size: int,
    candidate: Candidate,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    scores_by_sample: dict[str, np.ndarray] = {}
    boundaries: list[dict[str, Any]] = []
    for sample in samples:
        base_override = candidate.value if candidate.parameter == "base" else None
        scaling_override = candidate.value if candidate.parameter == "scaling_factor" else None
        try:
            query_encoding = generate_position_encoding(
                scheme,
                sample.query_position,
                config,
                base_override=base_override,
                scaling_override=scaling_override,
            )
            key_encodings = tuple(
                generate_position_encoding(
                    scheme,
                    position,
                    config,
                    base_override=base_override,
                    scaling_override=scaling_override,
                )
                for position in sample.key_positions
            )
            query_prepared = apply_input_precision_and_quantization(sample.query_features, config)
            keys_prepared = np.vstack(
                [
                    apply_input_precision_and_quantization(row, config)
                    for row in sample.key_features
                ]
            )
            query_rotated = apply_rotation(query_prepared, query_encoding)
            keys_rotated = apply_rotation_batch(keys_prepared, key_encodings)
            result = score_attention(query_rotated, keys_rotated, dimensions, chunk_size)
            scores_by_sample[sample.sample_id] = result.scores
        except EncodingBoundaryError as exc:
            boundaries.append(
                {
                    "sample_id": sample.sample_id,
                    "condition": exc.details.get("condition", "encoding_boundary"),
                    "details": exc.details,
                }
            )
    return scores_by_sample, boundaries


def scan_scheme_sensitivity(
    scheme: str,
    samples: Sequence[AttentionSample],
    baseline_scores: Mapping[str, ScoreResult],
    config: DiagnosticConfig,
    dimensions: Sequence[int],
    chunk_size: int,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for candidate in candidate_values(config.base, config.scaling_factor, config):
        candidate_scores, boundaries = _candidate_scores(
            scheme,
            samples,
            config,
            dimensions,
            chunk_size,
            candidate,
        )
        differences: list[float] = []
        affected_samples: list[str] = []
        if not boundaries:
            for sample in samples:
                baseline = baseline_scores.get(sample.sample_id)
                candidate_values_for_sample = candidate_scores.get(sample.sample_id)
                if baseline is None or candidate_values_for_sample is None:
                    continue
                distances = sample.relative_distances()
                mask = distances >= config.training_context_length
                if not np.any(mask):
                    continue
                delta = np.abs(
                    candidate_values_for_sample[mask].astype(np.float64)
                    - baseline.scores[mask].astype(np.float64)
                )
                differences.extend(float(value) for value in delta.tolist())
                if float(np.mean(delta)) > 0.0:
                    affected_samples.append(sample.sample_id)
        deviation = float(np.mean(np.asarray(differences, dtype=np.float64))) if differences else 0.0
        entries.append(
            {
                "scheme": scheme,
                "parameter": candidate.parameter,
                "value": candidate.value,
                "offset": candidate.offset,
                "step": candidate.step,
                "deviation_abs": None if boundaries else deviation,
                "boundary_triggered": bool(boundaries),
                "boundary_samples": boundaries,
                "affected_sample_ids": sorted(set(affected_samples)),
                "extrapolation_score_count": len(differences),
            }
        )

    entries.sort(
        key=lambda item: (
            0 if item["boundary_triggered"] else 1,
            0.0 if item["boundary_triggered"] else -float(item["deviation_abs"]),
            float(item["value"]),
            item["parameter"],
        )
    )
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank

    return {
        "status": "ok",
        "scheme": scheme,
        "neighbor_steps": {
            "base": config.sensitivity.base_step,
            "scaling_factor": config.sensitivity.scale_step,
            "radius": config.sensitivity.radius,
        },
        "sort_rule": (
            "boundary_triggered_first; then deviation_abs descending; "
            "ties by parameter value ascending"
        ),
        "entries": entries,
        "boundary_entry_count": sum(1 for entry in entries if entry["boundary_triggered"]),
    }

