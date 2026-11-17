"""Base-frequency and scaling-factor sensitivity and boundary diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .attention import AttentionResult, dot_scores
from .config import DiagnosticConfig
from .encoding import SampleEncoding, encode_positions, probe_positions, rotate_features
from .errors import WorkbenchError
from .samples import Sample
from .util import clean_float


def _scan_encoding(
    scheme: str,
    sample: Sample,
    config: DiagnosticConfig,
    base: float,
    scaling_factor: float,
) -> tuple[SampleEncoding | None, dict[str, Any] | None]:
    positions = np.asarray([sample.query_position, *sample.key_positions], dtype=np.int64)
    _, info = probe_positions(
        scheme,
        positions,
        config.dimension,
        base,
        scaling_factor,
        config.precision,
    )
    if info is not None:
        return None, info
    encoding = encode_positions(
        scheme,
        positions,
        config.dimension,
        base,
        scaling_factor,
        config.precision,
    )
    rotated = rotate_features(np.vstack([sample.query_array[None, :], sample.key_array]), encoding)
    return (
        SampleEncoding(
            scheme=scheme,
            sample_id=sample.sample_id,
            base=base,
            scaling_factor=scaling_factor,
            dimension=config.dimension,
            precision=config.precision,
            positions=positions,
            effective_positions=encoding.effective_positions,
            effective_base=encoding.effective_base,
            inv_freq=encoding.inv_freq,
            cos_values=encoding.cos_values,
            sin_values=encoding.sin_values,
            query_encoded=rotated[0],
            key_encoded=rotated[1:],
        ),
        None,
    )


def _boundary_row(
    *,
    scheme: str,
    sample_id: str,
    parameter: str,
    parameter_value: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scheme": scheme,
        "sample_id": sample_id,
        "parameter": parameter,
        "parameter_value": clean_float(parameter_value),
        "criteria": str(info.get("criteria", "invalid_parameter")),
        "reason": str(info.get("reason", "parameter cannot produce a finite encoding")),
        "trigger_dimensions": [int(item) for item in info.get("trigger_dimensions", [])],
    }


def _scan_parameter(
    *,
    config: DiagnosticConfig,
    scheme: str,
    parameter: str,
    values: tuple[float, ...],
    baseline_value: float,
    samples: list[Sample],
    baselines: dict[str, AttentionResult],
    train_context_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []

    for parameter_value in values:
        base = parameter_value if parameter == "base" else config.base
        scaling_factor = parameter_value if parameter == "scaling_factor" else config.scaling_factor
        deviations: list[float] = []
        reference_deviations: list[float] = []
        contributing_samples: set[str] = set()
        boundary_samples: set[str] = set()

        for sample in samples:
            baseline = baselines.get(sample.sample_id)
            if baseline is None:
                continue
            has_extrapolation = any(layer == "out_of_train" for layer in baseline.layers)
            try:
                scanned, info = _scan_encoding(
                    scheme,
                    sample,
                    config,
                    base,
                    scaling_factor,
                )
            except WorkbenchError as exc:
                info = {
                    "criteria": "invalid_parameter",
                    "reason": exc.message,
                    "trigger_dimensions": [],
                }
                scanned = None
            if info is not None:
                if has_extrapolation:
                    boundary_samples.add(sample.sample_id)
                    boundary_rows.append(
                        _boundary_row(
                            scheme=scheme,
                            sample_id=sample.sample_id,
                            parameter=parameter,
                            parameter_value=parameter_value,
                            info=info,
                        )
                    )
                continue
            assert scanned is not None
            scores = dot_scores(scanned.query_encoded, scanned.key_encoded, config.dimension)
            for key_index, layer in enumerate(baseline.layers):
                if layer != "out_of_train":
                    continue
                deviations.append(abs(float(scores[key_index]) - float(baseline.scores[key_index])))
                reference_deviations.append(
                    abs(float(scores[key_index]) - float(baseline.reference_scores[key_index]))
                )
                contributing_samples.add(sample.sample_id)

        if not deviations:
            status = "boundary"
        elif boundary_samples:
            status = "partial_boundary"
        else:
            status = "scored"
        entries.append(
            {
                "parameter": parameter,
                "parameter_value": clean_float(parameter_value),
                "is_baseline": bool(parameter_value == baseline_value),
                "score_deviation": clean_float(float(np.mean(deviations))) if deviations else None,
                "reference_deviation": clean_float(float(np.mean(reference_deviations)))
                if reference_deviations
                else None,
                "sample_count": len(contributing_samples),
                "score_count": len(deviations),
                "boundary_sample_count": len(boundary_samples),
                "status": status,
            }
        )

    ranking = sorted(
        (entry for entry in entries if entry["status"] == "scored"),
        key=lambda item: (-float(item["score_deviation"]), float(item["parameter_value"])),
    )
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank
    boundary_rows.sort(
        key=lambda item: (
            str(item["scheme"]),
            str(item["parameter"]),
            float(item["parameter_value"]),
            str(item["sample_id"]),
            str(item["criteria"]),
        )
    )
    return (
        {
            "scheme": scheme,
            "parameter": parameter,
            "baseline_value": clean_float(baseline_value),
            "tie_break": "score_deviation descending, parameter_value ascending",
            "scan_entries": entries,
            "ranking": ranking,
            "boundary_count": len(boundary_rows),
        },
        boundary_rows,
    )


def run_sensitivity_scan(
    config: DiagnosticConfig,
    samples: list[Sample],
    attention_results: dict[tuple[str, str], AttentionResult],
) -> dict[str, Any]:
    rankings: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for scheme in config.schemes:
        scheme_baselines = {
            sample.sample_id: attention_results[(scheme, sample.sample_id)]
            for sample in samples
            if (scheme, sample.sample_id) in attention_results
        }
        scheme_samples = [sample for sample in samples if sample.sample_id in scheme_baselines]
        base_ranking, base_boundaries = _scan_parameter(
            config=config,
            scheme=scheme,
            parameter="base",
            values=config.sensitivity_base_values,
            baseline_value=config.base,
            samples=scheme_samples,
            baselines=scheme_baselines,
            train_context_length=config.train_context_length,
        )
        scale_ranking, scale_boundaries = _scan_parameter(
            config=config,
            scheme=scheme,
            parameter="scaling_factor",
            values=config.sensitivity_scaling_factor_values,
            baseline_value=config.scaling_factor,
            samples=scheme_samples,
            baselines=scheme_baselines,
            train_context_length=config.train_context_length,
        )
        rankings.extend([base_ranking, scale_ranking])
        boundary_rows.extend(base_boundaries)
        boundary_rows.extend(scale_boundaries)

    from .config import SCHEME_RANK

    rankings.sort(key=lambda item: (SCHEME_RANK[str(item["scheme"])], str(item["parameter"])))
    boundary_rows.sort(
        key=lambda item: (
            SCHEME_RANK[str(item["scheme"])],
            str(item["parameter"]),
            float(item["parameter_value"]),
            str(item["sample_id"]),
            str(item["criteria"]),
        )
    )
    return {
        "ranking_rule": "within each scheme/parameter: score_deviation descending, parameter_value ascending",
        "rankings": rankings,
        "boundary_sensitive_samples": boundary_rows,
    }
