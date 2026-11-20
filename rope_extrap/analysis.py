from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

from .engine import run_full_diagnosis
from .errors import Failure
from .models import (
    SCHEME_LINEAR,
    SCHEME_NTK,
    SCHEME_STANDARD,
    SUPPORTED_SCHEMES,
    AttentionSample,
    PositionEncodingConfig,
)


def build_scheme_statistics(
    config: PositionEncodingConfig,
    samples: Mapping[str, AttentionSample] | Iterable[AttentionSample],
    *,
    loader_failures: Iterable[Failure] = (),
) -> dict[str, Any]:
    sample_map = samples if isinstance(samples, Mapping) else {sample.sample_id: sample for sample in samples}
    statistics: dict[str, Any] = {}
    for scheme in SUPPORTED_SCHEMES:
        scheme_config = config.changed(scheme=scheme)
        report = run_full_diagnosis(
            scheme_config,
            sample_map,
            loader_failures=loader_failures,
        )
        error_statistics = report["error_statistics"]
        first_sample = report["samples"][0] if report["samples"] else None
        statistics[scheme] = {
            "scheme": scheme,
            "sample_count": len(report["samples"]),
            "external_key_count": error_statistics["extrapolation_key_count"],
            "over_tolerance_key_count": error_statistics["over_tolerance_key_count"],
            "over_tolerance_sample_count": report["summary"]["extrapolation_over_tolerance_samples"],
            "mean_position_vector_deviation": error_statistics["mean_position_vector_deviation"],
            "max_position_vector_deviation": error_statistics["max_position_vector_deviation"],
            "mean_score_deviation": error_statistics["mean_score_deviation"],
            "max_score_deviation": error_statistics["max_score_deviation"],
            "mean_attention_weight_deviation": error_statistics["mean_attention_weight_deviation"],
            "max_attention_weight_deviation": error_statistics["max_attention_weight_deviation"],
            "mean_max_deviation": error_statistics["mean_max_deviation"],
            "max_max_deviation": error_statistics["max_max_deviation"],
            "first_exceeded_stage_counts": error_statistics["first_exceeded_stage_counts"],
            "first_sample_position_fingerprint": (
                first_sample["stage_fingerprints"]["position_fingerprint"] if first_sample is not None else None
            ),
            "first_sample_score_fingerprint": (
                first_sample["stage_fingerprints"]["score_fingerprint"] if first_sample is not None else None
            ),
            "mean_external_score": report["score_statistics"]["extrapolation"]["mean_score"],
            "mean_external_attention_weight": report["score_statistics"]["extrapolation"][
                "mean_attention_weight"
            ],
        }
    return statistics


def build_parameter_sensitivity(
    config: PositionEncodingConfig,
    samples: Mapping[str, AttentionSample] | Iterable[AttentionSample],
    scheme_statistics: Mapping[str, Mapping[str, Any]],
    *,
    loader_failures: Iterable[Failure] = (),
) -> list[dict[str, Any]]:
    sample_map = samples if isinstance(samples, Mapping) else {sample.sample_id: sample for sample in samples}
    baseline = scheme_statistics[config.scheme]
    baseline_metric = baseline.get("mean_max_deviation")
    if baseline_metric is None:
        return []
    rows: list[dict[str, Any]] = []

    def add_row(
        parameter: str,
        kind: str,
        variant_value: Any,
        metric_value: float | None,
        score_metric: float | None,
        attention_metric: float | None,
    ) -> None:
        if metric_value is None:
            return
        delta = float(metric_value) - float(baseline_metric)
        rows.append(
            {
                "rank": 0,
                "parameter": parameter,
                "kind": kind,
                "variant_value": variant_value,
                "mean_max_deviation": float(metric_value),
                "delta_mean_max_deviation": delta,
                "abs_delta": abs(delta),
                "delta_mean_score_deviation": (
                    float(score_metric) - float(baseline["mean_score_deviation"])
                    if score_metric is not None and baseline.get("mean_score_deviation") is not None
                    else None
                ),
                "delta_mean_attention_weight_deviation": (
                    float(attention_metric) - float(baseline["mean_attention_weight_deviation"])
                    if attention_metric is not None
                    and baseline.get("mean_attention_weight_deviation") is not None
                    else None
                ),
            }
        )

    for scheme in SUPPORTED_SCHEMES:
        if scheme == config.scheme:
            continue
        variant = scheme_statistics[scheme]
        add_row(
            "scheme",
            "scheme",
            scheme,
            variant.get("mean_max_deviation"),
            variant.get("mean_score_deviation"),
            variant.get("mean_attention_weight_deviation"),
        )

    variant_configs: list[tuple[str, str, Any, PositionEncodingConfig]] = [
        ("base", "numeric", config.base * 0.5, config.changed(base=config.base * 0.5)),
        ("base", "numeric", config.base * 2.0, config.changed(base=config.base * 2.0)),
        (
            "scale_factor",
            "numeric",
            config.scale_factor * 0.5,
            config.changed(scale_factor=config.scale_factor * 0.5),
        ),
        (
            "scale_factor",
            "numeric",
            config.scale_factor * 2.0,
            config.changed(scale_factor=config.scale_factor * 2.0),
        ),
        (
            "training_context_length",
            "integer",
            max(1, config.training_context_length // 2),
            config.changed(
                training_context_length=max(1, config.training_context_length // 2)
            ),
        ),
        (
            "training_context_length",
            "integer",
            min(config.extrapolation_target_length - 1, config.training_context_length * 2),
            config.changed(
                training_context_length=min(
                    config.extrapolation_target_length - 1,
                    config.training_context_length * 2,
                )
            ),
        ),
        ("pruning_policy", "categorical", "none", config.changed(pruning_policy="none")),
        (
            "pruning_tolerance",
            "numeric",
            0.0,
            config.changed(pruning_tolerance=0.0),
        ),
        (
            "pruning_tolerance",
            "numeric",
            1.0,
            config.changed(pruning_tolerance=1.0),
        ),
    ]
    seen: set[tuple[str, Any]] = set()
    for parameter, kind, variant_value, variant_config in variant_configs:
        marker = (parameter, variant_value)
        if marker in seen:
            continue
        seen.add(marker)
        report = run_full_diagnosis(variant_config, sample_map, loader_failures=loader_failures)
        add_row(
            parameter,
            kind,
            variant_value,
            report["error_statistics"]["mean_max_deviation"],
            report["error_statistics"]["mean_score_deviation"],
            report["error_statistics"]["mean_attention_weight_deviation"],
        )
    rows.sort(key=lambda row: (-float(row["abs_delta"]), row["parameter"], str(row["variant_value"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_extended_report(
    report: dict[str, Any],
    config: PositionEncodingConfig,
    samples: Mapping[str, AttentionSample] | Iterable[AttentionSample],
    *,
    loader_failures: Iterable[Failure] = (),
    include_sensitivity: bool = True,
) -> dict[str, Any]:
    extended = copy.deepcopy(report)
    scheme_statistics = build_scheme_statistics(
        config,
        samples,
        loader_failures=loader_failures,
    )
    extended["scheme_extrapolation_statistics"] = scheme_statistics
    if include_sensitivity:
        extended["parameter_sensitivity_ranking"] = build_parameter_sensitivity(
            config,
            samples,
            scheme_statistics,
            loader_failures=loader_failures,
        )
    else:
        extended["parameter_sensitivity_ranking"] = []
    return extended


def diagnostic_consistency_view(report: dict[str, Any]) -> dict[str, Any]:
    """Strip run-history-only fields for incremental/full equivalence tests.

    The execution audit necessarily differs between a warm incremental run and
    a blank full rebuild (the latter cannot have cache hits).  All substantive
    diagnostic fields are retained here and compared field by field.
    """

    view = copy.deepcopy(report)
    view.get("cache", {}).pop("audit", None)
    view.get("cache", {}).pop("stats", None)
    view.get("run", {}).pop("engine_max_observed_concurrency", None)
    view.get("run", {}).pop("elapsed_ms", None)
    view.get("run", {}).pop("mode", None)
    view.get("run", {}).pop("command_index", None)
    view.get("run", {}).pop("command_id", None)
    view.get("run", {}).pop("budget_ms", None)
    view.get("run", {}).pop("complete_after_command", None)
    view.pop("command_audit_history", None)
    view.pop("loader_failures", None)
    view.pop("command_load_failures", None)
    view.pop("interactive_business_code_counts", None)
    return view
