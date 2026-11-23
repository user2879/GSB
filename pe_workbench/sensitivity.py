"""Deterministic sensitivity matrix and flip attribution."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Any

from .config import WorkbenchConfig
from .encoding import encode_pair
from .error_tracking import build_reference, track_errors
from .hashing import sha256_json, stable_float
from .models import Sample
from .quantization import score_fixed_path

PRECISION_ORDER = {"fp16": 0, "fp32": 1, "fp64": 2}
QUANT_ORDER = {"fine": 0, "balanced": 1, "coarse": 2}


class SensitivityAnalyzer:
    def __init__(self, config: WorkbenchConfig, samples: list[Sample], options: dict[str, Any] | None = None):
        self.base_config = config
        self.samples = sorted(samples, key=lambda sample: sample.sample_id)
        self.options = options or {}
        self._pe_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._reference_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._score_cache: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
        self._error_cache: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}

    def _candidate_schemes(self) -> list[str]:
        values = self.options.get("scheme_candidates", self.base_config.scheme_candidates)
        result = list(dict.fromkeys([*values, self.base_config.scheme]))
        return result

    def _candidate_precisions(self) -> list[str]:
        values = self.options.get("precision_levels", self.base_config.precision_levels)
        result = list(dict.fromkeys([*values, self.base_config.precision_level]))
        return sorted(result, key=lambda value: (PRECISION_ORDER.get(value, 99), value))

    def _candidate_quantization(self) -> list[str]:
        values = self.options.get("quantization_levels", self.base_config.quantization_levels)
        result = list(dict.fromkeys([*values, self.base_config.quantization_level]))
        return sorted(result, key=lambda value: (QUANT_ORDER.get(value, 99), value))

    def _candidate_widths(self) -> list[int]:
        values = self.options.get("fixed_decimal_widths", self.base_config.fixed_decimal_widths)
        result = sorted(set([*values, self.base_config.fixed_decimal_bits]))
        return result

    def _data_scales(self) -> list[float]:
        values = self.options.get("data_scales", [0.25, 0.5, 1.0])
        return sorted(set(float(value) for value in values))

    def _variant_config(
        self,
        scheme: str,
        precision: str,
        quant: str,
        fixed_bits: int,
    ) -> WorkbenchConfig:
        return replace(
            self.base_config,
            scheme=scheme,
            precision_level=precision,
            quantization_level=quant,
            fixed_decimal_bits=fixed_bits,
        )

    def _pe(self, sample: Sample, config: WorkbenchConfig) -> dict[str, Any]:
        cache_key = (sample.sample_id, config.scheme, config.precision_level)
        cached = self._pe_cache.get(cache_key)
        if cached is None:
            cached = encode_pair(
                sample.sample_id,
                sample.query_position,
                sample.key_positions,
                sample.query_vector,
                sample.key_vectors,
                config,
            )
            self._pe_cache[cache_key] = cached
        return cached

    def _reference(self, sample: Sample, config: WorkbenchConfig) -> dict[str, Any]:
        cache_key = (sample.sample_id, config.scheme)
        cached = self._reference_cache.get(cache_key)
        if cached is None:
            cached = build_reference(sample, config)
            self._reference_cache[cache_key] = cached
        return cached

    def _score(
        self, sample: Sample, config: WorkbenchConfig, pe: dict[str, Any]
    ) -> dict[str, Any]:
        cache_key = (
            sample.sample_id,
            config.scheme,
            config.precision_level,
            config.quantization_level,
            config.fixed_decimal_bits,
        )
        cached = self._score_cache.get(cache_key)
        if cached is None:
            cached = score_fixed_path(
                pe["query"],
                pe["keys"],
                sample.query_position,
                sample.key_positions,
                dimension=config.dimension,
                dimension_block_size=config.dimension_block_size,
                quantization_level=config.quantization_level,
                fixed_decimal_bits=config.fixed_decimal_bits,
            )
            self._score_cache[cache_key] = cached
        return cached

    def _error(
        self,
        sample: Sample,
        config: WorkbenchConfig,
        pe: dict[str, Any],
        score: dict[str, Any],
    ) -> dict[str, Any]:
        cache_key = (
            sample.sample_id,
            config.scheme,
            config.precision_level,
            config.quantization_level,
            config.fixed_decimal_bits,
        )
        cached = self._error_cache.get(cache_key)
        if cached is None:
            cached = track_errors(
                sample,
                pe,
                score,
                config,
                cache_key=sha256_json(
                    {
                        "sensitivity": True,
                        "sample_id": sample.sample_id,
                        "config": {
                            "scheme": config.scheme,
                            "precision": config.precision_level,
                            "quantization": config.quantization_level,
                            "fixed_bits": config.fixed_decimal_bits,
                            "train_context_length": config.train_context_length,
                            "tolerances": config.tolerances.to_dict(),
                        },
                    }
                ),
                reference=self._reference(sample, config),
            )
            self._error_cache[cache_key] = cached
        return cached

    def _evaluate_variant(
        self,
        scheme: str,
        precision: str,
        quant: str,
        fixed_bits: int,
    ) -> dict[str, Any]:
        config = self._variant_config(scheme, precision, quant, fixed_bits)
        sample_metrics: dict[str, dict[str, Any]] = {}
        attention_sum = 0.0
        position_sum = 0.0
        dot_sum = 0.0
        key_count = 0
        max_attention = 0.0
        overtolerance = 0
        for sample in self.samples:
            pe = self._pe(sample, config)
            score = self._score(sample, config, pe)
            error = self._error(sample, config, pe, score)
            extrapolation_keys = [
                key for key in error["keys"] if key["region"] == "extrapolation"
            ]
            sample_key_count = len(extrapolation_keys)
            sample_attention = (
                max((key["attention_weight_deviation"] for key in extrapolation_keys), default=0.0)
                if sample_key_count
                else 0.0
            )
            sample_position = (
                max((key["position_deviation"] for key in extrapolation_keys), default=0.0)
                if sample_key_count
                else 0.0
            )
            sample_dot = (
                max((key["dot_score_deviation"] for key in extrapolation_keys), default=0.0)
                if sample_key_count
                else 0.0
            )
            sample_metrics[sample.sample_id] = {
                "decision": bool(error["attention_overtolerance"]),
                "extrapolation_key_count": sample_key_count,
                "max_attention_error": stable_float(sample_attention),
                "max_position_error": stable_float(sample_position),
                "max_dot_error": stable_float(sample_dot),
                "first_exceeded_stage": error["first_exceeded_stage"],
                "fixed_max_index": error["fixed"]["max_index"],
                "reference_max_index": error["reference"]["max_index"],
                "fixed_denominator": error["fixed"]["denominator"],
                "reference_denominator": error["reference"]["denominator"],
                "keys": [
                    {
                        "index": key["index"],
                        "attention_error": key["attention_weight_deviation"],
                        "position_error": key["position_deviation"],
                        "dot_error": key["dot_score_deviation"],
                        "fixed_score": key["fixed_score"],
                        "reference_score": key["reference_score"],
                        "fixed_weight": key["fixed_weight"],
                        "reference_weight": key["reference_weight"],
                    }
                    for key in extrapolation_keys
                ],
            }
            attention_sum += sum(
                key["attention_weight_deviation"] for key in extrapolation_keys
            )
            position_sum += sum(key["position_deviation"] for key in extrapolation_keys)
            dot_sum += sum(key["dot_score_deviation"] for key in extrapolation_keys)
            key_count += sample_key_count
            max_attention = max(max_attention, sample_attention)
            if error["attention_overtolerance"]:
                overtolerance += 1
        return {
            "scheme": scheme,
            "precision_level": precision,
            "quantization_level": quant,
            "fixed_decimal_bits": fixed_bits,
            "sample_metrics": sample_metrics,
            "summary": {
                "sample_count": len(self.samples),
                "extrapolation_key_count": key_count,
                "mean_attention_error": stable_float(attention_sum / key_count)
                if key_count
                else None,
                "max_attention_error": stable_float(max_attention),
                "mean_position_error": stable_float(position_sum / key_count)
                if key_count
                else None,
                "mean_dot_error": stable_float(dot_sum / key_count) if key_count else None,
                "overtolerance_samples": overtolerance,
            },
        }

    def _variant_lookup(
        self, variants: list[dict[str, Any]]
    ) -> dict[tuple[str, str, str, int], dict[str, Any]]:
        return {
            (
                variant["scheme"],
                variant["precision_level"],
                variant["quantization_level"],
                variant["fixed_decimal_bits"],
            ): variant
            for variant in variants
        }

    def _ranking(
        self,
        variants: list[dict[str, Any]],
        data_scale_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        lookup = self._variant_lookup(variants)
        active = self.base_config

        def spread(entries: list[dict[str, Any]]) -> dict[str, Any]:
            means = [
                entry["summary"]["mean_attention_error"]
                for entry in entries
                if entry["summary"]["mean_attention_error"] is not None
            ]
            if not means:
                return {"impact": 0.0, "min": None, "max": None, "values": []}
            return {
                "impact": stable_float(max(means) - min(means)),
                "min": stable_float(min(means)),
                "max": stable_float(max(means)),
                "values": [
                    {
                        "value": entry["scheme"]
                        if dimension == "scheme"
                        else entry["precision_level"]
                        if dimension == "precision_level"
                        else entry["quantization_level"]
                        if dimension == "quantization_level"
                        else entry["fixed_decimal_bits"],
                        "mean_attention_error": entry["summary"]["mean_attention_error"],
                    }
                    for entry in entries
                ],
            }

        rows: list[dict[str, Any]] = []
        schemes = self._candidate_schemes()
        precisions = self._candidate_precisions()
        quant_levels = self._candidate_quantization()
        widths = self._candidate_widths()
        scheme_entries = [
            lookup[(scheme, active.precision_level, active.quantization_level, active.fixed_decimal_bits)]
            for scheme in schemes
            if (scheme, active.precision_level, active.quantization_level, active.fixed_decimal_bits) in lookup
        ]
        dimension = "scheme"
        rows.append(
            {
                "parameter": dimension,
                "metric": "mean_attention_error",
                **spread(scheme_entries),
            }
        )
        precision_entries = [
            lookup[(active.scheme, precision, active.quantization_level, active.fixed_decimal_bits)]
            for precision in precisions
            if (active.scheme, precision, active.quantization_level, active.fixed_decimal_bits) in lookup
        ]
        dimension = "precision_level"
        rows.append(
            {
                "parameter": dimension,
                "metric": "mean_attention_error",
                **spread(precision_entries),
            }
        )
        quantization_entries = [
            lookup[(active.scheme, active.precision_level, quant, active.fixed_decimal_bits)]
            for quant in quant_levels
            if (active.scheme, active.precision_level, quant, active.fixed_decimal_bits) in lookup
        ]
        dimension = "quantization_level"
        rows.append(
            {
                "parameter": dimension,
                "metric": "mean_attention_error",
                **spread(quantization_entries),
            }
        )
        width_entries = [
            lookup[(active.scheme, active.precision_level, active.quantization_level, bits)]
            for bits in widths
            if (active.scheme, active.precision_level, active.quantization_level, bits) in lookup
        ]
        dimension = "fixed_decimal_bits"
        rows.append(
            {
                "parameter": dimension,
                "metric": "mean_attention_error",
                **spread(width_entries),
            }
        )
        scale_means = [
            row["mean_attention_error"]
            for row in data_scale_rows
            if row["mean_attention_error"] is not None
        ]
        rows.append(
            {
                "parameter": "data_scale",
                "metric": "mean_attention_error",
                "impact": stable_float(max(scale_means) - min(scale_means))
                if scale_means
                else 0.0,
                "min": stable_float(min(scale_means)) if scale_means else None,
                "max": stable_float(max(scale_means)) if scale_means else None,
                "values": [
                    {
                        "value": row["data_scale"],
                        "mean_attention_error": row["mean_attention_error"],
                    }
                    for row in data_scale_rows
                ],
            }
        )
        rows.sort(key=lambda row: (-float(row["impact"]), row["parameter"]))
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        return rows

    def _data_scale_sensitivity(self, variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
        lookup = self._variant_lookup(variants)
        rows: list[dict[str, Any]] = []
        active = self.base_config
        sample_ids = [sample.sample_id for sample in self.samples]
        for quant in self._candidate_quantization():
            variant = lookup.get(
                (active.scheme, active.precision_level, quant, active.fixed_decimal_bits)
            )
            if variant is None:
                continue
            for scale in self._data_scales():
                count = max(1, math.ceil(len(sample_ids) * scale))
                prefix = sample_ids[:count]
                errors: list[float] = []
                overtolerance = 0
                for sample_id in prefix:
                    metrics = variant["sample_metrics"][sample_id]
                    errors.extend(key["attention_error"] for key in metrics["keys"])
                    if metrics["decision"]:
                        overtolerance += 1
                rows.append(
                    {
                        "data_scale": scale,
                        "sample_count": count,
                        "quantization_level": quant,
                        "extrapolation_key_count": len(errors),
                        "mean_attention_error": stable_float(math.fsum(errors) / len(errors))
                        if errors
                        else None,
                        "max_attention_error": stable_float(max(errors)) if errors else None,
                        "overtolerance_samples": overtolerance,
                    }
                )
        rows.sort(key=lambda row: (row["quantization_level"], row["data_scale"]))
        return rows

    def _quantization_order_stability(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_scale: dict[float, list[dict[str, Any]]] = {}
        for row in rows:
            by_scale.setdefault(row["data_scale"], []).append(row)
        orders: list[dict[str, Any]] = []
        signatures: list[tuple[str, ...]] = []
        for scale in sorted(by_scale):
            ordered = sorted(
                by_scale[scale],
                key=lambda row: (
                    row["mean_attention_error"]
                    if row["mean_attention_error"] is not None
                    else -1.0,
                    QUANT_ORDER.get(row["quantization_level"], 99),
                ),
            )
            signature = tuple(row["quantization_level"] for row in ordered)
            signatures.append(signature)
            orders.append({"data_scale": scale, "order": list(signature)})
        return {
            "stable": len(set(signatures)) <= 1,
            "orders": orders,
        }

    def _focus_key_pair(
        self,
        metrics_a: dict[str, Any],
        metrics_b: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        left = {key["index"]: key for key in metrics_a["keys"]}
        right = {key["index"]: key for key in metrics_b["keys"]}
        common = sorted(set(left) & set(right))
        if not common:
            return None, None
        index = max(
            common,
            key=lambda item: abs(
                float(left[item]["attention_error"]) - float(right[item]["attention_error"])
            ),
        )
        return left[index], right[index]

    def _attribution_case(
        self,
        *,
        mode: str,
        variant_a: dict[str, Any],
        variant_b: dict[str, Any],
        sample_id: str,
    ) -> dict[str, Any] | None:
        metrics_a = variant_a["sample_metrics"][sample_id]
        metrics_b = variant_b["sample_metrics"][sample_id]
        if metrics_a["decision"] == metrics_b["decision"]:
            return None
        key_a, key_b = self._focus_key_pair(metrics_a, metrics_b)
        if key_a is None or key_b is None:
            return None
        position_effect = abs(
            float(key_a["position_error"]) - float(key_b["position_error"])
        )
        dot_effect = abs(float(key_a["dot_error"]) - float(key_b["dot_error"]))
        weight_effect = abs(
            float(key_a["attention_error"]) - float(key_b["attention_error"])
        )
        max_index_changed = metrics_a["fixed_max_index"] != metrics_b["fixed_max_index"]
        denominator_delta = abs(
            float(metrics_a["fixed_denominator"] or 0.0)
            - float(metrics_b["fixed_denominator"] or 0.0)
        )
        tolerance = self.base_config.tolerances
        angle_ratio = position_effect / tolerance.position_vector
        rounding_ratio = dot_effect / tolerance.dot_score
        if max_index_changed:
            dominant = "normalization_denominator"
        elif rounding_ratio > angle_ratio:
            dominant = "quantization_rounding"
        else:
            dominant = "position_encoding_angle"
        return {
            "mode": mode,
            "sample_id": sample_id,
            "scheme": variant_a["scheme"],
            "precision_a": variant_a["precision_level"],
            "precision_b": variant_b["precision_level"],
            "quantization_a": variant_a["quantization_level"],
            "quantization_b": variant_b["quantization_level"],
            "fixed_decimal_bits_a": variant_a["fixed_decimal_bits"],
            "fixed_decimal_bits_b": variant_b["fixed_decimal_bits"],
            "decision_a": metrics_a["decision"],
            "decision_b": metrics_b["decision"],
            "focus_key_index": key_a["index"],
            "dominant_source": dominant,
            "evidence": {
                "position_vector_effect": stable_float(position_effect),
                "dot_score_effect": stable_float(dot_effect),
                "attention_weight_effect": stable_float(weight_effect),
                "angle_ratio": stable_float(angle_ratio),
                "rounding_ratio": stable_float(rounding_ratio),
                "fixed_max_index_changed": max_index_changed,
                "fixed_max_index_a": metrics_a["fixed_max_index"],
                "fixed_max_index_b": metrics_b["fixed_max_index"],
                "denominator_delta": stable_float(denominator_delta),
            },
        }

    def _flip_attribution(
        self, variants: list[dict[str, Any]]
    ) -> dict[str, Any]:
        lookup = self._variant_lookup(variants)
        schemes = self._candidate_schemes()
        precisions = self._candidate_precisions()
        quant_levels = self._candidate_quantization()
        widths = self._candidate_widths()
        cases: list[dict[str, Any]] = []
        for scheme in schemes:
            for quant in quant_levels:
                for bits in widths:
                    for precision_a, precision_b in itertools.combinations(precisions, 2):
                        variant_a = lookup.get((scheme, precision_a, quant, bits))
                        variant_b = lookup.get((scheme, precision_b, quant, bits))
                        if variant_a is None or variant_b is None:
                            continue
                        for sample in self.samples:
                            case = self._attribution_case(
                                mode="precision",
                                variant_a=variant_a,
                                variant_b=variant_b,
                                sample_id=sample.sample_id,
                            )
                            if case is not None:
                                cases.append(case)
                    for precision in precisions:
                        for quant_a, quant_b in itertools.combinations(quant_levels, 2):
                            variant_a = lookup.get((scheme, precision, quant_a, bits))
                            variant_b = lookup.get((scheme, precision, quant_b, bits))
                            if variant_a is None or variant_b is None:
                                continue
                            for sample in self.samples:
                                case = self._attribution_case(
                                    mode="quantization",
                                    variant_a=variant_a,
                                    variant_b=variant_b,
                                    sample_id=sample.sample_id,
                                )
                                if case is not None:
                                    cases.append(case)
        cases.sort(
            key=lambda case: (
                case["mode"],
                case["scheme"],
                case["quantization_a"],
                case["quantization_b"],
                case["fixed_decimal_bits_a"],
                case["fixed_decimal_bits_b"],
                case["precision_a"],
                case["precision_b"],
                case["sample_id"],
            )
        )
        counts = {
            "position_encoding_angle": 0,
            "quantization_rounding": 0,
            "normalization_denominator": 0,
        }
        by_mode: dict[str, int] = {}
        sample_summary: dict[str, dict[str, Any]] = {}
        for case in cases:
            counts[case["dominant_source"]] += 1
            by_mode[case["mode"]] = by_mode.get(case["mode"], 0) + 1
            summary = sample_summary.setdefault(
                case["sample_id"],
                {
                    "sample_id": case["sample_id"],
                    "total_flips": 0,
                    "by_source": {
                        "position_encoding_angle": 0,
                        "quantization_rounding": 0,
                        "normalization_denominator": 0,
                    },
                    "by_mode": {},
                },
            )
            summary["total_flips"] += 1
            summary["by_source"][case["dominant_source"]] += 1
            summary["by_mode"][case["mode"]] = summary["by_mode"].get(case["mode"], 0) + 1
        representative: list[dict[str, Any]] = []
        seen: dict[tuple[str, str, str], int] = {}
        for case in cases:
            group = (case["sample_id"], case["dominant_source"], case["mode"])
            count = seen.get(group, 0)
            if count < 2:
                representative.append(case)
                seen[group] = count + 1
        sample_rows = []
        for sample_id in sorted(sample_summary):
            row = sample_summary[sample_id]
            row["by_mode"] = dict(sorted(row["by_mode"].items()))
            sample_rows.append(row)
        return {
            "decision_metric": "attention_weight_error_exceeds_tolerance",
            "counts_by_source": counts,
            "counts_by_mode": dict(sorted(by_mode.items())),
            "total_flips": len(cases),
            "sample_attribution_summary": sample_rows,
            "cases_are_representative": True,
            "representative_limit_per_sample_source_mode": 2,
            "cases_truncated": len(cases) - len(representative),
            "cases": representative,
        }

    def analyze(self) -> dict[str, Any]:
        variants: list[dict[str, Any]] = []
        schemes = self._candidate_schemes()
        precisions = self._candidate_precisions()
        quant_levels = self._candidate_quantization()
        widths = self._candidate_widths()
        for scheme in schemes:
            for precision in precisions:
                for quant in quant_levels:
                    for bits in widths:
                        variants.append(
                            self._evaluate_variant(scheme, precision, quant, bits)
                        )
        data_scale_rows = self._data_scale_sensitivity(variants)
        ranking = self._ranking(variants, data_scale_rows)
        flips = self._flip_attribution(variants)
        variant_summaries = [
            {
                key: variant[key]
                for key in (
                    "scheme",
                    "precision_level",
                    "quantization_level",
                    "fixed_decimal_bits",
                    "summary",
                )
            }
            for variant in variants
        ]
        return {
            "analysis_version": "sensitivity-v1",
            "sample_count": len(self.samples),
            "candidate_schemes": schemes,
            "candidate_precisions": precisions,
            "candidate_quantization_levels": quant_levels,
            "candidate_fixed_decimal_widths": widths,
            "variant_count": len(variants),
            "ranking": ranking,
            "data_scale_sensitivity": data_scale_rows,
            "quantization_order_stability": self._quantization_order_stability(
                data_scale_rows
            ),
            "flip_attribution": flips,
            "variant_summaries": variant_summaries,
            "reproducibility": {
                "stable_order": "scheme,precision,quantization,fixed_width; sample_id",
                "aggregation": "math.fsum over ascending sample and key index",
            },
        }


def sensitivity_cache_key(
    config: WorkbenchConfig,
    sample_content_hash: str,
    options: dict[str, Any] | None,
) -> str:
    return sha256_json(
        {
            "schema": "sensitivity-cache-v1",
            "care_semantics": {
                "scheme_candidates": config.scheme_candidates,
                "base_frequency": config.base_frequency,
                "train_context_length": config.train_context_length,
                "extrapolation_target_length": config.extrapolation_target_length,
                "dimension": config.dimension,
                "scale_factor": config.scale_factor,
                "precision_levels": config.precision_levels,
                "fixed_decimal_widths": config.fixed_decimal_widths,
                "quantization_levels": config.quantization_levels,
                "tolerances": config.tolerances.to_dict(),
                "dimension_block_size": config.dimension_block_size,
                "active": {
                    "scheme": config.scheme,
                    "precision_level": config.precision_level,
                    "quantization_level": config.quantization_level,
                    "fixed_decimal_bits": config.fixed_decimal_bits,
                },
            },
            "sample_content_hash": sample_content_hash,
            "options": options or {},
        }
    )
