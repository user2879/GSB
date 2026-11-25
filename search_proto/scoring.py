from __future__ import annotations

import math
from typing import Any

from .config import PrototypeConfig
from .fixed import (
    STAGE_ORDER,
    first_tolerance_stage,
    fixed_average,
    fixed_from_value,
    fixed_idf,
    fixed_mul,
    float_idf,
    saturation_fixed,
    saturation_float,
    stage_errors,
)
from .model import FIELD_ORDER, QuerySpec
from .util import format_fixed, sha256_json


def query_document_score_cache_key(
    query: QuerySpec,
    document_stats: dict[str, Any],
    term_stats: dict[str, dict[str, Any]],
    config: PrototypeConfig,
) -> str:
    referenced_terms: list[str] = []
    for term in query.sorted_unique_terms():
        term_stat = term_stats.get(term)
        if term_stat is None:
            continue
        if any(
            document["doc_id"] == document_stats["doc_id"]
            for document in term_stat["documents"]
        ):
            referenced_terms.append(term)
    document_content = (
        document_stats["content_fingerprint"] if referenced_terms else None
    )
    return sha256_json(
        {
            "query_fingerprint": query.content_fingerprint,
            "document": {
                "doc_id": document_stats["doc_id"],
                "version": document_stats["version"],
                "content_fingerprint": document_content,
            },
            "referenced_term_stats": {
                term: term_stats[term]["term_stats_fingerprint"]
                for term in referenced_terms
            },
            "field_config_fingerprint": config.field_config_fingerprint,
            "bm25_fingerprint": config.bm25_fingerprint,
            "fixed_fingerprint": config.fixed_fingerprint,
        }
    )


def _sum_in_fixed_order(values: list[float]) -> float:
    return math.fsum(values)


def score_document(
    query: QuerySpec,
    document_stats: dict[str, Any],
    term_stats: dict[str, dict[str, Any]],
    config: PrototypeConfig,
    document_count: int,
) -> dict[str, Any]:
    fixed = config.fixed
    scale = fixed.scale
    fields = [
        field_name
        for field_name in FIELD_ORDER
        if field_name in config.fields and field_name in query.field_preferences
    ]
    weights_fixed = {
        field_name: fixed_from_value(config.field_weights[field_name], fixed)
        for field_name in fields
    }
    k1_fixed = fixed_from_value(config.bm25.k1, fixed)
    b_fixed = fixed_from_value(config.bm25.b, fixed)

    fixed_total = 0
    float_contributions: list[float] = []
    variant_values: dict[str, list[float]] = {
        "float_reference": [],
        "idf_quantized_only": [],
        "length_quantized_only": [],
        "saturation_quantized_only": [],
        "field_weight_quantized_only": [],
    }
    term_traces: list[dict[str, Any]] = []
    max_stage_errors: dict[str, float] = {stage: 0.0 for stage in STAGE_ORDER}

    for term in query.sorted_unique_terms():
        term_stat = term_stats.get(term)
        if term_stat is None or term_stat["df"] <= 0:
            continue
        posting = None
        for document in term_stat["documents"]:
            if document["doc_id"] == document_stats["doc_id"]:
                posting = document
                break
        if posting is None:
            continue

        idf_fixed_units = fixed_idf(document_count, term_stat["df"], fixed)
        idf_float = float_idf(document_count, term_stat["df"])
        field_traces: list[dict[str, Any]] = []
        for field_name in fields:
            field_posting = posting["fields"].get(field_name)
            if field_posting is None:
                continue
            tf = int(field_posting["tf"])
            field_length = int(field_posting["field_length"])
            averages = term_stat["average_field_lengths"].get(field_name)
            if averages is None or averages["count"] <= 0:
                continue
            average_fixed = fixed_average(
                int(averages["total_length"]), int(averages["count"]), fixed
            )
            average_float = int(averages["total_length"]) / int(averages["count"])
            saturated_fixed = saturation_fixed(
                tf,
                field_length,
                average_fixed,
                k1_fixed,
                b_fixed,
                fixed,
            )
            saturated_float = saturation_float(
                tf,
                field_length,
                average_float,
                config.bm25.k1,
                config.bm25.b,
            )
            weight_fixed = weights_fixed[field_name]
            weight_float = config.field_weights[field_name]
            weighted_idf_fixed = fixed_mul(weight_fixed, idf_fixed_units, fixed)
            contribution_fixed = fixed_mul(
                weighted_idf_fixed, saturated_fixed["saturation"], fixed
            )
            contribution_float = (
                weight_float * idf_float * saturated_float["saturation"]
            )

            length_quantized_only_saturation = 0.0
            if average_float > 0 and tf > 0:
                length_norm_fixed_float = (
                    saturated_fixed["length_norm"] / scale
                )
                denominator = tf + config.bm25.k1 * (
                    1.0 - config.bm25.b + length_norm_fixed_float
                )
                if denominator != 0:
                    length_quantized_only_saturation = (
                        tf * (config.bm25.k1 + 1.0) / denominator
                    )

            field_trace = {
                "term": term,
                "field": field_name,
                "tf": tf,
                "field_length": field_length,
                "df": term_stat["df"],
                "idf_fixed_units": idf_fixed_units,
                "idf_float": idf_float,
                "average_field_length_fixed_units": average_fixed,
                "average_field_length_float": average_float,
                "length_ratio_fixed_units": saturated_fixed["length_ratio"],
                "length_ratio_float": saturated_float["length_ratio"],
                "length_norm_fixed_units": saturated_fixed["length_norm"],
                "length_norm_float": saturated_float["length_norm"],
                "denominator_base_fixed_units": saturated_fixed["denominator_base"],
                "denominator_base_float": saturated_float["denominator_base"],
                "denominator_fixed_units": saturated_fixed["denominator"],
                "denominator_float": saturated_float["denominator"],
                "saturation_fixed_units": saturated_fixed["saturation"],
                "saturation_float": saturated_float["saturation"],
                "field_weight_fixed_units": weight_fixed,
                "field_weight_float": weight_float,
                "contribution_fixed_units": contribution_fixed,
                "contribution_float": contribution_float,
            }
            errors = stage_errors(field_trace, fixed)
            field_trace["stage_abs_error_units"] = errors
            for stage, error in errors.items():
                max_stage_errors[stage] = max(max_stage_errors[stage], error)
            field_traces.append(field_trace)

            fixed_total += contribution_fixed
            float_contributions.append(contribution_float)
            variant_values["float_reference"].append(contribution_float)
            variant_values["idf_quantized_only"].append(
                weight_float
                * (idf_fixed_units / scale)
                * saturated_float["saturation"]
            )
            variant_values["length_quantized_only"].append(
                weight_float * idf_float * length_quantized_only_saturation
            )
            variant_values["saturation_quantized_only"].append(
                weight_float
                * idf_float
                * (saturated_fixed["saturation"] / scale)
            )
            variant_values["field_weight_quantized_only"].append(
                (weight_fixed / scale)
                * idf_float
                * saturated_float["saturation"]
            )

        if field_traces:
            term_traces.append(
                {
                    "term": term,
                    "df": term_stat["df"],
                    "fields": field_traces,
                }
            )

    float_score = _sum_in_fixed_order(float_contributions)
    accumulation_error = abs(fixed_total - float_score * scale)
    max_stage_errors["accumulation"] = max(
        max_stage_errors.get("accumulation", 0.0), accumulation_error
    )
    variants = {
        name: _sum_in_fixed_order(values)
        for name, values in variant_values.items()
    }
    variants["fixed"] = fixed_total / scale
    first_stage = first_tolerance_stage(
        max_stage_errors, config.stage_tolerance_units
    )
    return {
        "query_id": query.query_id,
        "doc_id": document_stats["doc_id"],
        "doc_version": document_stats["version"],
        "document_content_fingerprint": (
            document_stats["content_fingerprint"] if term_traces else None
        ),
        "fixed_score_units": fixed_total,
        "fixed_score": format_fixed(fixed_total, scale, fixed.decimal_places),
        "float_score": float_score,
        "fixed_scale": scale,
        "quantum_units": fixed.quantum,
        "terms": term_traces,
        "diagnostics": {
            "stage_max_abs_error_units": {
                stage: max_stage_errors[stage] for stage in STAGE_ORDER
            },
            "first_tolerance_exceeded_stage": first_stage,
            "tolerance_units": config.stage_tolerance_units,
            "variant_scores": variants,
            "fixed_minus_float_units": fixed_total - float_score * scale,
        },
    }
