"""Per-item fixed-point quantization error bounds across two versions."""

from __future__ import annotations

from dataclasses import dataclass

from .differ import ADDED, COEFFICIENT_CHANGED, REMOVED, DiffResult
from .fixedpoint import (
    BoundComponents,
    compute_item_bounds,
    quantize_value,
    validate_scale,
    within_bound,
)
from .model import Manifest


@dataclass
class QuantizedVersion:
    scale_exponent: int
    scale: int
    values: dict[str, int]
    edges: dict[tuple[str, str], int]
    bounds: dict[str, BoundComponents]


def quantize_version(manifest: Manifest, scale_exponent: int) -> QuantizedVersion:
    scale = validate_scale(scale_exponent)
    values = {key: quantize_value(manifest.items[key].value, scale)
              for key in sorted(manifest.items)}
    edges: dict[tuple[str, str], int] = {}
    for key in sorted(manifest.items):
        for edge in manifest.items[key].derived_from:
            edges[(edge.parent, key)] = quantize_value(edge.coefficient, scale)
    raw_items = {key: {"derived_from": [
        edge.parent for edge in manifest.items[key].derived_from]}
        for key in manifest.items}
    bounds = compute_item_bounds(raw_items, list(manifest.order), values,
                                 edges, scale)
    return QuantizedVersion(scale_exponent, scale, values, edges, bounds)


def analyze_error_bounds(baseline: Manifest, candidate: Manifest,
                         diff: DiffResult,
                         scale_exponent: int) -> dict:
    q_base = quantize_version(baseline, scale_exponent)
    q_cand = quantize_version(candidate, scale_exponent)
    scheme = {
        "scale_exponent": scale_exponent,
        "scale": q_base.scale,
        "truncation": "toward_zero",
        "overflow": "saturate_signed_int64",
        "arithmetic": "integer_multiply_add",
        "residual_unit": "one_quantized_integer_unit",
    }
    reports: dict[str, dict] = {}
    for key in diff.changed_keys:
        change_types = diff.item_diffs[key].change_types
        report = _value_report(key, change_types, baseline, candidate,
                               q_base, q_cand)
        coeff_report = _coefficient_report(key, change_types, q_base, q_cand)
        reports[key] = {
            "key": key,
            "change_types": change_types,
            "quantized_baseline": q_base.values.get(key),
            "quantized_candidate": q_cand.values.get(key),
            "value_error": report,
            "coefficient_errors": coeff_report,
            "baseline_bound": q_base.bounds[key].as_dict()
            if key in q_base.bounds else None,
            "candidate_bound": q_cand.bounds[key].as_dict()
            if key in q_cand.bounds else None,
        }
    return {
        "precision_scheme": scheme,
        "item_reports": {key: reports[key] for key in sorted(reports)},
    }


def _value_report(key: str, change_types: list[str], baseline: Manifest,
                  candidate: Manifest, q_base: QuantizedVersion,
                  q_cand: QuantizedVersion) -> dict:
    if ADDED in change_types:
        bound = q_cand.bounds[key]
        return {
            "kind": "added",
            "single_value_quantization_error": 1,
            "cumulative_error_bound": bound.as_dict(),
            "within_bound": None,
            "judgment_components": None,
        }
    if REMOVED in change_types:
        bound = q_base.bounds[key]
        return {
            "kind": "removed",
            "single_value_quantization_error": 1,
            "cumulative_error_bound": bound.as_dict(),
            "within_bound": None,
            "judgment_components": None,
        }
    delta = q_cand.values[key] - q_base.values[key]
        # Directly declared values are re-quantized independently, so the
        # cross-version residual is 2; derived integers are recomputed from
        # quantized inputs and need no extra residual beyond both bounds.
    extra_residual = 2 if not baseline.items[key].derived_from else 0
    ok, components = within_bound(
        delta,
        q_base.bounds[key].total,
        q_cand.bounds[key].total,
        extra_residual,
    )
    return {
        "kind": "shared",
        "single_value_quantization_error": 1,
        "cumulative_error_bound": {
            "baseline": q_base.bounds[key].as_dict(),
            "candidate": q_cand.bounds[key].as_dict(),
        },
        "delta_quantized": delta,
        "cross_version_residual": extra_residual,
        "within_bound": ok,
        "judgment_components": components,
    }


def _coefficient_report(key: str, change_types: list[str],
                        q_base: QuantizedVersion,
                        q_cand: QuantizedVersion) -> list[dict]:
    if COEFFICIENT_CHANGED not in change_types:
        return []
    out: list[dict] = []
    edges = sorted({p for p, child in set(q_base.edges) | set(q_cand.edges)
                    if child == key})
    for parent in edges:
        b_coeff = q_base.edges.get((parent, key))
        c_coeff = q_cand.edges.get((parent, key))
        if b_coeff == c_coeff:
            continue
        if b_coeff is None or c_coeff is None:
            out.append({"parent": parent, "baseline_quantized": b_coeff,
                        "candidate_quantized": c_coeff,
                        "within_bound": None,
                        "judgment_components": None})
            continue
        delta = c_coeff - b_coeff
        ok, components = within_bound(delta, 1, 1, 2)
        out.append({
            "parent": parent,
            "baseline_quantized": b_coeff,
            "candidate_quantized": c_coeff,
            "delta_quantized": delta,
            "within_bound": ok,
            "judgment_components": components,
        })
    return out
