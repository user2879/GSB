"""Requirement 3: integer fixed-point quantization and error bounds."""

import pytest

from config_regression.errorbound import quantize_version, analyze_error_bounds
from config_regression.fixedpoint import (
    INT64_MAX,
    coefficient_overbound,
    quantize_value,
    saturate,
    within_bound,
)
from config_regression.differ import diff_manifests
from config_regression.model import load_manifest, load_registry


def test_truncation_toward_zero_and_saturation():
    scale = 1000
    assert quantize_value(0.1, scale) == 100
    assert quantize_value(-0.1, scale) == -100
    assert quantize_value(1.999999, scale) == 1999
    assert quantize_value(-1.999999, scale) == -1999
    assert saturate(INT64_MAX + 5) == (INT64_MAX, True)


def test_coefficient_overbound_is_integer_and_sound():
    # |0.3| at scale 10 -> q=3 -> overbound ceil(3/10) = 1 >= 0.3
    assert coefficient_overbound(3, 10) == 1
    assert coefficient_overbound(25, 10) == 3  # 2.5 -> 3


def test_bound_judgement_emits_all_components():
    ok, components = within_bound(delta_int=5, bound_a=1, bound_b=1,
                                  extra_residual=2)
    assert ok is False
    assert components == {"abs_delta": 5, "baseline_bound": 1,
                          "candidate_bound": 1, "cross_version_residual": 2,
                          "limit": 4}
    ok2, _ = within_bound(4, 1, 1, 2)
    assert ok2 is True
    ok3, _ = within_bound(3, 1, 1, 2)
    assert ok3 is True


def _registry():
    return load_registry({"nodes": [{"node_id": "node-a", "port": 0}]})


def _manifests():
    base = load_manifest({"version": 1, "items": [
        {"key": "p", "value": 1.0, "targets": ["node-a"], "derived_from": []},
        {"key": "d", "value": 2.0, "targets": ["node-a"], "derived_from": [
            {"key": "p", "coefficient": 2.0}]}]}, _registry())
    cand = load_manifest({"version": 2, "items": [
        {"key": "p", "value": 1.0000001, "targets": ["node-a"],
         "derived_from": []},
        {"key": "d", "value": 2.0, "targets": ["node-a"], "derived_from": [
            {"key": "p", "coefficient": 2.0}]}]}, _registry())
    return base, cand


def test_tiny_real_change_is_within_quantization_bound():
    base, cand = _manifests()
    diff = diff_manifests(base, cand)
    report = analyze_error_bounds(base, cand, diff, scale_exponent=6)
    item = report["item_reports"]["p"]
    assert item["quantized_baseline"] == item["quantized_candidate"]
    assert item["value_error"]["judgment_components"]["limit"] == 4
    assert item["value_error"]["within_bound"] is True
    scheme = report["precision_scheme"]
    assert scheme["truncation"] == "toward_zero"
    assert scheme["overflow"] == "saturate_signed_int64"


def test_cumulative_bound_grows_along_derivation_chain():
    base, _ = _manifests()
    q = quantize_version(base, scale_exponent=6)
    assert q.bounds["p"].total == 1
    # derived item bound strictly exceeds the root's own residual
    assert q.bounds["d"].total > q.bounds["p"].total
    assert q.bounds["d"].truncation == 1
