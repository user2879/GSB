"""Requirement 5: error bound derivation, per-metric attribution, dominant
stage with explicit threshold, versioned comparison after edits."""

import copy
from fractions import Fraction

from obs_pipeline.error_model import compare_versions, derive_error_model
from obs_pipeline.spec import validate_spec


def test_bound_composition_and_shares(sample_spec):
    reports = derive_error_model(sample_spec, {"m0": 100})
    report = reports["m0"]
    assert report.final_bound_ulp > 0
    stage_ids = [sid for sid, _ in report.shares]
    assert stage_ids == ["s_diff", "s_avg", "s_bucket", "s_norm"]
    total_share = sum((share for _, share in report.shares), Fraction(0))
    assert total_share <= 1  # introduced components are a subset of the bound
    assert report.dominance_threshold_ppm == sample_spec.dominance_threshold_ppm
    # std_diff introduces nothing itself
    assert dict(report.shares)["s_diff"] == 0


def test_components_declared(sample_spec):
    reports = derive_error_model(sample_spec, {"m0": 100})
    nodes = {n.stage_id: n for n in reports["m0"].stage_nodes}
    labels = {lbl for node in nodes.values() for lbl, _ in node.components}
    assert "window_boundary" in labels
    assert "bucket_boundary" in labels
    assert "truncation" in labels


def test_dominant_stage_threshold_explicit(sample_spec):
    reports = derive_error_model(sample_spec, {"m0": 100})
    report = reports["m0"]
    if report.dominant_stage is not None:
        share = dict(report.shares)[report.dominant_stage]
        assert share * 1000000 >= report.dominance_threshold_ppm


def test_scale_change_updates_bound_and_keeps_old_version(sample_spec_dict):
    old_spec = validate_spec(sample_spec_dict)
    evolved = copy.deepcopy(sample_spec_dict)
    evolved["spec_version"] = 2
    evolved["stages"][3]["scale"] = 10000  # threshold_norm scale up
    new_spec = validate_spec(evolved)
    old_reports = derive_error_model(old_spec, {"m0": 100})
    new_reports = derive_error_model(new_spec, {"m0": 100})
    assert old_reports["m0"].final_bound_ulp != new_reports["m0"].final_bound_ulp
    diff = compare_versions(old_reports, new_reports)
    assert diff[0]["old_bound"] != diff[0]["new_bound"]
    assert diff[0]["old_bound"] is not None  # old version retained


def test_stage_order_change_updates_bound(sample_spec_dict):
    base = validate_spec(sample_spec_dict)
    swapped = copy.deepcopy(sample_spec_dict)
    swapped["spec_version"] = 2
    # swap window_avg and quantile_bucket in the chain
    swapped["stages"][1], swapped["stages"][2] = swapped["stages"][2], swapped["stages"][1]
    swapped["stages"][1]["upstream"] = ["s_diff"]
    swapped["stages"][2]["upstream"] = ["s_bucket"]
    swapped["stages"][3]["upstream"] = ["s_avg"]
    other = validate_spec(swapped)
    assert derive_error_model(base, {"m0": 100})["m0"].final_bound_ulp != \
        derive_error_model(other, {"m0": 100})["m0"].final_bound_ulp
