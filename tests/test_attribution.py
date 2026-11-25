from __future__ import annotations

from candidate_diagnostics.attribution import attribute_request
from candidate_diagnostics.config import Config, deep_merge


def test_boundary_comparison_cause(config, requests):
    report = attribute_request(requests["req-bcmp"], config)
    assert report["first_exceeded_stage"] == "truncation_boundary"
    causes = {item["candidate_id"]: item["cause"] for item in report["flip_attribution"]}
    assert causes[705] == "boundary_comparison"
    assert causes[706] == "boundary_comparison"


def test_normalization_quantization_cause(config, requests):
    coarse = Config(deep_merge(config.raw, {"quantization": {"fixed_decimals": 0, "gear_exponent": 0}}))
    report = attribute_request(requests["req-qflip"], coarse, config)
    assert report["first_exceeded_stage"] == "normalization"
    assert all(
        item["cause"] == "normalization_quantization"
        for item in report["flip_attribution"]
        if item["stage"] == "truncation"
    )


def test_sampling_quota_rounding_cause(config, requests):
    more_sampling = Config(
        deep_merge(config.raw, {"sampling": {"ratio": 0.8, "ratio_permille": 800}})
    )
    report = attribute_request(requests["req-001"], config, more_sampling)
    causes = [item["cause"] for item in report["flip_attribution"]]
    assert "sampling_quota_rounding" in causes
    quota_diffs = report["stages"]["stratified_sampling"]["quota_diffs"]
    assert any(item["abs_delta"] > 0 for item in quota_diffs)
