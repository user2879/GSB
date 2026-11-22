from __future__ import annotations

from pathlib import Path

from tests.helpers import base_config, base_sample, run_in_dir


def test_group_sensitivity_tracks_width_and_quantile_parameters(tmp_path: Path) -> None:
    config = base_config()
    config["tensors"][0]["quantile_lower_permille"] = 50
    config["tensors"][0]["quantile_upper_permille"] = 950
    report_a, config_path, sample_path = run_in_dir(
        tmp_path, config, base_sample(), name="a"
    )
    conclusions_a = {
        (entry["group_id"], entry["bit_width"]): entry["conclusion"]
        for entry in report_a["sensitivity"]["groups"]
    }
    assert set(width for _, width in conclusions_a) == {4, 8}
    assert conclusions_a[("left", 4)] != conclusions_a[("left", 8)]
    assert "q=[50,950]" in conclusions_a[("left", 4)]

    config_b = base_config()
    config_b["tensors"][0]["quantile_lower_permille"] = 20
    config_b["tensors"][0]["quantile_upper_permille"] = 980
    report_b, _, _ = run_in_dir(
        tmp_path, config_b, base_sample(), name="b"
    )
    conclusions_b = {
        (entry["group_id"], entry["bit_width"]): entry["conclusion"]
        for entry in report_b["sensitivity"]["groups"]
    }
    assert conclusions_b[("left", 4)] != conclusions_a[("left", 4)]
    assert "q=[20,980]" in conclusions_b[("left", 4)]
    assert report_a["sensitivity"]["groups"] != report_b["sensitivity"]["groups"]


def test_group_limit_violations_are_reported(tmp_path: Path) -> None:
    config = base_config(group_over_tolerance_limit=0.0)
    config["tensors"][0]["tolerance"] = 1e-9
    report, _, _ = run_in_dir(tmp_path, config, base_sample(), name="viol")
    violations = report["sensitivity"]["violations"]
    assert violations
    assert all(item["over_tolerance_ratio"] > 0.0 for item in violations)
    assert {item["bit_width"] for item in violations}.issubset({4, 8})
