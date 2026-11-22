from __future__ import annotations

import shutil

from image_edit_workbench.codec_v1 import parse_v1
from image_edit_workbench.diagnostics import cross_version_diagnostics, validate_injection_cases
from image_edit_workbench.exchange import MaterializedExchange
from image_edit_workbench.runner import ScenarioState, cold_render_exchange
from image_edit_workbench.util import sha256_bytes


def test_cross_version_attribution_and_sidecar(workspace, tmp_path):
    diagnostics = cross_version_diagnostics(
        regions=list(workspace["regions"]),
        images=workspace["images"],
        config=workspace["config"],
        work_root=tmp_path / "cross",
    )
    assert diagnostics["sidecar_restore_hits"] == 6
    assert diagnostics["sidecar_restore_misses"] == 0
    assert diagnostics["restored_records_field_equal"] == 6
    assert diagnostics["repair_mapping"]["all_restored"] is True
    assert diagnostics["downgrade_info"]["rotation_feather_loss_regions"] == 4
    assert diagnostics["style_strength_max_abs_delta"] == "0.0004"
    rows = {row["region_key"]: row for row in diagnostics["regions"]}
    assert rows["scene_alpha.png#002"]["boundary_sensitive"] is False
    assert rows["scene_alpha.png#001"]["boundary_sensitive"] is True
    assert "coordinate_integerization" in rows["scene_alpha.png#001"]["causes"]
    assert "strength_permille_truncation" in rows["scene_gamma.png#001"]["causes"]
    assert "dropped_rotation_or_feather" in rows["scene_beta.png#002"]["causes"]
    assert diagnostics["boundary_business_codes"] == ["E_BOUNDARY_SENSITIVE"]
    assert sum(diagnostics["coordinate_deviation_histogram_aggregate"].values()) == 24


def test_injection_configuration_all_hits(workspace):
    keys = {region.region_key for region in workspace["regions"]}
    results = validate_injection_cases(
        workspace["root"],
        images=workspace["images"],
        manifest_keys=keys,
    )
    assert len(results) == 4
    assert all(result["hit"] for result in results)
    assert [result["expected_code"] for result in results] == [
        "E_FIELD_TRUNCATED",
        "E_STRUCTURE_INVALID",
        "E_REGION_OUT_OF_RANGE",
        "E_REGION_OUT_OF_RANGE",
    ]


def test_malformed_v1_isolated_without_blocking_valid_regions(workspace, tmp_path):
    sample = workspace["sample_dir"]
    keys = {region.region_key for region in workspace["regions"]}
    data = (sample / "exchange_v1.txt").read_bytes() + b"this-line-is-too-short\n"
    parsed = parse_v1(data, images=workspace["images"], manifest_keys=keys)
    assert len(parsed.regions) == 6
    assert [failure.code for failure in parsed.failures] == ["E_FIELD_TRUNCATED"]
    exchange = MaterializedExchange(
        version=1,
        text=data.decode("utf-8"),
        parsed=parsed,
        exchange_fingerprint=sha256_bytes(data),
        sidecar_hash="",
        sidecar=None,
    )
    state = ScenarioState(regions=list(parsed.regions), config=workspace["config"], exchange_version=1)
    result, _, _ = cold_render_exchange(
        exchange=exchange,
        state=state,
        images=workspace["images"],
        config=workspace["config"],
        state_dir=tmp_path / "state",
        output_dir=tmp_path / "outputs",
        drain=True,
    )
    assert len(result.region_records) == 6
    assert any(failure.code == "E_FIELD_TRUNCATED" for failure in result.failures)
    assert len(result.image_fingerprints) == 3

