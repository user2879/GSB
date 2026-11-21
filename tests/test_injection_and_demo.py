from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from remote_seg.demo import run_acceptance_demo
from remote_seg.engine import RunOptions, run_until_complete
from remote_seg.injections import materialize_injections

def _workspace(sample_workspace, destination: Path) -> Path:
    destination.mkdir(parents=True)
    data = sample_workspace["data"]
    shutil.copytree(data / "tiles", destination / "tiles")
    shutil.copytree(data / "pool", destination / "pool")
    for name in ("index.json", "fusion_versions.json", "patch_plan.json"):
        shutil.copyfile(data / name, destination / name)
    shutil.copyfile(sample_workspace["config"], destination / "segmentation.json")
    return destination


@pytest.mark.parametrize(
    ("profile", "expected_code"),
    [
        ("truncate_png", "E_TILE_DECODE"),
        ("band_count_mismatch", "E_TILE_SHAPE_MISMATCH"),
        ("patch_bounds", "E_PATCH_BOUNDS"),
        ("unknown_band_version", "E_BAND_VERSION_UNKNOWN"),
        ("unknown_fusion_version", "E_FUSION_VERSION_UNKNOWN"),
    ],
)
def test_injection_profiles_are_isolated_and_reported(
    sample_workspace, tmp_path: Path, profile: str, expected_code: str
) -> None:
    source = _workspace(sample_workspace, tmp_path / "source")
    target = tmp_path / "injected"
    materialize_injections(
        source,
        target,
        Path(__file__).parents[1] / "configs" / "injections.json",
        profile=profile,
    )
    options = RunOptions(
        index_path=target / "index.json",
        config_path=target / "segmentation.json",
        fusion_versions_path=target / "fusion_versions.json",
        state_dir=tmp_path / "state",
        report_path=tmp_path / "report.json",
        max_batches=100,
    )
    report = run_until_complete(options)
    assert any(failure["code"] == expected_code for failure in report["failures"])
    assert any(hit["hit"] for hit in report["injection_hits"])
    if expected_code != "E_FUSION_VERSION_UNKNOWN":
        assert report["status"] == "complete"


def test_edit_sequence_incremental_equals_full_rebuild(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    result = run_acceptance_demo(
        tmp_path / "demo",
        sample_dir=root / "sample_data",
        config_path=root / "configs" / "segmentation.json",
        sequence_path=root / "configs" / "edit_sequence.json",
    )
    assert result["status"] == "pass", json.dumps(result["comparison"], indent=2)
    assert result["comparison"]["equal"] is True
    assert result["comparison"]["categories"]["global_classes"] is True
    assert result["comparison"]["categories"]["fusion_region_votes"] is True
    assert result["comparison"]["categories"]["reuse_recompute_ledger"] is True
    assert all(entry["comparison"]["equal"] for entry in result["step_equivalence"])
