from __future__ import annotations

from collections import Counter
from pathlib import Path

from instruction_factory.constants import E_SNAPSHOT_STALE
from instruction_factory.loader import load_inputs
from instruction_factory.runner import run_incremental, run_verification

from .conftest import DATA_DIR, read_json, write_json


def _actions(report: dict) -> Counter:
    return Counter((item["stage"], item["action"]) for item in report["invalidation"])


def test_verification_proves_incremental_full_and_repeat_equivalence(tmp_path: Path) -> None:
    inputs = load_inputs(DATA_DIR)
    result, code = run_verification(inputs=inputs, output_dir=tmp_path / "verification")
    assert code == 0
    assert result["ok"]
    assert result["incremental_vs_full"]["equal"]
    assert result["full_repeat_determinism"]["equal"]
    batch_seven = read_json(tmp_path / "verification" / "incremental" / "reports" / "batch_0007.json")
    batch_eight = read_json(tmp_path / "verification" / "incremental" / "reports" / "batch_0008.json")
    actions_seven = _actions(batch_seven)
    assert actions_seven[("sampling", "recompute")] == 3
    assert actions_seven[("sampling", "reuse")] == 12
    assert actions_seven[("scoring", "recompute")] == 3
    assert actions_seven[("scoring", "reuse")] == 12
    assert _actions(batch_eight) == Counter({("sampling", "reuse"): 15, ("scoring", "reuse"): 15, ("dedupe", "reuse"): 3})


def test_second_incremental_run_reuses_snapshot_without_stale_code(tmp_path: Path) -> None:
    inputs = load_inputs(DATA_DIR)
    output = tmp_path / "out"
    first, first_code = run_incremental(inputs=inputs, output_dir=output, reset=True)
    second, second_code = run_incremental(inputs=inputs, output_dir=output, reset=False)
    assert first_code == second_code == 0
    assert second["snapshot"]["global_version"] == first["snapshot"]["global_version"]
    assert second["snapshot"]["changed_phases"] == []
    assert not any(item["code"] == E_SNAPSHOT_STALE for item in second["failures"])
    assert _actions(second) == Counter({("sampling", "reuse"): 15, ("scoring", "reuse"): 15, ("dedupe", "reuse"): 3})


def test_snapshot_stale_rebuilds_and_reports_business_code(tmp_path: Path) -> None:
    sample_dir = tmp_path / "data"
    import shutil

    shutil.copytree(DATA_DIR, sample_dir)
    output = tmp_path / "out"
    first_inputs = load_inputs(sample_dir)
    first, first_code = run_incremental(inputs=first_inputs, output_dir=output, reset=True)
    assert first_code == 0

    scoring_path = sample_dir / "scoring.json"
    scoring = read_json(scoring_path)
    scoring["pass_threshold"] = 87.5
    write_json(scoring_path, scoring)
    second_inputs = load_inputs(sample_dir)
    second, second_code = run_incremental(inputs=second_inputs, output_dir=output, reset=False)

    assert second_code == 0
    stale = [item for item in second["failures"] if item["code"] == E_SNAPSHOT_STALE]
    assert stale and "scoring" in stale[0]["message"]
    assert second["snapshot"]["global_version"] != first["snapshot"]["global_version"]
    assert "scoring" in second["snapshot"]["changed_phases"]
    first_rebuilt_batch = read_json(tmp_path / "out" / "reports" / "batch_0001.json")
    rebuild_actions = _actions(first_rebuilt_batch)
    assert rebuild_actions[("sampling", "reuse")] == 15
    assert rebuild_actions[("scoring", "recompute")] == 15


def test_report_contains_required_audit_sections(tmp_path: Path) -> None:
    inputs = load_inputs(DATA_DIR)
    report, code = run_incremental(inputs=inputs, output_dir=tmp_path / "out", reset=True)
    assert code == 0
    for field in (
        "task_statistics",
        "candidates",
        "near_duplicate_groups",
        "removed_duplicates",
        "budget_ledger",
        "snapshot",
        "failure_business_codes",
        "upstream_artifacts",
    ):
        assert field in report
    assert "dimension_distribution" in report["task_statistics"]["summarize"]
    assert report["snapshot"]["global_version"]
    assert report["budget_ledger"]
