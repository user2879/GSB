from __future__ import annotations

import shutil
from pathlib import Path

from pref_reward_lab.canonical import failure_sort_key
from pref_reward_lab.engine import ExperimentLab
from pref_reward_lab.sample import generate_dataset


def test_all_injections_hit_and_are_isolated(tmp_path: Path) -> None:
    data = tmp_path / "data"
    generate_dataset(data, pair_count=20, seed=17)
    lab = ExperimentLab(data, state_dir=tmp_path / "state", output_dir=tmp_path / "out")
    report = lab.run()
    injections = {item["id"]: item for item in report["injections"]}
    assert set(injections) == {"INJ-001", "INJ-002", "INJ-003", "INJ-004"}
    assert all(item["hit"] for item in injections.values())
    assert {item["business_code"] for item in injections.values()} == {
        "E_HASH_BUCKETS_ZERO",
        "E_NONFINITE_WEIGHT",
        "W_BOUNDARY_EQUAL_THRESHOLD",
        "E_SIDECAR_MISSING",
    }
    record_ids = {record["pair_id"] for record in report["records"]}
    assert "pair-000001" not in record_ids
    assert "pair-000002" not in record_ids
    assert "pair-000003" in record_ids
    assert "pair-000004" in record_ids
    assert len(record_ids) == 18
    injection_failures = [
        item for item in report["failures"] if item.get("injection_id")
    ]
    assert {item["code"] for item in injection_failures} == {
        "E_HASH_BUCKETS_ZERO",
        "E_NONFINITE_WEIGHT",
        "W_BOUNDARY_EQUAL_THRESHOLD",
        "E_SIDECAR_MISSING",
    }
    assert report["failures"] == sorted(report["failures"], key=failure_sort_key)
    assert "E_FORMAT_INVALID" in report["failure_business_codes"]
    boundary = next(
        item for item in report["boundary_samples"] if item["pair_id"] == "pair-000003"
    )
    assert boundary["reward_delta"] == boundary["effective_threshold"]


def test_concurrency_one_and_eight_produce_same_core(tmp_path: Path) -> None:
    data = tmp_path / "data"
    generate_dataset(data, pair_count=24, seed=23)
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    shutil.copytree(data, serial_dir)
    shutil.copytree(data, parallel_dir)
    serial = ExperimentLab(
        serial_dir, state_dir=tmp_path / "sstate", output_dir=tmp_path / "sout"
    )
    parallel = ExperimentLab(
        parallel_dir, state_dir=tmp_path / "pstate", output_dir=tmp_path / "pout"
    )
    serial.update_config({"concurrency": 1})
    parallel.update_config({"concurrency": 8})
    serial_report = serial.run()
    parallel_report = parallel.run()
    assert (
        serial_report["deterministic_core_fingerprint"]
        == parallel_report["deterministic_core_fingerprint"]
    )
    assert serial_report["records"] == parallel_report["records"]
    assert serial_report["conversion"] == parallel_report["conversion"]


def test_repeated_run_is_field_deterministic(tmp_path: Path) -> None:
    data = tmp_path / "data"
    generate_dataset(data, pair_count=18, seed=31, include_invalid_rows=False)
    (data / "injections.json").unlink()
    lab = ExperimentLab(data, state_dir=tmp_path / "state", output_dir=tmp_path / "out")
    first = lab.run()
    second = lab.run()
    assert first["deterministic_core_fingerprint"] == second["deterministic_core_fingerprint"]
    assert first["records"] == second["records"]
    assert first["feature_bucket_summary"] == second["feature_bucket_summary"]
    assert second["invalidation"]["reused_counts"]["features"] == 18
    assert second["invalidation"]["reused_counts"]["scoring"] == 18
    assert second["invalidation"]["reused_counts"]["conversion"] == 36
    assert second["invalidation"]["metrics_recomputed"] is False
    assert second["invalidation"]["strategy_reused"] is True
