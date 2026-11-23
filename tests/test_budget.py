from __future__ import annotations

from rerank_lab.engine import run_experiment
from tests.conftest import patch_config, write_commands


def test_global_budget_pending_queue(tmp_path):
    report = run_experiment("samples/budget", tmp_path / "out")
    assert report["summary"]["completed_requests"] == 1
    assert report["summary"]["pending_requests"] == 3
    assert [item["request_id"] for item in report["pending"]] == [
        "r-002",
        "r-003",
        "r-004",
    ]
    assert {item["code"] for item in report["pending"]} == {
        "E_GLOBAL_BUDGET_PENDING"
    }
    assert report["schedule"]["mac_units_charged"] == 88
    assert report["schedule"]["mac_units_remaining"] == 12


def test_per_request_quota_aborts_request(sample_copy, tmp_path):
    sample = sample_copy("default")
    patch_config(sample, per_request_quota=10)
    write_commands(sample, [])
    report = run_experiment(sample, tmp_path / "out")
    assert report["summary"]["completed_requests"] == 0
    assert {failure["code"] for failure in report["failures"]} == {
        "E_REQUEST_BUDGET_EXHAUSTED"
    }
    assert report["summary"]["eligible_requests"] == 0


def test_serial_and_parallel_produce_same_results(sample_copy, tmp_path):
    serial_sample = sample_copy("budget")
    parallel_sample = sample_copy("budget")
    patch_config(serial_sample, concurrency=1, global_mac_budget=5000)
    patch_config(parallel_sample, concurrency=8, global_mac_budget=5000)
    write_commands(serial_sample, [])
    write_commands(parallel_sample, [])
    serial = run_experiment(serial_sample, tmp_path / "serial", enable_snapshots=False)
    parallel = run_experiment(
        parallel_sample, tmp_path / "parallel", enable_snapshots=False
    )
    assert serial["requests"] == parallel["requests"]
    assert serial["schedule"]["mac_units_charged"] == parallel["schedule"][
        "mac_units_charged"
    ]
    assert serial["schedule"]["concurrency"] == 1
    assert parallel["schedule"]["concurrency"] == 8


def test_eviction_forces_recompute_but_not_different_numbers(sample_copy, tmp_path):
    sample = sample_copy("default")
    patch_config(sample, cache_residency_budget=1)
    report = run_experiment(sample, tmp_path / "out")
    assert report["cache"]["capacity_blocks"] == 1
    assert report["cache"]["evictions"]
    assert report["consistency"]["fresh_cold_match"] is True
