from __future__ import annotations

import copy
import threading
import time

import numpy as np

import cluster_workbench.pipeline as pipeline_module
from cluster_workbench.cache import CacheStore
from cluster_workbench.config import normalize_config
from cluster_workbench.data import normalize_data_description
from cluster_workbench.errors import (
    E_CAND_BUDGET_EXHAUSTED,
    E_GLOBAL_BUDGET_PENDING,
    E_RESULT_STALE,
)
from cluster_workbench.pipeline import execute_pipeline
from cluster_workbench.scheduler import plan_budget
from cluster_workbench.snapshot import SnapshotStore
from cluster_workbench.workbench import Workbench

from conftest import candidate, normalized_pair


def make_state(config, data, candidates, *, overrides=None, injections=None, command_seq=0):
    config = copy.deepcopy(config)
    config["candidates"] = copy.deepcopy(candidates)
    return {
        "config": config,
        "data_description": copy.deepcopy(data),
        "candidates": copy.deepcopy(candidates),
        "data_overrides": copy.deepcopy(overrides or {}),
        "injections": copy.deepcopy(injections or []),
        "command_seq": command_seq,
    }


def report_actions(report, stage: str, candidate_id: str):
    return [
        entry["action"]
        for entry in report["reuse_ledger"]
        if entry["stage"] == stage and entry["candidate"] == candidate_id
    ]


def test_scheduler_reservation_and_global_pending_are_stable():
    candidates = [
        {"stable_key": "b", "method": "kmeans", "params": {}, "runtime_settings": {"max_iter": 20}, "valid": True},
        {"stable_key": "a", "method": "kmeans", "params": {}, "runtime_settings": {"max_iter": 20}, "valid": True},
    ]
    plan = plan_budget(candidates, global_budget=3, per_candidate_quota=2)
    assert [entry["candidate"] for entry in plan["entries"]] == ["a", "b"]
    assert plan["entries"][0]["reserved"] == 2
    assert plan["entries"][1]["outcome"] == "global_budget_pending"
    assert plan["entries"][1]["pending_reason"] == E_GLOBAL_BUDGET_PENDING


def test_quota_exhaustion_and_global_pending_in_pipeline(tmp_path, base_config, base_data):
    config, data = normalized_pair(base_config, base_data)
    config["scheduler"] = dict(
        config["scheduler"],
        global_iteration_budget=1,
        per_candidate_iteration_quota=1,
    )
    config["methods"]["kmeans"] = {"max_iter": 8, "tolerance": 0.0}
    candidates = [
        candidate("a", params={"n_clusters": 3}),
        candidate("b", params={"n_clusters": 3}, features=["n1", "n2"]),
    ]
    state = make_state(config, data, candidates)
    report = execute_pipeline(state, CacheStore(tmp_path / "cache"))
    by_id = {item["id"]: item for item in report["candidates"]}
    assert any(error["code"] == E_CAND_BUDGET_EXHAUSTED for error in by_id["a"]["errors"])
    assert any(error["code"] == E_GLOBAL_BUDGET_PENDING for error in by_id["b"]["errors"])
    assert by_id["b"]["status"] == "pending"
    assert [entry["candidate"] for entry in report["budget"]["entries"]] == ["a", "b"]


def test_concurrency_never_exceeds_configured_limit(monkeypatch, tmp_path, base_config, base_data):
    config, data = normalized_pair(base_config, base_data)
    config["scheduler"] = dict(config["scheduler"], concurrency=2)
    candidates = [
        candidate("a", params={"n_clusters": 2}, features=["n1", "n2"]),
        candidate("b", params={"n_clusters": 2}, features=["n1", "n2"]),
        candidate("c", params={"n_clusters": 2}, features=["n1", "n2"]),
        candidate("d", params={"n_clusters": 2}, features=["n1", "n2"]),
    ]
    state = make_state(config, data, candidates)
    original = pipeline_module.kmeans
    lock = threading.Lock()
    active = 0
    maximum = 0

    def wrapped(*args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.04)
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(pipeline_module, "kmeans", wrapped)
    report = execute_pipeline(state, CacheStore(tmp_path / "cache"))
    assert report["counts"]["completed"] == 4
    assert maximum <= 2
    assert maximum == 2


def test_single_column_change_invalidates_only_candidates_using_it(tmp_path, base_config, base_data):
    config, data = normalized_pair(base_config, base_data)
    candidates = [
        candidate("uses-n1", params={"n_clusters": 2}, features=["n1"]),
        candidate("uses-redundant", params={"n_clusters": 2}, features=["redundant_n1"]),
        candidate("uses-n3", params={"n_clusters": 2}, features=["n3"]),
    ]
    cache = CacheStore(tmp_path / "cache")
    first_state = make_state(config, data, candidates)
    execute_pipeline(first_state, cache)

    second_state = make_state(
        config,
        data,
        candidates,
        overrides={"n1": {"kind": "offset", "value": 4.0}},
    )
    report = execute_pipeline(second_state, cache)
    assert report_actions(report, "preprocess", "uses-n1") == ["recompute"]
    assert report_actions(report, "cluster", "uses-n1") == ["recompute"]
    assert report_actions(report, "evaluation", "uses-n1") == ["recompute"]
    assert report_actions(report, "preprocess", "uses-redundant") == ["recompute"]
    assert report_actions(report, "cluster", "uses-redundant") == ["recompute"]
    assert report_actions(report, "preprocess", "uses-n3") == ["reuse"]
    assert report_actions(report, "cluster", "uses-n3") == ["reuse"]
    assert report_actions(report, "evaluation", "uses-n3") == ["reuse"]


def test_hyperparameter_change_reuses_preprocess_but_recomputes_downstream(tmp_path, base_config, base_data):
    config, data = normalized_pair(base_config, base_data)
    candidates = [candidate("a", params={"n_clusters": 2}, features=["n1", "n2"])]
    cache = CacheStore(tmp_path / "cache")
    execute_pipeline(make_state(config, data, candidates), cache)
    changed = [candidate("a", params={"n_clusters": 3}, features=["n1", "n2"])]
    report = execute_pipeline(make_state(config, data, changed), cache)
    assert report_actions(report, "preprocess", "a") == ["reuse"]
    assert report_actions(report, "cluster", "a") == ["recompute"]
    assert report_actions(report, "evaluation", "a") == ["recompute"]


def test_interaction_returns_stale_then_fresh_summary(tmp_path, base_config, base_data):
    config, data = normalized_pair(base_config, base_data)
    config["scheduler"] = dict(
        config["scheduler"],
        interaction_latency_budget_ms=50,
        concurrency=1,
    )
    workbench = Workbench(
        config,
        data,
        workspace=tmp_path / "workspace",
        injections=[
            {
                "id": "slow",
                "type": "slow_candidate",
                "candidate": "slow-candidate",
                "delay_ms": 150,
            }
        ],
    )
    response = workbench.submit_command(
        {
            "type": "add_candidate",
            "id": "slow-candidate",
            "method": "kmeans",
            "params": {"n_clusters": 2},
            "feature_columns": ["n1", "n2"],
            "train_fraction": 0.6,
        }
    )
    assert response["stale"] is True
    assert response["code"] == E_RESULT_STALE
    assert response["within_budget"] is True
    assert response["latency_ms"] <= response["interaction_latency_budget_ms"]
    assert any(issue["code"] == E_RESULT_STALE for issue in response["issues"])
    assert response["affected_candidates"] == ["slow-candidate"]
    workbench.drain()
    fresh = workbench.submit_command({"type": "view", "part": "candidates", "offset": 0, "limit": 5})
    assert fresh["stale"] is False
    assert fresh["result_command_seq"] == 1
    assert fresh["summary"]["completed_candidates"] == 1
    workbench.close()


def test_incremental_equals_full_rebuild_and_snapshot_recovers(tmp_path, base_config, base_data):
    from cluster_workbench.cli import canonical_report_fields

    config, data = normalized_pair(base_config, base_data)
    config["candidates"] = [candidate("a", params={"n_clusters": 2}, features=["n1", "n2"])]
    workbench = Workbench(config, data, workspace=tmp_path / "workspace")
    commands = [
        {
            "type": "add_candidate",
            "id": "b",
            "method": "gmm",
            "params": {"n_components": 2, "covariance_type": "diag"},
            "feature_columns": ["n1", "n2", "n3"],
            "train_fraction": 0.6,
        },
        {
            "type": "update_candidate",
            "id": "a",
            "params": {"n_clusters": 3},
        },
        {
            "type": "replace_column",
            "column": "n3",
            "override": {"kind": "constant", "value": 0.0},
        },
        {"type": "view", "part": "candidates", "offset": 0, "limit": 10},
    ]
    for command in commands:
        workbench.submit_command(command)
    workbench.close()
    incremental = workbench.latest_report
    assert incremental is not None
    full = workbench.full_rebuild_report(tmp_path / "blank-cache")
    assert canonical_report_fields(incremental) == canonical_report_fields(full)
    recovered, issues = SnapshotStore(tmp_path / "workspace" / "snapshots").recover()
    assert recovered is not None
    assert not [issue for issue in issues if issue.severity == "error"]
    assert recovered["state"]["command_seq"] == 4
    versions = workbench.snapshots.versions()
    assert len(versions) >= 2
    assert [item["version"] for item in versions] == sorted(item["version"] for item in versions)


def test_snapshot_pointer_version_mismatch_is_detected(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    store.commit({"command_seq": 1}, {"command_seq": 1})
    store.pointer_path.write_text(
        '{"schema_version": 99, "directory": "missing"}', encoding="utf-8"
    )
    payload, issues = store.recover()
    assert payload is None
    assert any(issue.code == "E_SNAPSHOT_VERSION_MISMATCH" for issue in issues)


def test_command_config_validation_is_persisted_in_final_report(tmp_path, base_config, base_data):
    config, data = normalized_pair(base_config, base_data)
    workbench = Workbench(config, data, workspace=tmp_path / "workspace")
    workbench.submit_command(
        {
            "type": "set_scheduler",
            "scheduler": {
                "concurrency": 0,
                "global_iteration_budget": 0,
                "interaction_latency_budget_ms": 0,
            },
        }
    )
    workbench.submit_command(
        {
            "type": "set_preprocessing",
            "preprocessing": {"method": "not-a-method"},
        }
    )
    workbench.drain()
    workbench.close()
    report = workbench.latest_report
    assert report is not None
    codes = {failure["code"] for failure in report["failures"]}
    assert "E_CONCURRENCY_NON_POSITIVE" in codes
    assert "E_GLOBAL_BUDGET_NON_POSITIVE" in codes
    assert "E_LATENCY_BUDGET_NON_POSITIVE" in codes
    assert "E_CONFIG_INVALID" in codes
