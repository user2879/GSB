from __future__ import annotations

from candidate_diagnostics.config import Config, deep_merge
from candidate_diagnostics.engine import IncrementalEngine
from candidate_diagnostics.scheduler import (
    SnapshotStore,
    run_scheduled_snapshot,
    schedule_requests,
)


def test_budget_consumes_stable_request_order_and_marks_pending(config, requests):
    plan = schedule_requests(requests, config)
    assert plan["processed"] == ["req-001", "req-002", "req-003", "req-004"]
    assert plan["completed_cursor"] == "req-004"
    assert plan["pending"][0] == "req-005"
    assert plan["remaining_budget"] == 0
    pending_failures = [f for f in plan["failures"] if f["code"] == "E_POINT_BUDGET_PENDING"]
    assert [f["request_id"] for f in pending_failures] == plan["pending"]


def test_request_quota_exceeded_is_isolated(config, requests):
    strict = Config(deep_merge(config.raw, {"budget": {"request_point_quota": 2}}))
    plan = schedule_requests(requests, strict)
    assert "req-001" in plan["quota_exceeded"]
    assert any(f["code"] == "E_REQUEST_QUOTA_EXCEEDED" and f["request_id"] == "req-001"
               for f in plan["failures"])


def test_snapshot_atomic_commit_resume_and_version_change(tmp_path, config, requests):
    store = SnapshotStore(tmp_path / "state", config)
    engine = IncrementalEngine(config)
    run, plan, manifest = run_scheduled_snapshot(engine, requests, config, store)
    assert run is not None
    assert (tmp_path / "state" / "manifest.json").exists()
    assert (tmp_path / "state" / "sampling.json").exists()
    assert (tmp_path / "state" / "pending_queue.json").exists()
    assert manifest["completed_cursor"] == plan["completed_cursor"]

    resumed = SnapshotStore(tmp_path / "state", config).begin_run()
    assert resumed["startup_decision"] == "RESUME_SAME_VERSION"

    changed = Config(deep_merge(config.raw, {"top_k": config.top_k + 2}))
    changed_manifest = SnapshotStore(tmp_path / "state", changed).begin_run()
    assert changed_manifest["startup_decision"] == "RESTART_VERSION_CHANGED"
    assert changed_manifest["global_version"] == manifest["global_version"] + 1
