from __future__ import annotations

import pytest

from trace_diag.errors import SnapshotError
from trace_diag.ledger import Ledger
from trace_diag.spec import PoliciesDoc
from trace_diag.workflow import (
    SimulatedCrash,
    analyze_and_persist,
    recover_latest,
)


def test_ledger_commits_complete_batches_and_verifies_snapshot(tmp_path, service_doc, policy_doc):
    db = tmp_path / "ledger.sqlite"
    result = analyze_and_persist(service_doc, policy_doc, db)
    assert result.view["run"]["status"] == "complete"
    assert result.view["run"]["active"] is True
    assert result.view["run"]["committed_batches"] == 2
    with Ledger(db) as ledger:
        view = ledger.get_run_view(verify=True)
        assert view["run"]["run_id"] == result.run_id
        assert ledger.verify_run(result.run_id) == view["run"]["snapshot_digest"]
        bucket_count = ledger.connection.execute(
            "SELECT COUNT(*) AS n FROM bucket_stats WHERE run_id=?", (result.run_id,)
        ).fetchone()["n"]
        attribution_count = ledger.connection.execute(
            "SELECT COUNT(*) AS n FROM attribution_chain WHERE run_id=?", (result.run_id,)
        ).fetchone()["n"]
        assert bucket_count == 2 * 2 * 16
        assert attribution_count == 2 * (2 * 16 + 2)
        assert view["flips"]


def test_changed_policy_gets_monotonic_version_and_stales_old_run(tmp_path, service_doc, policy_doc):
    db = tmp_path / "ledger.sqlite"
    first = analyze_and_persist(service_doc, policy_doc, db)
    changed_policy = policy_doc.policies[0].variant(scale_power=2)
    changed_doc = PoliciesDoc(
        version=1,
        policies=(changed_policy, *policy_doc.policies[1:]),
        description=policy_doc.description,
    )
    second = analyze_and_persist(service_doc, changed_doc, db)
    assert second.bundle.policies_doc.version == 2
    assert second.bundle.service_analyses[0]["policies"][changed_policy.id]["policy"][
        "scale_power"
    ] == 2
    with Ledger(db) as ledger:
        runs = ledger.list_runs()
        by_id = {run["run_id"]: run for run in runs}
        assert by_id[first.run_id]["stale"] == 1
        assert by_id[first.run_id]["active"] == 0
        assert by_id[second.run_id]["active"] == 1
        assert by_id[second.run_id]["stale"] == 0
        current = ledger.get_run_view(verify=True)
        assert current["run"]["run_id"] == second.run_id
        assert [service["service_id"] for service in current["services"]] == [
            "svc-a",
            "svc-b",
        ]


def test_forced_stop_restores_only_committed_batches(tmp_path, service_doc, policy_doc):
    db = tmp_path / "ledger.sqlite"
    with pytest.raises(SimulatedCrash):
        analyze_and_persist(service_doc, policy_doc, db, crash_after=1)
    with Ledger(db) as ledger:
        run = ledger.latest_any_run()
        assert run["status"] == "recovering"
        assert ledger.get_current_run_row() is None
        batches = ledger.list_batches(run["run_id"])
        assert [batch["status"] for batch in batches] == ["committed", "unknown"]
    result = recover_latest(db)
    assert result.mode == "resume"
    assert result.newly_committed == ("svc-b",)
    assert result.skipped_committed == ("svc-a",)
    assert result.view["run"]["status"] == "complete"
    assert all(service["status"] == "committed" for service in result.view["services"])


def test_snapshot_digest_injection_is_detected(tmp_path, service_doc, policy_doc):
    db = tmp_path / "ledger.sqlite"
    result = analyze_and_persist(service_doc, policy_doc, db)
    with Ledger(db, mark_pending_unknown=False) as ledger:
        original = ledger.corrupt_snapshot_digest(result.run_id)
        with pytest.raises(SnapshotError) as captured:
            ledger.get_run_view(result.run_id, verify=True)
        assert captured.value.code == "E_SNAPSHOT"
        ledger.restore_snapshot_digest(result.run_id, original)
        assert ledger.verify_run(result.run_id)


def test_changed_sample_description_gets_monotonic_service_version(tmp_path, service_doc, policy_doc):
    from dataclasses import replace

    from trace_diag.spec import ServicesDoc

    db = tmp_path / "ledger.sqlite"
    first = analyze_and_persist(service_doc, policy_doc, db)
    changed_services = tuple(
        replace(service, description="edited in the workbench")
        if service.id == "svc-a"
        else service
        for service in service_doc.services
    )
    changed_doc = ServicesDoc(
        version=1,
        services=changed_services,
        description=service_doc.description,
    )
    second = analyze_and_persist(changed_doc, policy_doc, db)
    assert second.bundle.services_doc.version == 2
    with Ledger(db) as ledger:
        by_id = {run["run_id"]: run for run in ledger.list_runs()}
        assert by_id[first.run_id]["stale"] == 1
        assert by_id[second.run_id]["service_version"] == 2
        assert by_id[second.run_id]["active"] == 1
