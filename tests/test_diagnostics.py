from __future__ import annotations

from quantledger.diagnostics import (
    run_fault_injections,
    run_forced_kill_recovery,
    run_lease_checks,
)
from quantledger.engine import LedgerEngine


def test_fault_injection_ids_and_protocol_isolation(local_fixture):
    report = run_fault_injections(
        local_fixture["config"],
        local_fixture["injections"],
        runtime_dir=local_fixture["root"] / "faults",
    )
    assert report["all_hit"], report
    assert [item["id"] for item in report["injections"]] == [
        "T-INJ-001",
        "T-INJ-002",
        "T-INJ-003",
        "T-INJ-004",
    ]
    assert report["isolation"]["valid_request_after_faults"] is True
    assert report["worker_status"]["active_leases"] == 0


def test_real_lease_expiry_and_forced_kill_recovery(local_fixture):
    lease = run_lease_checks(local_fixture["config"], runtime_dir=local_fixture["root"] / "leases")
    assert lease["ok"], lease
    assert lease["expiry"]["fetch_code"] == "E_RECONCILE_PENDING"
    assert lease["expiry"]["E_LEASE_EXPIRED"] >= 1
    assert lease["expiry"]["active_leases_after"] == 0
    assert lease["renewal"]["stale_release_code"] == "E_STALE_GENERATION"
    assert lease["renewal"]["generation_unchanged"] is True

    seed_dir = local_fixture["root"] / "seed"
    LedgerEngine(
        local_fixture["config"],
        commands_path=local_fixture["commands"],
        runtime_dir=seed_dir,
        request_timeout_s=10,
    ).run(rebuild=True)
    recovery = run_forced_kill_recovery(
        local_fixture["config"],
        seed_ledger=seed_dir / "ledger.json",
        runtime_dir=local_fixture["root"] / "recovery",
    )
    assert recovery["worker_force_killed"] is True
    assert recovery["atomic_ledger_only"] is True
    assert recovery["pending_queue"]
    assert recovery["E_RECONCILE_PENDING"] >= 1

