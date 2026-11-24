from __future__ import annotations

import pytest

from probelease.errors import LeaseStateError
from probelease.transport import ACK, NO_RESPONSE, SEQ_MISMATCH


def test_real_udp_outcomes_are_distinct(orchestrator):
    alpha = orchestrator.execute_round("alpha")
    outcomes = {item["probe_id"]: item["outcome"] for item in alpha.probes}
    assert outcomes == {
        "alpha_ok_1": ACK,
        "alpha_ok_2": ACK,
        "alpha_drop": NO_RESPONSE,
    }
    assert alpha.ratio_bp == 6667

    mismatch = orchestrator.execute_round("mismatch")
    probe = mismatch.probes[0]
    assert probe["outcome"] == SEQ_MISMATCH
    assert probe["response_seq"] == 2
    assert probe["reason"] == "sequence_mismatch:expected=1:received=2"
    assert mismatch.status == "PAUSED"
    assert mismatch.pause_reason == "LEASE_EXPIRED"

    late = orchestrator.execute_round("late")
    late_probe = late.probes[0]
    assert late_probe["outcome"] == NO_RESPONSE
    elapsed_ns = late_probe["receive_mono_ns"] - late_probe["send_mono_ns"]
    assert elapsed_ns >= 55_000_000
    assert late.ratio_bp == 0


def test_renewal_expiry_and_dependency_gate(orchestrator):
    versions = [orchestrator.execute_round("alpha").lease_version for _ in range(3)]
    assert versions == [2, 3, 4]
    assert orchestrator.query_lease("alpha")["status"] == "ACTIVE"

    first_lossy = orchestrator.execute_round("lossy")
    assert (first_lossy.ratio_bp, first_lossy.status) == (5000, "ACTIVE")
    assert first_lossy.lease_version == 1
    second_lossy = orchestrator.execute_round("lossy")
    assert second_lossy.ratio_bp == 5000
    assert second_lossy.status == "PAUSED"
    assert second_lossy.pause_reason == "LEASE_EXPIRED"
    assert second_lossy.lease_version == 1

    child = orchestrator.query_lease("child")
    assert child["status"] == "PAUSED"
    assert child["pause_reason"] == "DEPENDENCY_LOST"
    assert child["lease_version"] == 1
    with pytest.raises(LeaseStateError) as caught:
        orchestrator.execute_round("lossy")
    assert caught.value.code == "E_LEASE"

    terminated = orchestrator.terminate_session("mismatch", "TEST_TERMINATE")
    assert terminated["status"] == "TERMINATED"
    with pytest.raises(LeaseStateError):
        orchestrator.execute_round("mismatch")


def test_resume_preserves_cursor_and_continues_cycle(orchestrator):
    orchestrator.execute_round("alpha")
    state = orchestrator.pause_session("alpha", "TEST_PAUSE")
    assert state["cycle_cursor"] == 1
    resumed = orchestrator.resume_session("alpha")
    assert resumed["status"] == "ACTIVE"
    result = orchestrator.execute_round("alpha")
    assert result.cycle == 2
    events = orchestrator.ledger.pause_events("alpha")
    assert [(e["kind"], e["cycle"], e["reason"]) for e in events] == [
        ("PAUSE", 2, "TEST_PAUSE"),
        ("RESUME", 2, "USER_RESUME"),
    ]

