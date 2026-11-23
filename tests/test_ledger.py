from __future__ import annotations

import pytest

from joinlab import LabError, load_workload
from joinlab.ledger import Ledger
from joinlab.session import ExperimentSession

from conftest import SAMPLE


def test_batch_transaction_snapshot_and_replay(tmp_path):
    session = ExperimentSession(load_workload(SAMPLE))
    db = tmp_path / "ledger.sqlite"
    with Ledger(db) as ledger:
        report = session.run(ledger)
        assert len(report["commits"]) == 2
        assert [batch["batch_index"] for batch in ledger.committed_batches(session.run_id)] == [0, 1]
        assert ledger.latest_snapshot(session.run_id)["version"] == 2
        integrity = ledger.verify_integrity(session.run_id)
        assert integrity["batches_checked"] == 2
        replay = session.replay_check(ledger)
        assert replay["ok"] is True
        assert replay["restored_snapshot_version"] == 2
        assert replay["remaining_candidate_ids"] == []
        assert all(item["digests_consistent"] for item in replay["checked_batches"])


def test_interrupted_batch_is_unknown_and_resume_reruns_it(tmp_path):
    workload = load_workload(SAMPLE)
    session = ExperimentSession(workload)
    db = tmp_path / "resume.sqlite"
    with Ledger(db) as ledger:
        with pytest.raises(LabError) as caught:
            session.run(ledger, interrupt_before_commit=1)
        assert caught.value.code == "E_INTERRUPTED"
        assert len(ledger.committed_batches(session.run_id)) == 1
        restored, snapshot = session.restore_from_ledger(ledger)
        assert snapshot["version"] == 1
        assert restored.progress["cursor"] == 1
        report = restored.run(ledger)
        assert report["recovered_unknown_attempts"]
        assert report["recovered_unknown_attempts"][0]["requires_rerun"] is True
        assert len(ledger.committed_batches(session.run_id)) == 2
        assert restored.replay_check(ledger)["ok"] is True


def test_ledger_digest_and_normalized_content_tampering_is_detected(tmp_path):
    session = ExperimentSession(load_workload(SAMPLE))
    with Ledger(tmp_path / "tamper.sqlite") as ledger:
        session.run(ledger)
        ledger.tamper_for_test("batch_digest")
        with pytest.raises(LabError) as caught:
            ledger.verify_integrity(session.run_id)
        assert caught.value.code == "E_LEDGER"
        assert caught.value.field == "batches.content_digest"

    second = ExperimentSession(load_workload(SAMPLE))
    with Ledger(tmp_path / "tamper2.sqlite") as ledger:
        second.run(ledger)
        ledger.tamper_for_test("estimate_cost")
        with pytest.raises(LabError) as caught:
            ledger.verify_integrity(second.run_id)
        assert caught.value.code == "E_LEDGER"
        assert caught.value.field.startswith("batch.estimates[")


def test_replay_finds_first_operator_field_when_ledger_is_self_consistent(tmp_path):
    session = ExperimentSession(load_workload(SAMPLE))
    with Ledger(tmp_path / "replay.sqlite") as ledger:
        session.run(ledger)
        ledger.tamper_for_test("replay_operator")
        assert ledger.verify_integrity(session.run_id)["ok"] is True
        with pytest.raises(LabError) as caught:
            session.replay_check(ledger, reexecute=False)
        error = caught.value
        assert error.code == "E_REPLAY"
        assert error.candidate_id
        assert error.operator_id
        assert error.field.endswith("output_rows_fx")


def test_snapshot_version_restore_and_format_version(tmp_path):
    session = ExperimentSession(load_workload(SAMPLE))
    with Ledger(tmp_path / "versions.sqlite") as ledger:
        session.run(ledger)
        first = ledger.get_snapshot(session.run_id, 1)
        second = ledger.get_snapshot(session.run_id, 2)
        assert first["version"] == 1
        assert second["version"] == 2
        assert first["format_version"] == 1
        restored, snapshot = session.restore_from_ledger(ledger, snapshot_version=1)
        assert snapshot["version"] == 1
        assert restored.progress["cursor"] == 1
        assert len(restored.execution_summaries) == 1

