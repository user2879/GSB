from __future__ import annotations

import copy
import json

import pytest

from modeldash.errors import WorkbenchError
from modeldash.gui.view import format_text_view
from modeldash.replay import ReplayVerifier
from modeldash.scheduler import Scheduler
from modeldash.spec import validate_spec
from modeldash.storage import Ledger
from modeldash.util import canonical_json, digest_json

from conftest import FakeDispatcher
from test_scheduler import minimal_raw


def execute(tmp_path, raw, *, database_name="ledger.sqlite3"):
    spec = validate_spec(raw)
    ledger = Ledger(tmp_path / database_name)
    activation = ledger.activate_spec(spec)
    summary = Scheduler(spec, ledger, activation.session_id, FakeDispatcher(delay_ms=2)).run()
    return spec, ledger, activation, summary


def evolution_raw(raw):
    value = copy.deepcopy(raw)
    value["spec_version"] = 2
    value["configs"] = [
        {
            "version": 3,
            "priority": 5,
            "target_nodes": ["n1"],
            "shards": 2,
            "items": {"alpha": 21, "beta": 8, "gamma": "v3"},
        }
    ]
    value["budget"] = {
        "global_concurrency": 1,
        "timeout_ms": 900,
        "retry_backoff_ms": [2, 4],
        "max_retries": 1,
    }
    value["tasks"] = [{"id": "new-task", "config_version": 3, "node": "n1", "priority": 5}]
    return value


def test_spec_revision_replaces_view_and_marks_old_rows_stale(tmp_path):
    raw = minimal_raw(shards=2)
    old_spec, ledger, old_activation, old_summary = execute(tmp_path, raw)
    ledger.create_snapshot(old_activation.session_id, ledger.latest_state(old_activation.session_id), "old-final")
    new_raw = evolution_raw(raw)
    new_spec = validate_spec(new_raw)
    try:
        result = ledger.activate_spec(new_spec)
        assert result.action == "replaced"
        assert result.state is None
        assert result.stale_summary is not None
        assert result.stale_summary["affected_nodes"] == ["n1"]
        assert result.stale_summary["affected_tasks"] == ["t1"]
        old_row = ledger.conn.execute(
            "SELECT is_stale,replacement_reason FROM sessions WHERE session_id=?", (old_activation.session_id,)
        ).fetchone()
        assert old_row["is_stale"] == 1
        assert "replaced" in old_row["replacement_reason"]
        for table in ("rounds", "tasks", "preemptions", "receipts", "snapshots"):
            stale_count = ledger.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id=? AND is_stale=1", (old_activation.session_id,)
            ).fetchone()[0]
            total_count = ledger.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id=?", (old_activation.session_id,)
            ).fetchone()[0]
            assert stale_count == total_count
        current = ledger.current_session()
        assert current["session_id"] == result.session_id
        view = ledger.diagnostic_view()
        assert view["session"]["spec_version"] == 2
        assert view["priority_queue"] == []
        assert view["receipts"] == []
        evolutions = ledger.stale_evolutions()
        assert evolutions[-1]["old_session_id"] == old_activation.session_id
        assert evolutions[-1]["new_session_id"] == result.session_id
    finally:
        ledger.close()


def test_lower_changed_spec_version_is_rejected_before_replacement(tmp_path):
    raw = minimal_raw()
    _, ledger, _, _ = execute(tmp_path, raw)
    try:
        changed = evolution_raw(raw)
        changed["spec_version"] = 1
        with pytest.raises(WorkbenchError) as exc:
            ledger.activate_spec(validate_spec(changed))
        assert exc.value.code == "E_SPEC"
        assert ledger.current_session()["spec_version"] == 1
    finally:
        ledger.close()


def test_snapshot_digest_mismatch_forces_whole_replacement(tmp_path):
    raw = minimal_raw(shards=1)
    spec, ledger, activation, _ = execute(tmp_path, raw)
    snapshot = ledger.create_snapshot(activation.session_id, ledger.latest_state(activation.session_id), "good")
    ledger.conn.execute(
        "UPDATE snapshots SET payload_json=? WHERE snapshot_id=?",
        (canonical_json({"format_version": 1, "tampered": True}), snapshot["snapshot_id"]),
    )
    try:
        load = ledger.load_snapshot(snapshot["snapshot_id"], spec)
        assert load.action == "REPLACED"
        replaced = ledger.restore_snapshot_or_replace(activation.session_id, snapshot["snapshot_id"], spec)
        assert replaced.action == "REPLACED"
        old = ledger.conn.execute(
            "SELECT is_stale FROM sessions WHERE session_id=?", (activation.session_id,)
        ).fetchone()
        assert old["is_stale"] == 1
        assert ledger.current_session()["session_id"] != activation.session_id
        assert ledger.diagnostic_view()["priority_queue"] == []
    finally:
        ledger.close()


def test_read_only_diagnostic_refresh_is_stable_and_does_not_write(tmp_path):
    raw = minimal_raw(shards=2)
    spec, ledger, activation, _ = execute(tmp_path, raw)
    ledger.create_snapshot(activation.session_id, ledger.latest_state(activation.session_id), "one")
    try:
        data_version = ledger.conn.execute("PRAGMA data_version").fetchone()[0]
        total_changes = ledger.conn.execute("SELECT total_changes()").fetchone()[0]
        first = ledger.diagnostic_view()
        second = ledger.diagnostic_view()
        assert canonical_json(first) == canonical_json(second)
        assert format_text_view(first) == format_text_view(second)
        assert ledger.conn.execute("PRAGMA data_version").fetchone()[0] == data_version
        assert ledger.conn.execute("SELECT total_changes()").fetchone()[0] == total_changes
        assert [item["node_id"] for item in first["node_untrust"]] == ["n1"]
        assert first["snapshots"][0]["label"] == "one"
    finally:
        ledger.close()


def test_offline_replay_verifies_rounds_tasks_wait_and_untrust(tmp_path):
    raw = minimal_raw(shards=2)
    spec, ledger, activation, _ = execute(tmp_path, raw)
    snapshot = ledger.create_snapshot(activation.session_id, ledger.latest_state(activation.session_id), "final")
    summary = ReplayVerifier(
        spec,
        ledger,
        activation.session_id,
        snapshot_id=snapshot["snapshot_id"],
        redispatch=False,
    ).verify()
    assert summary.checks > 0
    assert summary.redispatch_count == 0
    ledger.close()


def test_replay_reports_first_output_mismatch(tmp_path):
    raw = minimal_raw(shards=1)
    spec, ledger, activation, _ = execute(tmp_path, raw)
    rounds = ledger.load_rounds(activation.session_id)
    last = rounds[-1]
    payload = last["payload"]
    state = payload["state"]
    state["tasks"]["t1"]["output_summary"] = "tampered"
    new_payload = canonical_json(payload)
    new_digest = digest_json(payload)
    ledger.conn.execute(
        "UPDATE rounds SET payload_json=?,content_digest=? WHERE session_id=? AND round_seq=?",
        (new_payload, new_digest, activation.session_id, last["round_seq"]),
    )
    try:
        with pytest.raises(WorkbenchError) as exc:
            ReplayVerifier(spec, ledger, activation.session_id).verify()
        assert exc.value.code == "E_REPLAY"
        assert exc.value.context["first_task_id"] == "t1"
        assert exc.value.field == "output_summary"
    finally:
        ledger.close()
