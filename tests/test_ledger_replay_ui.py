from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from leasebench.errors import SnapshotError
from leasebench.ledger import Ledger
from leasebench.models import content_digest
from leasebench.service import Workbench, load_view_from_ledger, replay_ledger
from leasebench.ui import render_view_text


def run_two_versions(tmp_path, declaration_v1, declaration_v2, events, session):
    db = tmp_path / "ledger.sqlite"
    with Workbench(
        db_path=db,
        declaration=declaration_v1,
        events=events,
        session=session,
    ) as workbench:
        first = workbench.run(limit_event_seq=6)
        assert first.final_cursor == 6
    with Workbench(
        db_path=db,
        declaration=declaration_v2,
        events=events,
        session=session,
    ) as workbench:
        second = workbench.run()
        assert second.final_cursor == 11
        assert second.declaration_version == 2
        assert second.recovery_status == "CLEAN"
        assert second.view["invalidations"]
        assert all(
            item["declaration_version"] == 2
            for item in second.view["preemptions"]
        )
        assert any(
            item["kind"] == "PREEMPTION"
            for item in second.view["ledger_invalidations"]
        )
    return db


def test_ledger_replay_and_views(tmp_path, declaration_v1, declaration_v2, events, session) -> None:
    db = run_two_versions(tmp_path, declaration_v1, declaration_v2, events, session)
    result = replay_ledger(db_path=db)
    assert result.ok
    assert result.batches_checked == 9
    assert result.events_checked == 11
    assert result.declaration_versions == [1, 1, 1, 1, 2, 2, 2, 2, 2]

    view = load_view_from_ledger(db)
    assert view["declaration_version"] == 2
    assert view["ledger_cursor"] == 11
    assert view["unknown_lease_ids"] == []
    assert any(item["kind"] == "LEASE" for item in view["invalidations"])
    assert all(item["reason"] for item in view["invalidations"])
    assert any(item["borrow_return_error_fp"] >= 0 for item in view["watermarks"])


def test_pending_intent_is_unknown_then_recomputed(tmp_path, declaration_v1, events, session) -> None:
    db = tmp_path / "pending.sqlite"
    ledger = Ledger(db)
    ledger.begin_pending(
        batch_id="B0001-00000001-00000003",
        declaration_version=1,
        start_event_seq=1,
        end_event_seq=3,
        candidate_lease_ids=["L00000001", "L00000002", "L00000003"],
    )
    report = ledger.recover()
    assert report.status == "PENDING_RECOMPUTE"
    assert report.committed_cursor == 0
    assert set(report.unknown_lease_ids) == {"L00000001", "L00000002", "L00000003"}
    ledger.close()

    with Workbench(
        db_path=db,
        declaration=declaration_v1,
        events=events,
        session=session,
    ) as workbench:
        result = workbench.run()
        assert result.recovery_status == "CLEAN"
        assert result.unknown_lease_ids == []
        assert set(result.recomputed_unknown_lease_ids) == {
            "L00000001",
            "L00000002",
            "L00000003",
        }
        assert result.final_cursor == 11


def test_snapshot_tamper_is_detected(tmp_path, declaration_v1, events, session) -> None:
    db = tmp_path / "tamper.sqlite"
    with Workbench(
        db_path=db,
        declaration=declaration_v1,
        events=events,
        session=session,
    ) as workbench:
        workbench.run(limit_event_seq=6)
        workbench.ledger.tamper_content_digest()
    with pytest.raises(SnapshotError) as caught:
        replay_ledger(db_path=db)
    assert caught.value.code == "E_SNAPSHOT"
    assert caught.value.field == "content_digest"


def _rewrite_chain(db, payload_mutator) -> None:
    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    try:
        rows = list(connection.execute("SELECT * FROM batches ORDER BY sequence_no"))
        previous = ""
        for index, row in enumerate(rows):
            payload = json.loads(row["result_json"])
            if index == 0:
                payload_mutator(payload)
            result_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            content = content_digest(payload)
            chain = hashlib.sha256((previous + content).encode("utf-8")).hexdigest()
            connection.execute(
                "UPDATE batches SET result_json=?, content_digest=?, prev_chain_digest=?, chain_digest=? WHERE batch_id=?",
                (result_json, content, previous, chain, row["batch_id"]),
            )
            previous = chain
        connection.execute(
            "UPDATE meta SET value=? WHERE key='chain_digest'", (previous,)
        )
        connection.commit()
    finally:
        connection.close()


def test_replay_reports_first_semantic_mismatch(tmp_path, declaration_v1, events, session) -> None:
    db = tmp_path / "semantic.sqlite"
    with Workbench(
        db_path=db,
        declaration=declaration_v1,
        events=events,
        session=session,
    ) as workbench:
        workbench.run(limit_event_seq=6)

    def mutate(payload):
        for operation in payload["operations"]:
            if operation["type"] == "GRANT":
                operation["lease"]["concurrency"] += 1
                return
        raise AssertionError("no grant operation")

    _rewrite_chain(db, mutate)
    with pytest.raises(Exception) as caught:
        replay_ledger(db_path=db)
    assert getattr(caught.value, "code", None) == "E_REPLAY"
    assert caught.value.details["event_seq"] == 1
    assert caught.value.details["lease_id"] == "L00000001"


def test_text_rendering_is_stable_and_sorted(tmp_path, declaration_v1, declaration_v2, events, session) -> None:
    db = run_two_versions(tmp_path, declaration_v1, declaration_v2, events, session)
    view = load_view_from_ledger(db)
    first = render_view_text(view)
    second = render_view_text(view)
    assert first == second
    assert "ACTIVE LEASES" in first
    assert "FIXED-POINT WATERMARKS" in first
    assert "STALE INVALIDATIONS" in first

    reversed_view = json.loads(json.dumps(view))
    for key in (
        "functions",
        "leases",
        "queue",
        "loans",
        "preemptions",
        "watermarks",
        "invalidations",
        "ledger_invalidations",
    ):
        reversed_view[key] = list(reversed(reversed_view[key]))
    assert render_view_text(reversed_view) == first

    before = view["view_digest"]
    _ = load_view_from_ledger(db)
    after = load_view_from_ledger(db)["view_digest"]
    assert before == after
