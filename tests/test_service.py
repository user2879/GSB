from __future__ import annotations

import json
import sqlite3

from graygate.routing import decide_route
from graygate.service import WorkbenchService
from graygate.util import canonical_json, content_digest


def _sample(index: int, size: int = 8) -> list[float]:
    return [((index + offset * 3) % 17) / 17.0 for offset in range(size)]


def _ingest_many(service: WorkbenchService, count: int, start: int = 0) -> list[dict]:
    results = []
    for index in range(start, start + count):
        results.append(
            service.ingest(
                f"client-{index % 9}",
                "alpha:payload" if index % 2 == 0 else "beta:payload",
                _sample(index),
            )
        )
    return results


def test_snapshot_is_authoritatively_consistent(sample_spec, tmp_path):
    service = WorkbenchService(sample_spec, tmp_path / "workbench.db")
    try:
        responses = _ingest_many(service, 64)
        assert all(response["ok"] for response in responses)
        snapshot = service.rebuild_now()
        assert snapshot.source_ledger_version == 64
        result = service.check_consistency()
        assert result.ok, result.canonical()
        assert result.code == "OK"
        assert result.stats_digest.startswith("sha256:")
        replay = service.recover_and_replay()
        assert replay["ledger_chain"]["ok"] is True
        assert all(item["ok"] for item in replay["replays"])
        assert replay["exit_report_claimed"] is False
    finally:
        service.close()


def test_rebuild_worker_excludes_hits_but_admission_continues(sample_spec, tmp_path):
    service = WorkbenchService(sample_spec, tmp_path / "workbench.db")
    try:
        _ingest_many(service, 12)
        rebuild_id = service.start_rebuild(pause_ms=300)
        added = _ingest_many(service, 8, start=100)
        assert all(response["ok"] for response in added)
        info = service.wait_rebuild(rebuild_id, timeout=10)
        assert info["status"] == "COMPLETE"
        snapshot = service.current_snapshot
        assert snapshot is not None
        assert snapshot.source_ledger_version == 12
        assert snapshot.observed_ledger_version >= 20
        assert snapshot.excluded_hits >= 8
        assert sum(row["stable_hits"] + row["canary_hits"] for row in snapshot.stats) == 12
        result = service.check_consistency()
        assert result.ok, result.canonical()

        hits = service.ledger.hits_upto(snapshot.observed_ledger_version)
        for hit in hits:
            route = sample_spec.routes_by_id[hit.rule_id]
            decision = decide_route(
                sample_spec, route, hit.request_seq, hit.client_key
            )
            assert decision.side == hit.side
            assert decision.backend_version == hit.backend_version
    finally:
        service.close()


def test_snapshot_consistency_reports_first_mismatch(sample_spec, tmp_path):
    service = WorkbenchService(sample_spec, tmp_path / "workbench.db")
    try:
        _ingest_many(service, 20)
        service.rebuild_now()
        snapshot = service.current_snapshot
        assert snapshot is not None
        snapshot.body["stats"][0]["stable_hits"] += 1
        result = service.check_consistency()
        assert result.code == "E_SNAPSHOT"
        assert result.rule_id == snapshot.stats[0]["rule_id"]
        assert result.stage_id == snapshot.stats[0]["stage_id"]
        assert result.field_name == "stable_hits"
    finally:
        service.close()


def test_recovery_replay_reports_first_snapshot_field_mismatch(sample_spec, tmp_path):
    db_path = tmp_path / "workbench.db"
    service = WorkbenchService(sample_spec, db_path)
    try:
        _ingest_many(service, 20)
        snapshot = service.rebuild_now()
    finally:
        service.close()

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT snapshot_id, body_json FROM snapshots WHERE snapshot_id=?",
            (snapshot.snapshot_id,),
        ).fetchone()
        body = json.loads(row[1])
        body["stats"][0]["stable_hits"] += 1
        connection.execute(
            "UPDATE snapshots SET body_json=?, content_digest=? WHERE snapshot_id=?",
            (canonical_json(body), content_digest(body), row[0]),
        )
        connection.commit()

    recovered = WorkbenchService(sample_spec, db_path)
    try:
        report = recovered.recover_and_replay()
        assert report["replays"][0]["code"] == "E_REPLAY"
        assert report["replays"][0]["first_mismatch"]["rule_id"] == body["stats"][0]["rule_id"]
        assert report["replays"][0]["first_mismatch"]["field"] == "stable_hits"
    finally:
        recovered.close()


def test_query_over_budget_reports_e_latency(sample_spec, tmp_path, monkeypatch):
    import graygate.service as service_module

    service = WorkbenchService(sample_spec, tmp_path / "workbench.db")
    try:
        values = iter([1000, 1000 + sample_spec.query_latency_budget_ms + 17])
        monkeypatch.setattr(service_module, "monotonic_ms", lambda: next(values))
        dashboard = service.dashboard()
        assert dashboard["ok"] is False
        assert dashboard["errors"][0]["code"] == "E_LATENCY"
        assert dashboard["errors"][0]["over_by_ms"] == 17
        assert "routing unaffected" in dashboard["errors"][0]["impact_scope"]
    finally:
        service.close()


def test_changed_rule_stage_is_reported(sample_spec, tmp_path):
    service = WorkbenchService(sample_spec, tmp_path / "workbench.db")
    try:
        _ingest_many(service, 10)
        first = service.rebuild_now()
        _ingest_many(service, 6, start=50)
        second = service.rebuild_now()
        assert second.source_ledger_version > first.source_ledger_version
        changes = service.last_changes
        assert changes
        assert all("rule_id" in item and "stage_id" in item for item in changes)
        assert all(item["changed_fields"] for item in changes)
    finally:
        service.close()
