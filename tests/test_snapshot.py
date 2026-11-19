from __future__ import annotations

import json

import pytest

from tcp_workbench.errors import E_REPLAY, E_VERSION, BusinessError
from tcp_workbench.models import Event
from tcp_workbench.snapshot import seed_snapshot
from tests.conftest import SAMPLES


def _ingest_sample(service, sample_events):
    for raw in sample_events:
        service.ingest_event(Event.from_dict({**raw, "event_seq": 0}))


def test_snapshot_save_verify_and_historical_versions(service_factory, sample_events):
    service = service_factory()
    _ingest_sample(service, sample_events)
    baseline_id = service.save_snapshot("current-v1")
    assert service.verify_snapshot(baseline_id)["verified"] is True
    seeded = []
    for path in sorted((SAMPLES / "snapshots").glob("*.json")):
        seed = json.loads(path.read_text(encoding="utf-8"))
        seeded.append(
            seed_snapshot(
                service.ledger,
                seed,
                mono_ms=seed["created_mono_ms"],
                base_dir=path.parent,
            )
        )
    assert len(seeded) == 2
    for snapshot_id in seeded:
        assert service.verify_snapshot(snapshot_id)["verified"] is True
    rows = service.ledger.snapshots()
    assert len(rows) >= 3
    assert sorted({(row["config_version"], row["script_version"]) for row in rows}) == [
        (1, 1),
        (2, 2),
    ]


def test_restore_mismatched_snapshot_fully_replaces_current(
    service_factory, sample_events
):
    service = service_factory()
    _ingest_sample(service, sample_events)
    seed_path = SAMPLES / "snapshots" / "02_v2_revised.json"
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    snapshot_id = seed_snapshot(
        service.ledger,
        seed,
        mono_ms=seed["created_mono_ms"],
        base_dir=seed_path.parent,
    )
    report = service.restore_snapshot(snapshot_id)
    assert report.reason == "snapshot_version_content_replacement"
    assert report.old_config_version == 1 and report.new_config_version == 2
    assert report.old_script_version == 1 and report.new_script_version == 2
    assert report.affected_fields
    assert [change.field for change in report.affected_fields] == sorted(
        change.field for change in report.affected_fields
    )
    assert all(item.config_version == 2 for item in service.items.values())
    assert all(item.stale for item in service.ledger.conclusion_history())
    service.verify_incremental_equivalence()


def test_snapshot_missing_version_is_e_version(service_factory, sample_events):
    service = service_factory()
    _ingest_sample(service, sample_events)
    snapshot_id = service.save_snapshot("missing-version")
    with service.ledger.conn:
        service.ledger.conn.execute(
            "UPDATE snapshots SET config_version = NULL WHERE snapshot_id = ?",
            (snapshot_id,),
        )
    with pytest.raises(BusinessError) as shown:
        service.restore_snapshot(snapshot_id)
    assert shown.value.code == E_VERSION
    assert "snapshot.config_version" in shown.value.fields


def test_replay_detects_first_inconsistent_event_and_field(
    service_factory, sample_events
):
    service = service_factory()
    _ingest_sample(service, sample_events)
    payload = service.snapshot_payload("tampered-replay")
    payload["events"][4]["rtt_sample_ms"] = 99
    bad_id = service.ledger.save_snapshot(payload)
    with pytest.raises(BusinessError) as shown:
        service.verify_snapshot(bad_id)
    error = shown.value
    assert error.code == E_REPLAY
    assert error.context["first_event_seq"] == 5
    assert error.context["field"].startswith("rtt.")
    assert "event:5" in error.fields
