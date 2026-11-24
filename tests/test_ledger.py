from __future__ import annotations

from pathlib import Path

import pytest

from cachelease.errors import ReplayError
from cachelease.ledger import Ledger
from cachelease.spec import load_config, load_events

from .helpers import one_node_config, write_events


def test_batch_atomic_records_and_replay_tamper_detection(tmp_path: Path) -> None:
    _, config_path = one_node_config(tmp_path)
    config = load_config(config_path)
    events_path = write_events(
        tmp_path / "events.ndjson",
        [{"seq": 1, "batch": 1, "type": "write", "artifact": "core", "lease_ms": 1000}],
    )
    event = load_events(events_path, config)[0]
    ledger = Ledger(tmp_path / "ledger.sqlite", config)
    try:
        ledger.begin_batch(1, [event])
        watermark = {
            "ordinal": 1,
            "seq": 1,
            "node_id": "node-a",
            "quota_version": 1,
            "used_slots": 1,
            "free_slots": 3,
            "active_leases": 1,
            "lease_remaining": 3,
            "watermark_units": 250,
            "quantization_error_upper": 125,
            "stale": 0,
            "elapsed_ms": 1,
        }
        snapshot = {
            "format_version": 1,
            "config_digest": config.config_digest,
            "last_ordinal": 1,
            "last_seq": 1,
            "nodes": {
                "node-a": {
                    "node_id": "node-a",
                    "quota_version": 1,
                    "capacity_slots": 4,
                    "lease_limit": 4,
                    "watermark": {
                        "used_slots": 1,
                        "free_slots": 3,
                        "active_leases": 1,
                        "lease_remaining": 3,
                        "watermark_units": 250,
                        "quantization_error_upper": 125,
                    },
                    "entries": [],
                    "leases": [],
                    "reservations": [],
                }
            },
            "artifacts": {},
        }
        ledger.commit_batch(
            1,
            events=[
                {
                    "ordinal": 1,
                    "seq": 1,
                    "event_type": "write",
                    "event_digest": event.digest,
                    "event": event.canonical_payload(),
                    "outcome": {"node_id": "node-a"},
                    "status": "ok",
                    "error_code": None,
                }
            ],
            watermarks=[watermark],
            snapshot=snapshot,
        )
        assert ledger.replay_and_validate({"node-a"})["events"] == 1
        ledger.connection.execute("UPDATE watermarks SET watermark_units=999 WHERE ordinal=1")
        with pytest.raises(ReplayError) as exc:
            ledger.replay_and_validate({"node-a"})
        assert exc.value.code == "E_REPLAY"
    finally:
        ledger.close()


def test_open_batch_is_marked_unknown_for_rerun(tmp_path: Path) -> None:
    _, config_path = one_node_config(tmp_path)
    config = load_config(config_path)
    events_path = write_events(
        tmp_path / "events.ndjson",
        [{"seq": 1, "batch": 1, "type": "write", "artifact": "core", "lease_ms": 1000}],
    )
    event = load_events(events_path, config)[0]
    ledger = Ledger(tmp_path / "ledger.sqlite", config)
    try:
        ledger.begin_batch(1, [event])
        open_rows = ledger.mark_open_batches_unknown("simulated hard kill")
        assert open_rows and open_rows[0]["batch_id"] == 1
        status = ledger.connection.execute("SELECT status FROM batches WHERE batch_id=1").fetchone()
        assert status["status"] == "unknown"
        actions = ledger.recovery_actions()
        assert actions[0]["kind"] == "UNKNOWN_UNCOMMITTED_BATCH"
        ledger.replay_and_validate({"node-a"})
    finally:
        ledger.close()

