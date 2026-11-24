from __future__ import annotations

import base64
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tunnel_interop.errors import E_REPLAY, E_SNAPSHOT, TunnelError
from tunnel_interop.fragmenter import build_bundle, context_for_plan
from tunnel_interop.ledger import SessionLedger
from tunnel_interop.spec import load_spec_mapping
from tunnel_interop.workflow import (
    batch_context,
    forward_bundle,
    inject_anomaly,
    resume_session,
    snapshot_replacement_report,
)

from conftest import make_spec_data, write_spec


def _local_commit(ledger: SessionLedger, spec, bundle, message):
    plan = message.plan
    batch_id = plan.message_key
    ledger.begin_batch(
        batch_id=batch_id,
        message_key=batch_id,
        context=batch_context(plan, bundle.target_format, len(message.fragments)),
        fragment_indices=[fragment.fragment_index for fragment in message.fragments],
        target_format=bundle.target_format,
    )
    rows = []
    for fragment in message.fragments:
        rows.append(
            {
                "fragment_index": fragment.fragment_index,
                "message_key": fragment.message_key,
                "fragment_offset": fragment.fragment_offset,
                "fragment_length": fragment.fragment_length,
                "total_fragments": fragment.total_fragments,
                "message_id": fragment.message_id,
                "tunnel_id": fragment.tunnel_id,
                "target_format": fragment.target_format,
                "retry_count": fragment.normalized.retry_count,
                "fragment_version": fragment.fragment_version,
                "wire_b64": base64.b64encode(fragment.wire).decode("ascii"),
                "wire_sha256": fragment.wire_sha256,
                "payload_sha256": fragment.normalized.payload_sha256,
                "header_json": json.dumps(fragment.normalized.to_dict(), sort_keys=True),
            }
        )
    return ledger.commit_batch(
        batch_id=batch_id,
        fragments=rows,
        reassembly={
            "message_key": message.reassembly.message_key,
            "payload_sha256": message.reassembly.payload_sha256,
            "payload_b64": base64.b64encode(message.reassembly.payload).decode("ascii"),
            "reassembly_version": message.reassembly.reassembly_version,
            "complete": True,
        },
        scheduling_statistics={"global_peak_inflight": 1, "queued_fragments": 0},
        scheduling_events=[{"kind": "test"}],
        cursor={"state": "committed"},
    )


def test_intent_is_unknown_without_exit_report(tmp_path, spec):
    path = tmp_path / "session.db"
    bundle = build_bundle(spec)
    message = bundle.messages[0]
    with SessionLedger(path) as ledger:
        ledger.open(spec)
        ledger.begin_batch(
            batch_id=message.plan.message_key,
            message_key=message.plan.message_key,
            context=batch_context(message.plan, bundle.target_format, len(message.fragments)),
            fragment_indices=[fragment.fragment_index for fragment in message.fragments],
            target_format=bundle.target_format,
        )
    with SessionLedger(path) as ledger:
        opened = ledger.open(spec)
        replay = ledger.replay_committed(spec)
    assert opened.unknown_batches
    assert opened.unknown_batches[0]["unknown_fragment_indices"] == [0, 1, 2]
    assert opened.can_claim_exit_report is False
    assert opened.exit_report is None
    assert replay.checked_batches == 0


def test_committed_batch_replays_and_tamper_is_e_replay(tmp_path, spec):
    path = tmp_path / "session.db"
    bundle = build_bundle(spec)
    with SessionLedger(path) as ledger:
        ledger.open(spec)
        _local_commit(ledger, spec, bundle, bundle.messages[0])
    with SessionLedger(path) as ledger:
        opened = ledger.open(spec)
        replay = ledger.replay_committed(spec)
    assert opened.can_claim_exit_report is True
    assert replay.checked_batches == 1
    assert replay.checked_fragments == 3
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE fragments SET wire_sha256=? WHERE batch_id=? AND fragment_index=1", ("0" * 64, "tun-0"))
        connection.commit()
    with SessionLedger(path) as ledger:
        generation = int(ledger._get_meta("active_generation"))
        digest = ledger._compute_content_digest(generation)
        ledger.connection.execute("UPDATE meta SET value=? WHERE key='ledger_content_digest'", (digest,))
        ledger.connection.commit()
        opened = ledger.open(spec)
        assert opened.replaced is False
        with pytest.raises(TunnelError) as captured:
            ledger.replay_committed(spec)
    assert captured.value.code == E_REPLAY
    assert captured.value.path == "batches[tun-0].fragments[1].wire_sha256"
    assert captured.value.details["fragment_index"] == 1
    assert captured.value.details["field"] == "wire_sha256"


def test_snapshot_replacement_is_whole_and_lists_stable_diff(tmp_path, spec, spec_data):
    import copy

    path = tmp_path / "session.db"
    changed = copy.deepcopy(spec_data)
    changed["versions"]["declaration"] += 1
    changed["versions"]["endpoint_set"] += 1
    changed["versions"]["route_table"] += 1
    changed["endpoints"][0]["mtu"] -= 10
    changed["routes"][0]["priority"] += 1
    changed["routes"][1]["next_hop"] = changed["endpoints"][0]["id"]
    new_spec = load_spec_mapping(changed, source_path="<new>")
    bundle = build_bundle(spec)
    with SessionLedger(path) as ledger:
        ledger.open(spec)
        _local_commit(ledger, spec, bundle, bundle.messages[0])
    result = snapshot_replacement_report(old_spec=spec, new_spec=new_spec, ledger_path=path)
    assert result["ok"] is True
    assert result["replacement_mode"] == "whole_snapshot"
    diff = result["replacement"]["replacement_diff"]
    assert diff["endpoints"]["changed"] == [{"id": "ep-0", "fields": ["mtu"]}]
    assert diff["routes"]["changed"] == [
        {"id": "tun-0", "fields": ["priority"]},
        {"id": "tun-1", "fields": ["next_hop"]},
    ]
    with SessionLedger(path) as ledger:
        opened = ledger.open(new_spec)
        replay = ledger.replay_committed(new_spec)
    assert opened.replaced is False
    assert opened.generation == 2
    assert opened.can_claim_exit_report is True
    assert replay.checked_batches == 0


def test_ledger_digest_injection_is_e_snapshot(tmp_path, spec):
    path = tmp_path / "session.db"
    spec_path = write_spec(tmp_path / "spec.json", make_spec_data())
    bundle = build_bundle(spec)
    with SessionLedger(path) as ledger:
        ledger.open(spec)
        _local_commit(ledger, spec, bundle, bundle.messages[0])
    with pytest.raises(TunnelError) as captured:
        inject_anomaly(
            kind="snapshot-digest",
            spec_path=spec_path,
            output_dir=tmp_path / "inject",
            ledger_path=path,
        )
    assert captured.value.code == E_SNAPSHOT
    assert captured.value.path == "ledger.meta.content_digest"


def test_forced_process_exit_then_resume_reports_unknown(tmp_path):
    data = make_spec_data(route_count=2)
    spec_path = write_spec(tmp_path / "spec.json", data)
    ledger_path = tmp_path / "crash.db"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tunnel_interop",
            "forward",
            "--spec",
            str(spec_path),
            "--ledger",
            str(ledger_path),
            "--message",
            "tun-0",
            "--message",
            "tun-1",
            "--crash-after-batch",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 97
    spec = load_spec_mapping(data, source_path=str(spec_path))
    with SessionLedger(ledger_path) as ledger:
        opened = ledger.open(spec)
        replay = ledger.replay_committed(spec)
    assert replay.checked_batches == 1
    assert opened.unknown_batches
    assert opened.unknown_batches[0]["batch_id"] == "tun-1"
    assert opened.can_claim_exit_report is False
