from __future__ import annotations

import json
import socket

from tcp_workbench.cli import main
from tcp_workbench.models import Event
from tcp_workbench.ui_model import build_ui_model, render_ui_text
from tests.conftest import SAMPLES


def _log_records(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_headless_ui_stable_render_order_and_snapshot_list(
    service_factory, sample_events
):
    service = service_factory()
    for raw in sample_events:
        service.ingest_event(Event.from_dict({**raw, "event_seq": 0}))
    snapshot_id = service.save_snapshot("ui")
    model = build_ui_model(service)
    assert [row["item"] for row in model["conclusion_rows"]] == model[
        "stable_item_order"
    ]
    assert model["header"]["connection_state"] == "CLOSED"
    assert model["header"]["rtt"]["scale"] == 8
    assert model["snapshots"][0]["snapshot_id"] == snapshot_id
    text = render_ui_text(model)
    assert "ITEM | VALUE | REV | BASIS | AFFECTED | VERSION | DEPENDS" in text
    assert "SNAPSHOTS" in text


def test_cli_script_injection_logs_e_script(tmp_path):
    log_path = tmp_path / "script-inject.jsonl"
    rc = main(
        [
            "--log",
            str(log_path),
            "inject",
            "script",
            "--config",
            str(SAMPLES / "session.json"),
            "--script",
            str(SAMPLES / "invalid" / "peer_script_corrupt.json"),
        ]
    )
    assert rc == 2
    records = _log_records(log_path)
    assert records[-1]["event"] == "command_failed"
    assert records[-1]["code"] == "E_SCRIPT"


def test_cli_port_injection_logs_e_port(tmp_path):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    log_path = tmp_path / "port-inject.jsonl"
    rc = main(
        [
            "--log",
            str(log_path),
            "inject",
            "port",
            "--config",
            str(SAMPLES / "session.json"),
            "--port",
            str(port),
        ]
    )
    assert rc == 2
    records = _log_records(log_path)
    assert records[-1]["code"] == "E_PORT"


def test_cli_version_injection_logs_e_version(service_factory, sample_events, tmp_path):
    service = service_factory("version.sqlite3")
    for raw in sample_events[:3]:
        service.ingest_event(Event.from_dict({**raw, "event_seq": 0}))
    snapshot_id = service.save_snapshot("version-inject")
    db_path = str(service.ledger.path)
    service.close()
    log_path = tmp_path / "version-inject.jsonl"
    rc = main(
        [
            "--log",
            str(log_path),
            "snapshot",
            "restore",
            "--db",
            db_path,
            "--snapshot-id",
            str(snapshot_id),
            "--inject-missing-version",
        ]
    )
    assert rc == 2
    records = _log_records(log_path)
    assert records[-1]["code"] == "E_VERSION"
