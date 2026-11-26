from __future__ import annotations

import json
import socket
from pathlib import Path

from lease_arbitrator.cli import main
from lease_arbitrator.replay import verify_ledger
from lease_arbitrator.spec import load_bundle, load_spec


ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def test_deterministic_demo_scheduling_generations_and_verification(tmp_path) -> None:
    spec = load_spec(ROOT / "samples" / "spec.json")
    for node in spec["nodes"]:
        node["port"] = free_port()

    spec_path = tmp_path / "spec.json"
    events_path = tmp_path / "events.ndjson"
    ledger_path = tmp_path / "demo.sqlite3"
    report_path = tmp_path / "report.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    events_path.write_text((ROOT / "samples" / "events.ndjson").read_text(encoding="utf-8"))

    exit_code = main(
        [
            "run",
            "--spec",
            str(spec_path),
            "--events",
            str(events_path),
            "--ledger",
            str(ledger_path),
            "--report",
            str(report_path),
            "--reset-ledger",
            "--settle-seconds",
            "1.0",
        ]
    )
    assert exit_code == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["verification"]["ok"] is True
    assert report["invariant_violations"] == []

    leases = {
        lease["lease_id"]: lease
        for lease in report["current_view"]["leases"] + report["stale_view"]["leases"]
    }
    assert leases["L-A-LONG"]["state"] == "released"
    assert leases["L-A-LONG"]["generation_current"] == 4
    assert leases["L-A-WAIT"]["state"] == "expired"
    assert leases["L-A-WAIT"]["generation_current"] == 2
    assert leases["L-A-WAIT"]["wait_ms"] is not None
    assert leases["L-A-WAIT"]["wait_ms"] >= 60
    assert leases["L-B-HOLD"]["state"] == "revoked"
    assert leases["L-C-HIGH"]["state"] == "revoked"
    assert leases["L-C-LOW"]["state"] == "revoked"
    assert leases["L-V2-NEW"]["state"] == "released"
    assert leases["L-V2-NEW"]["generation_current"] == 2

    boundary_decisions = [
        decision
        for decision in report["current_view"]["decisions"]
        if decision.get("error_affected") and decision.get("winner") == "L-C-HIGH"
    ]
    assert boundary_decisions
    decision = boundary_decisions[0]
    assert "L-C-HIGH" in decision["boundary_set"]
    assert "L-C-LOW" in decision["boundary_set"]
    assert decision["error_bound"]["total_bound_ulp"] >= 4
    assert decision["tie_rule"] == "quota_version_desc,priority_desc,lease_id_asc"

    spec_loaded, events_loaded = load_bundle(spec_path, events_path)
    verified = verify_ledger(spec_loaded, events_loaded, ledger_path)
    assert verified["ok"] is True
    assert "L-A-LONG" in verified["verified_leases"]
