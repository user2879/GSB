from __future__ import annotations

import socket

from lease_arbitrator.errors import VerificationError
from lease_arbitrator.hashing import sha256_digest
from lease_arbitrator.ledger import Ledger
from lease_arbitrator.replay import verify_ledger


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def test_verify_locates_first_generation_mismatch(tmp_path) -> None:
    spec = {
        "format_version": 1,
        "scale": 0,
        "priority_range": [0, 10],
        "global_budget": 2,
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": free_port(),
                "quota": 2,
                "max_active": 1,
                "seed": 1,
                "quota_version": 1,
            }
        ],
        "leases": [
            {
                "lease_id": "L1",
                "node_id": "n1",
                "units": 1,
                "priority": 5,
                "hold_ms": 1000,
                "quota_version": 1,
            }
        ],
    }
    events = [
        {"seq": 1, "arrival_ms": 0, "type": "apply", "lease_id": "L1"},
        {"seq": 2, "arrival_ms": 10, "type": "release", "lease_id": "L1", "generation": 1},
    ]
    ledger_path = tmp_path / "divergent.sqlite3"
    ledger = Ledger(ledger_path)
    ledger.conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES('spec_digest',?)",
        (sha256_digest(spec),),
    )
    ledger.conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES('events_digest',?)",
        (sha256_digest(events),),
    )
    ledger.conn.commit()
    ledger.append(
        kind="grant",
        ts_monotonic_ms=1,
        payload={"wrong": "generation_after"},
        event_seq=1,
        lease_id="L1",
        node_id="n1",
        state_before="queued",
        state_after="granted",
        generation_used=1,
        generation_before=0,
        generation_after=99,
        global_used=1,
        global_remaining=1,
        node_used=1,
        node_remaining=1,
    )
    ledger.append(
        kind="release",
        ts_monotonic_ms=2,
        payload={},
        event_seq=2,
        lease_id="L1",
        node_id="n1",
        state_before="granted",
        state_after="released",
        generation_used=1,
        generation_before=1,
        generation_after=2,
        global_used=0,
        global_remaining=2,
        node_used=0,
        node_remaining=2,
    )
    ledger.close()

    try:
        verify_ledger(spec, events, ledger_path)
    except VerificationError as exc:
        assert exc.code == "E_VERIFY"
        assert exc.details["lease_id"] == "L1"
        assert exc.details["field"] == "generation_after"
        assert exc.details["generation"] == 99
    else:
        raise AssertionError("verification must fail for divergent generation")
