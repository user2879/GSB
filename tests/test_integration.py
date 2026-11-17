import json
import socket
import threading
import time

import pytest

from sharedbuf.arbiter import Request
from sharedbuf.errors import (E_FIELD, E_HANDSHAKE, E_NODE, E_PORT, E_VER,
                              PlatformError)
from sharedbuf.orchestrator import Orchestrator
from sharedbuf.platform import Platform


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_cfg(port0, port1):
    sectors = ["b0#0", "b0#1", "b1#0", "b1#1"]
    return {
        "seed": 99,
        "sectors": sectors,
        "sector_index": {s: i for i, s in enumerate(sectors)},
        "owners": {"b0#0": "n0", "b1#0": "n0", "b0#1": "n1", "b1#1": "n1"},
        "nodes": {"n0": {"port": port0, "spawn": True},
                  "n1": {"port": port1, "spawn": True}},
        "lease_ttl_ms": 5000,
        "global_limit": 4,
        "node_quota": 2,
        "adjudication_timeout_s": 2.0,
        "adjudication_retries": 1,
    }


@pytest.fixture
def running(tmp_path):
    cfg = make_cfg(free_port(), free_port())
    p = Platform(cfg, db_path=str(tmp_path / "i.db"))
    orch = Orchestrator(p, cfg["nodes"],
                        log_path=str(tmp_path / "platform.log"))
    orch.start()
    p.node_runner = orch.node_runner()
    yield p, orch, cfg
    orch.shutdown()
    p.close()


def test_end_to_end_read_write_over_tcp(running):
    p, orch, cfg = running
    r1 = p.submit_and_wait(Request("r1", "n0", "read", "sum", ("b0#0",),
                                   count=2), timeout=30)
    assert r1["value"] == r1["expected"]
    w1 = p.submit_and_wait(Request("w1", "n0", "write", "mean", ("b0#0", "b1#0"),
                                   delta=0.5, count=2), timeout=30)
    assert p.versions["b0#0"] == 2
    assert p.versions["b1#0"] == 2
    batches = p.store.list_batches()
    assert len(batches) == 2 and all(b["match"] for b in batches)
    assert batches[1]["lease_id"] == w1["lease_id"]
    assert batches[1]["versions"] == {"b0#0": 0, "b1#0": 0}


def test_node_rejects_malformed_message_over_wire(running):
    p, orch, cfg = running
    conn = orch.conns["n0"]
    bad = {"type": "op", "wl": p.wl, "lease_id": "LX", "node_id": "n0",
           "op": "read", "agg": "sum", "sectors": ["b0#0"], "seq_lo": 1,
           "seq_hi": 1, "delta": 0.0, "extra": 1}
    with conn.lock:
        conn.send(bad)
        resp = conn.recv(p.wl, 5.0)
    assert resp["type"] == "error"
    assert resp["code"] == E_FIELD
    assert resp["path"] == "$.extra"
    r = p.submit_and_wait(Request("ok", "n0", "read", "sum", ("b0#0",),
                                  count=1), timeout=30)
    assert r["value"] == r["expected"]


def test_corrupt_node_flagged_e_ver(tmp_path):
    cfg = make_cfg(free_port(), free_port())
    p = Platform(cfg, db_path=str(tmp_path / "c.db"))
    orch = Orchestrator(p, cfg["nodes"], corrupt_nodes={"n1"},
                        log_path=str(tmp_path / "platform.log"))
    try:
        orch.start()
        p.node_runner = orch.node_runner()
        with pytest.raises(PlatformError) as ei:
            p.submit_and_wait(Request("bad", "n1", "read", "sum", ("b0#1",),
                                      count=1), timeout=30)
        assert ei.value.code == E_VER
        assert ei.value.detail["node"] == "n1"
        assert ei.value.detail["stage"] == "value"
        assert "lease" in ei.value.detail
    finally:
        orch.shutdown()
        p.close()


def test_port_occupied_reports_e_port(tmp_path):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    cfg = make_cfg(port, free_port())
    p = Platform(cfg, db_path=str(tmp_path / "p.db"))
    log_path = tmp_path / "platform.log"
    orch = Orchestrator(p, cfg["nodes"], log_path=str(log_path))
    try:
        with pytest.raises(PlatformError) as ei:
            orch.start()
        assert ei.value.code == E_PORT
        assert ei.value.detail["port"] == port
        lines = log_path.read_text().strip().splitlines()
        rec = json.loads(lines[-1])
        assert rec["code"] == E_PORT and rec["result"] == "error"
    finally:
        blocker.close()
        p.close()


def test_handshake_timeout_reports_e_handshake(tmp_path):
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)
    port = silent.getsockname()[1]
    accepted = []

    def serve():
        conn, _ = silent.accept()
        accepted.append(conn)
        time.sleep(3)

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    cfg = make_cfg(port, free_port())
    cfg["nodes"] = {"n0": {"port": port, "spawn": False}}
    p = Platform(cfg, db_path=str(tmp_path / "h.db"))
    orch = Orchestrator(p, cfg["nodes"], handshake_timeout=0.5,
                        log_path=str(tmp_path / "platform.log"))
    try:
        with pytest.raises(PlatformError) as ei:
            orch.start()
        assert ei.value.code == E_HANDSHAKE
    finally:
        for c in accepted:
            c.close()
        silent.close()
        p.close()


def test_node_exit_reports_e_node(tmp_path):
    import sys
    cfg = make_cfg(free_port(), free_port())
    cfg["nodes"] = {"n0": {"port": cfg["nodes"]["n0"]["port"], "spawn": True,
                           "cmd": [sys.executable, "-c",
                                   "import sys; sys.exit(3)"]}}
    p = Platform(cfg, db_path=str(tmp_path / "n.db"))
    orch = Orchestrator(p, cfg["nodes"], handshake_timeout=2.0,
                        log_path=str(tmp_path / "platform.log"))
    try:
        with pytest.raises(PlatformError) as ei:
            orch.start()
        assert ei.value.code == E_NODE
    finally:
        p.close()
