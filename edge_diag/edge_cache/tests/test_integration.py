"""真实子进程 + 回环 TCP 的集成测试。"""

import json
import socket
import time

import pytest

from edge_cache import determinism
from edge_cache.clock import MonoClock
from edge_cache.engine import Engine, NodeProcess
from edge_cache.errors import PortError, TimeoutError
from edge_cache.protocol import LineConnection, connect_loopback
from edge_cache.replay import verify
from edge_cache.spec import spec_digest


def _write_spec(tmp_path, port=19201):
    spec = {
        "session": {"session_id": "it", "global_lease_limit": 2,
                    "global_byte_budget": 4096, "session_scale": 1000},
        "nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": port,
             "capacity_slots": 2, "per_node_lease_limit": 2,
             "node_seed": 101, "quota_version": 1}],
        "contents": [
            {"content_key": "c", "total_bytes": 512, "shard_count": 2,
             "content_seed": 9001}],
        "events": [
            {"seq": 1, "type": "write", "node_id": "n1",
             "content_key": "c", "shard_index": 0, "ttl_ms": 30000},
            {"seq": 2, "type": "read", "node_id": "n1",
             "content_key": "c", "shard_index": 0},
            {"seq": 3, "type": "renew", "node_id": "n1",
             "content_key": "c", "shard_index": 0, "ttl_ms": 30000},
            {"seq": 4, "type": "release", "node_id": "n1",
             "content_key": "c", "shard_index": 0}],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def test_node_real_tcp_handshake_and_digest(tmp_path):
    spec_path, spec = _write_spec(tmp_path)
    node = NodeProcess(spec["nodes"][0], spec["contents"], 1000)
    node.start(str(spec_path))
    try:
        ack = node.request({"type": "hello"})
        assert ack["type"] == "hello_ack"
        batch = node.request({
            "type": "batch", "batch_id": "b1",
            "items": [{"op": "write", "lease_id": "L",
                       "content_key": "c", "shard_index": 0,
                       "quota_version": 1}]})
        receipt = batch["receipts"][0]
        payload = determinism.generate_shard_bytes("c", 9001, 0, 256)
        expected = determinism.compute_digest(
            "n1", 101, "c", 0, payload, 1000)
        assert receipt["digest"] == expected["digest"]
        assert receipt["elapsed_ms"] >= 0
    finally:
        node.stop()


def test_port_conflict_is_E_PORT(tmp_path):
    spec_path, spec = _write_spec(tmp_path, port=19202)
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", 19202))
    holder.listen(1)
    node = NodeProcess(spec["nodes"][0], spec["contents"], 1000)
    try:
        with pytest.raises(PortError) as ei:
            node.start(str(spec_path))
        assert ei.value.code == "E_PORT"
        node.proc = None
    finally:
        holder.close()


def test_unreachable_port_raises_diag_error():
    # 不同沙箱对未监听端口可能返回拒绝(E_PORT)或丢包(E_TIMEOUT)，
    # 两者都必须映射为受控业务错误而非原生异常
    from edge_cache.errors import DiagError
    with pytest.raises(DiagError) as ei:
        connect_loopback("127.0.0.1", 19999, connect_timeout_ms=500)
    assert ei.value.code in ("E_PORT", "E_TIMEOUT")


def test_read_timeout_is_E_TIMEOUT(tmp_path):
    # 接受连接但不回数据 -> 真实套接字读超时
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 19203))
    server.listen(1)
    server.settimeout(2)
    sock = connect_loopback("127.0.0.1", 19203, connect_timeout_ms=1000)
    conn = LineConnection(sock, io_timeout_ms=300)
    conn.send_message({"type": "ping"})
    accepted, _ = server.accept()
    try:
        with pytest.raises(TimeoutError) as ei:
            conn.read_message()
        assert ei.value.code == "E_TIMEOUT"
    finally:
        accepted.close()
        server.close()
        conn.close()


def test_end_to_end_run_and_replay(tmp_path):
    spec_path, _ = _write_spec(tmp_path, port=19204)
    db = tmp_path / "ledger.sqlite3"
    engine = Engine(__import__("edge_cache.spec", fromlist=["load_spec"])
                    .load_spec(spec_path), spec_path, db, clock=MonoClock())
    try:
        engine.start_nodes()
        engine.run()
        report = engine.water_report()
        assert report["last_committed_seq"] == 4
    finally:
        engine.stop_nodes()
        engine.close()
    result = verify(spec_path, db)
    assert result["ok"] is True


def test_snapshot_mismatch_replaces_whole_ledger(tmp_path):
    spec_path, spec = _write_spec(tmp_path, port=19205)
    db = tmp_path / "ledger.sqlite3"
    from edge_cache.ledger import Ledger
    from edge_cache.spec import load_spec
    parsed = load_spec(spec_path)
    ledger = Ledger(db, parsed, spec_digest(parsed))
    ledger.close()
    # 用不同摘要重新打开 -> 整体替换
    ledger2 = Ledger(db, parsed, "0" * 64)
    assert ledger2.replaced is True
    assert ledger2.last_committed_seq() == 0
    ledger2.close()


def _crash_spec(tmp_path, base_port):
    spec = {
        "session": {"session_id": "crash", "global_lease_limit": 4,
                    "global_byte_budget": 8192, "session_scale": 1000},
        "nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": base_port,
             "capacity_slots": 3, "per_node_lease_limit": 2,
             "node_seed": 101, "quota_version": 1},
            {"node_id": "n2", "host": "127.0.0.1", "port": base_port + 1,
             "capacity_slots": 2, "per_node_lease_limit": 2,
             "node_seed": 202, "quota_version": 1}],
        "contents": [
            {"content_key": "c", "total_bytes": 1024, "shard_count": 4,
             "content_seed": 9001}],
        "events": [
            {"seq": 1, "type": "write", "node_id": "n1",
             "content_key": "c", "shard_index": 0, "ttl_ms": 30000},
            {"seq": 2, "type": "write", "node_id": "n2",
             "content_key": "c", "shard_index": 1, "ttl_ms": 30000},
            {"seq": 3, "type": "write", "node_id": "n1",
             "content_key": "c", "shard_index": 2, "ttl_ms": 30000},
            {"seq": 4, "type": "read", "node_id": "n1",
             "content_key": "c", "shard_index": 0},
            {"seq": 5, "type": "release", "node_id": "n2",
             "content_key": "c", "shard_index": 1},
            {"seq": 6, "type": "quota_change", "node_id": "n1",
             "new_quota_version": 2}],
    }
    path = tmp_path / "crash_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def test_force_kill_recovery_field_consistent(tmp_path):
    from edge_cache.spec import load_spec
    spec_path, spec = _crash_spec(tmp_path, 19301)
    db = tmp_path / "crash.sqlite3"
    parsed = load_spec(spec_path)

    first = Engine(parsed, spec_path, db, clock=MonoClock())
    first.start_nodes()
    first.run_until(2)
    first.force_kill_node("n2")
    first.stop_nodes()
    first.close()

    second = Engine(load_spec(spec_path), spec_path, db,
                    clock=MonoClock())
    second.start_nodes(skip={"n2"})
    last_seq, snap = second.recover()
    assert last_seq == 2
    assert len(snap["leases"]) == 2
    second.restart_node("n2")
    second.nodes["n2"].request({
        "type": "restore",
        "leases": [{"lease_id": l["lease_id"], "shard_id": l["shard_id"],
                    "quota_version": l["quota_version"],
                    "grant_ms": l["grant_ms"], "expire_ms": l["expire_ms"],
                    "renew_count": l["renew_count"],
                    "lease_version": l["lease_version"]}
                   for l in snap["leases"] if l["node_id"] == "n2"]})
    try:
        second.run(start_after_seq=last_seq)
    finally:
        second.stop_nodes()
        second.close()
    result = verify(spec_path, db)
    assert result["ok"] is True
    assert result["last_committed_seq"] == 6


def test_lease_optimistic_version_conflict(tmp_path):
    spec_path, spec = _write_spec(tmp_path, port=19206)
    node = NodeProcess(spec["nodes"][0], spec["contents"], 1000)
    node.start(str(spec_path))
    try:
        node.request({"type": "grant", "lease_id": "L1",
                      "content_key": "c", "shard_index": 0,
                      "quota_version": 1, "grant_ms": 1,
                      "expire_ms": 10_000_000_000, "renew_count": 0,
                      "lease_version": 1})
        # 正确版本续约成功
        ok = node.request({"type": "batch", "batch_id": "r1",
                           "items": [{"op": "renew", "lease_id": "L1",
                                      "expected_version": 1,
                                      "ttl_ms": 1000, "content_key": "c",
                                      "shard_index": 0}]})
        assert ok["receipts"][0]["ok"] is True
        assert ok["receipts"][0]["lease_version"] == 2
        # 旧版本续约 -> E_LEASE，且不延期（节点返回错误）
        bad = node.request({"type": "batch", "batch_id": "r2",
                            "items": [{"op": "renew", "lease_id": "L1",
                                       "expected_version": 1,
                                       "ttl_ms": 1000, "content_key": "c",
                                       "shard_index": 0}]})
        receipt = bad["receipts"][0]
        assert receipt["ok"] is False and receipt["code"] == "E_LEASE"
    finally:
        node.stop()
