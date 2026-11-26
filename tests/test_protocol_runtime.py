from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
import socket
import threading
import time
import pytest

from gsb.client import NodeClient, NodeProcess, execute_shard
from gsb.errors import GSBTimeoutError, PortError, ProtocolError
from gsb.model import NodeSpec
from gsb.protocol import JsonLineChannel
from gsb.runtime import ClusterRuntime
from gsb.samples import generate_samples
from gsb.spec import parse_spec
from helpers import base_raw_spec, free_port


def _start_fake_server(port: int, reply: bytes | None, hold_s: float = 0.5) -> threading.Thread:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(2)

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with closing(connection):
                channel = JsonLineChannel(connection)
                try:
                    channel.receive(2.0, stage="fake_request")
                except Exception:
                    return
                if reply is not None:
                    connection.sendall(reply)
                else:
                    time.sleep(hold_s)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread


def _fake_node(port: int) -> NodeSpec:
    return NodeSpec(
        id="fake",
        host="127.0.0.1",
        port=port,
        seed=1,
        concurrency=1,
        lease_limit=1,
        quota=1,
        used=0,
        fixed_scale=1000,
    )


def test_batch_slicing_and_concurrency_preserve_digest(tmp_path) -> None:
    spec = parse_spec(base_raw_spec())
    node_a = spec.node_map()["node-a"]
    node_b = spec.node_map()["node-b"]
    shard_a = spec.shard_map()["raw-a"]
    shard_b = spec.shard_map()["raw-b"]
    rows_a = generate_samples(node_a, shard_a)
    rows_b = generate_samples(node_b, shard_b)
    with NodeProcess(node_a, startup_timeout_s=2) as process_a, NodeProcess(
        node_b, startup_timeout_s=2
    ):
        assert process_a.process is not None and process_a.process.poll() is None
        first = execute_shard(node_a, shard_a, rows_a, spec.capacity_version, read_timeout_s=2)
        second = execute_shard(
            node_a, shard_a, rows_a, spec.capacity_version, read_timeout_s=2, batch_size=1
        )
        third = execute_shard(
            node_a, shard_a, rows_a, spec.capacity_version, read_timeout_s=2, batch_size=5
        )
        assert first["aggregate_digest"] == second["aggregate_digest"] == third["aggregate_digest"]
        assert first["sample_digest"] == second["sample_digest"] == third["sample_digest"]

        def run(node, shard, rows) -> str:
            return execute_shard(
                node, shard, rows, spec.capacity_version, read_timeout_s=2
            )["aggregate_digest"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = list(pool.map(lambda args: run(*args), [(node_a, shard_a, rows_a), (node_b, shard_b, rows_b)]))
        assert concurrent[0] == first["aggregate_digest"]


def test_cluster_batch_override_topology_and_wait_log(tmp_path) -> None:
    raw = base_raw_spec(global_budget=6, max_workers=3)
    spec = parse_spec(raw)
    from gsb.ledger import Ledger

    ledger = Ledger(tmp_path / "run.db", session_version=1)
    with ClusterRuntime(spec, ledger=ledger, session_version=1, batch_size_overrides={"raw-a": 1}) as runtime:
        result = runtime.execute()
    assert tuple(result.records) == spec.topological_order()
    assert any(event["shard_id"] == "raw-d" for event in result.waits)
    assert any("global_budget" in event["reasons"] for event in result.waits)
    assert result.lease_stats["global_units"] == 0
    ledger.close()


def test_parallelism_and_batch_slicing_do_not_change_aggregate_digest(tmp_path) -> None:
    from gsb.ledger import Ledger

    spec = parse_spec(base_raw_spec(global_budget=6, max_workers=3))
    baseline_ledger = Ledger(tmp_path / "parallel.db", session_version=1)
    with ClusterRuntime(spec, ledger=baseline_ledger, session_version=1) as runtime:
        baseline = runtime.execute()
    baseline_ledger.close()

    serial_spec = replace(spec, max_workers=1)
    overrides = {shard.id: 1 for shard in serial_spec.shards}
    serial_ledger = Ledger(tmp_path / "serial.db", session_version=1)
    with ClusterRuntime(
        serial_spec,
        ledger=serial_ledger,
        session_version=1,
        batch_size_overrides=overrides,
    ) as runtime:
        serial = runtime.execute()
    serial_ledger.close()

    for shard_id in spec.topological_order():
        assert serial.records[shard_id]["aggregate_digest"] == baseline.records[shard_id]["aggregate_digest"]
        assert serial.records[shard_id]["sample_digest"] == baseline.records[shard_id]["sample_digest"]
        assert serial.records[shard_id]["participating_rows"] == baseline.records[shard_id]["participating_rows"]


def test_port_occupied_reports_e_port() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    try:
        with pytest.raises(PortError) as captured:
            NodeProcess(_fake_node(port), startup_timeout_s=2).start()
        assert captured.value.code == "E_PORT"
    finally:
        listener.close()


def test_handshake_timeout_reports_e_timeout() -> None:
    port = free_port()
    thread = _start_fake_server(port, None, hold_s=0.5)
    with pytest.raises(GSBTimeoutError) as captured:
        NodeClient(_fake_node(port), read_timeout_s=0.1).handshake()
    assert captured.value.code == "E_TIMEOUT"
    thread.join(timeout=2)


def test_missing_reply_fields_reports_e_proto() -> None:
    port = free_port()
    thread = _start_fake_server(port, b'{"ok":true}\n')
    with pytest.raises(ProtocolError) as captured:
        NodeClient(_fake_node(port), read_timeout_s=0.5).handshake()
    assert captured.value.code == "E_PROTO"
    thread.join(timeout=2)
