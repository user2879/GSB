from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from trace_arbiter.engine import build_scheduled_batches
from trace_arbiter.errors import ArbiterError
from trace_arbiter.pool import BatchScheduler, JsonLineClient, WorkerPool
from trace_arbiter.spec import load_dataset

from conftest import SAMPLE_SPANS, free_port, make_nodes, write_json


def _dataset(tmp_path: Path):
    nodes = write_json(tmp_path / "nodes.json", make_nodes())
    return load_dataset(str(nodes), str(SAMPLE_SPANS))


def test_service_same_batch_returns_identical_summary_bytes(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    with WorkerPool(dataset.nodes, handshake_timeout_ms=3000):
        finalize = build_scheduled_batches(dataset, batch_size=2)[-1]
        node = next(node for node in dataset.nodes if node.node_id == finalize.node_id)
        client = JsonLineClient(node, timeout_s=3, retries=0)
        first = client.request(finalize.request)
        second = client.request(finalize.request)
        assert first.response["decision_summary"].encode("utf-8") == second.response["decision_summary"].encode("utf-8")
        assert first.response["decision_digest"] == second.response["decision_digest"]
        assert isinstance(first.response["processing_us"], int)
        assert first.response["processing_us"] >= 0


def test_worker_port_error(tmp_path: Path) -> None:
    port = free_port()
    nodes = make_nodes()
    nodes["nodes"][0]["port"] = port
    nodes_path = write_json(tmp_path / "nodes.json", nodes)
    dataset = load_dataset(str(nodes_path), str(SAMPLE_SPANS))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        pool = WorkerPool(dataset.nodes[:1], handshake_timeout_ms=3000)
        with pytest.raises(ArbiterError) as exc:
            pool.start()
        assert exc.value.code == "E_PORT"
        pool.terminate()


def test_worker_spawn_and_handshake_errors(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    pool = WorkerPool(
        dataset.nodes[:1],
        handshake_timeout_ms=1000,
        python_executable=str(tmp_path / "does-not-exist-python"),
    )
    with pytest.raises(ArbiterError) as exc:
        pool.start()
    assert exc.value.code == "E_SPAWN"
    pool.terminate()

    slow_pool = WorkerPool(
        dataset.nodes[:1],
        handshake_timeout_ms=50,
        command_factory=lambda _node: [sys.executable, "-c", "import time; time.sleep(2)"],
    )
    with pytest.raises(ArbiterError) as exc:
        slow_pool.start()
    assert exc.value.code == "E_HANDSHAKE"
    slow_pool.terminate()


def test_socket_timeout_retries_identical_request_bytes() -> None:
    bodies: list[bytes] = []
    ready = threading.Event()
    port_holder: dict[str, int] = {}

    lock = threading.Lock()

    def handle_connection(conn: socket.socket, index: int) -> None:
        with conn:
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            with lock:
                bodies.append(data.split(b"\n", 1)[0])
            if index == 0:
                time.sleep(0.2)
            else:
                conn.sendall(b'{"ok":true,"decision_summary":"SAME","processing_us":1}\n')

    def server() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(2)
            port_holder["port"] = int(listener.getsockname()[1])
            ready.set()
            for index in range(2):
                conn, _ = listener.accept()
                threading.Thread(target=handle_connection, args=(conn, index), daemon=True).start()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    assert ready.wait(2)
    from trace_arbiter.spec import Node

    node = Node("fake", "127.0.0.1", port_holder["port"], 1, 10000, 1)
    client = JsonLineClient(node, timeout_s=0.05, retries=1)
    result = client.request({"batch": 1, "nested": {"k": 2}})
    thread.join(timeout=2)
    assert result.attempts == 2
    assert len(bodies) == 2
    assert bodies[0] == bodies[1]
    assert result.response["decision_summary"] == "SAME"


def test_scheduler_enforces_both_concurrency_limits(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    batches = build_scheduled_batches(dataset, batch_size=1)
    for batch in batches:
        batch.request["simulate_delay_ms"] = 60
        from trace_arbiter.canonical import canonical_text, digest_text

        batch.request_text = canonical_text(batch.request)
        batch.request_digest = digest_text(batch.request_text)
    with WorkerPool(dataset.nodes, handshake_timeout_ms=3000) as pool:
        scheduler = BatchScheduler(
            pool,
            global_concurrency=2,
            per_node_concurrency=1,
            timeout_ms=3000,
            retries=0,
        )
        outcomes, errors = scheduler.run(batches, abort_on_error=False)
        stats = scheduler.snapshot()
    assert not errors
    assert all(outcome.error is None for outcome in outcomes)
    assert stats.max_global_running <= 2
    assert all(value <= 1 for value in stats.max_node_running.values())
    assert any(wait > 0 for _index, _node, wait in stats.wait_intervals_us)
