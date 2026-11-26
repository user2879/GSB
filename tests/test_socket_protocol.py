import json
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from dispatch_tokens.client import NodeClient
from dispatch_tokens.errors import (FieldError, PortError, TimeoutError)
from dispatch_tokens.protocol import (DISPATCH, HANDSHAKE_REQUEST,
                                      decode_frame, encode_frame)
from dispatch_tokens.sample import generate_batch
from dispatch_tokens.spec import NodeSpec, load_spec
from dispatch_tokens.supervisor import Supervisor

SRC = Path(__file__).resolve().parents[1] / "src"
SPEC_PATH = Path(__file__).resolve().parents[1] / "samples" / "demo_spec.json"


def _free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


@pytest.fixture()
def worker():
    spec = load_spec(SPEC_PATH)
    node = spec.node("node-a")
    port = _free_port()
    node = NodeSpec(node.node_id, "127.0.0.1", port, node.rate, node.burst,
                    node.bucket_key, node.seed, node.param_version)
    process = subprocess.Popen(
        [sys.executable, "-m", "dispatch_tokens.child",
         json.dumps(asdict(node), sort_keys=True), "100"],
        stderr=subprocess.PIPE, cwd=str(SRC))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((node.host, node.port), 0.2):
                break
        except OSError:
            time.sleep(0.02)
    yield node
    process.kill()
    process.wait()


def test_real_handshake_dispatch_and_receipt(worker):
    client = NodeClient(worker)
    client.connect()
    try:
        spec = load_spec(SPEC_PATH)
        shard = spec.shard("sh-0")
        batch = generate_batch(worker.seed, "sh-0", shard.sample_rows, 1)
        receipt = client.dispatch("sh-0", 1, batch, "tok-test", 1)
        for field_name in ("node_id", "shard_id", "param_version",
                           "summary", "elapsed_ms"):
            assert field_name in receipt
        assert receipt["node_id"] == worker.node_id
        assert isinstance(receipt["elapsed_ms"], int)
    finally:
        client.close()


def test_port_conflict_raises_e_port(worker):
    # worker 已在该端口真实监听；supervisor 预检探测到连接可建立，
    # 认定端口被占用，直接抛 E_PORT 而不再 spawn 第二个进程。
    with socket.create_connection((worker.host, worker.port), timeout=1):
        pass
    with pytest.raises(PortError) as exc:
        Supervisor([worker], 100).start_all()
    assert exc.value.code == "E_PORT"


def test_connect_refused_raises_e_port():
    node = NodeSpec("ghost", "127.0.0.1", _free_port(), 1, 1, 1, 1, 1)
    client = NodeClient(node, connect_timeout=0.5)
    with pytest.raises(PortError) as exc:
        client.connect()
    assert exc.value.code == "E_PORT"


def test_read_timeout_raises_e_timeout(worker):
    # 客户端完成握手后保持静默，真实套接字读超时（非线程超时）。
    client = NodeClient(worker, connect_timeout=1.0, read_timeout=0.2)
    client.connect()
    with pytest.raises(TimeoutError) as exc:
        client._read_line()
    assert exc.value.code == "E_TIMEOUT"
    client.close()


def test_missing_frame_field_returns_e_field_error(worker):
    raw = socket.create_connection((worker.host, worker.port), timeout=2)
    raw.settimeout(1.0)
    raw.sendall(encode_frame({"type": HANDSHAKE_REQUEST,
                              "node_id": worker.node_id}))
    decode_frame(raw.recv(65536))
    bad_frame = {"type": DISPATCH, "shard_id": "sh-0"}
    raw.sendall(encode_frame(bad_frame))
    reply = decode_frame(raw.recv(65536))
    assert reply["type"] == "DT_ERROR"
    assert reply["code"] == "E_FIELD"
    assert reply["field"] == "param_version"
    raw.close()


def test_client_missing_receipt_field_raises_e_field(worker, monkeypatch):
    client = NodeClient(worker)
    client.connect()
    try:
        # 篡改客户端读取结果，模拟真实回执缺字段。
        def fake_read_line():
            return {"type": "DT_RECEIPT", "node_id": worker.node_id}
        monkeypatch.setattr(client, "_read_line", fake_read_line)
        with pytest.raises(FieldError) as exc:
            client.dispatch("sh-0", 1, {"groups": [], "values": []},
                            "tok", 1)
        assert exc.value.code == "E_FIELD"
    finally:
        client.close()
