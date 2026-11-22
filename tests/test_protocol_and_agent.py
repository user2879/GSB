from __future__ import annotations

import socket
import threading
import time

import pytest

from driftwatch.errors import DriftError
from driftwatch.models import CollectionTask
from driftwatch.processes import AgentProcessManager
from driftwatch.protocol import AgentClient, JsonLineChannel, handshake

from .conftest import free_port, make_agent


def test_agent_subprocess_is_deterministic_and_socket_backed():
    port = free_port()
    profile = {"host-a": {metric: {"base": 1000, "spread": 20} for metric in ("cpu_quota", "memory_pages", "processes")}}
    agent = make_agent("agent-one", port, profiles=profile)
    task = CollectionTask(agent, 3, "host-a", "cpu_quota", 301)
    with AgentProcessManager((agent,), handshake_timeout_ms=5000):
        client = AgentClient(agent, 2000, 2000)
        first = client.observe(task)
        second = client.observe(task)
    assert first.value == second.value
    assert first.agent_id == "agent-one"
    assert first.observation_seq == 301
    assert first.aggregation_version == "agg-v1"
    assert first.monotonic_ms >= 0


def test_occupied_port_reports_e_port():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    agent = make_agent("agent-one", port)
    try:
        with pytest.raises(DriftError) as caught:
            with AgentProcessManager((agent,), handshake_timeout_ms=500):
                pass
        assert caught.value.code == "E_PORT"
    finally:
        listener.close()


def test_handshake_timeout_is_real_and_not_an_observation():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    stop = threading.Event()

    def silent_server() -> None:
        connection, _ = listener.accept()
        try:
            stop.wait(1.0)
        finally:
            connection.close()

    thread = threading.Thread(target=silent_server, daemon=True)
    thread.start()
    agent = make_agent("agent-one", port)
    try:
        started = time.monotonic()
        with pytest.raises(DriftError) as caught:
            handshake(agent, 150, retry_until_deadline=False)
        elapsed = time.monotonic() - started
        assert caught.value.code == "E_TIMEOUT"
        assert elapsed >= 0.10
        assert "record" not in caught.value.details
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=2)


def test_missing_record_field_reports_e_record():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    ready = threading.Event()

    def malformed_server() -> None:
        ready.set()
        connection, _ = listener.accept()
        channel = JsonLineChannel(connection)
        try:
            hello = channel.receive(time.monotonic() + 2)
            assert hello["op"] == "hello"
            channel.send({"ok": True, "agent_id": "agent-one", "protocol": 1})
            request = channel.receive(time.monotonic() + 2)
            channel.send(
                {
                    "ok": True,
                    "record": {
                        "agent_id": "agent-one",
                        "observation_seq": request["observation_seq"],
                        "host_id": request["host_id"],
                        "metric_id": request["metric_id"],
                        "aggregation_version": "agg-v1",
                        "monotonic_ms": 1,
                    },
                }
            )
        finally:
            channel.close()
            listener.close()

    thread = threading.Thread(target=malformed_server, daemon=True)
    thread.start()
    ready.wait(1)
    agent = make_agent("agent-one", port)
    task = CollectionTask(agent, 1, "host-a", "cpu_quota", 101)
    with pytest.raises(DriftError) as caught:
        AgentClient(agent, 1000, 1000).observe(task)
    assert caught.value.code == "E_RECORD"
    assert any("value" in problem for problem in caught.value.details["problems"])
    thread.join(timeout=2)
