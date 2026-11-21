from __future__ import annotations

import copy
import socket

import pytest

from modeldash.errors import DispatchError
from modeldash.nodes import NodeSupervisor, SocketDispatcher, request_id_for
from modeldash.spec import validate_spec

from conftest import free_port
from test_scheduler import minimal_raw


def one_node_raw(*, timeout_ms=1000):
    raw = minimal_raw(nodes=1, shards=2)
    raw["budget"]["timeout_ms"] = timeout_ms
    raw["configs"].append(
        {
            "version": 2,
            "priority": 2,
            "target_nodes": ["n1"],
            "shards": 2,
            "items": {"alpha": 11, "beta": 12, "gamma": 13},
        }
    )
    raw["tasks"].append({"id": "t2", "config_version": 2, "node": "n1", "priority": 2})
    return raw


def test_node_protocol_is_deterministic_and_version_bound(tmp_path):
    raw = one_node_raw()
    raw["configs"][0]["items"] = {"alpha": 1, "beta": 2, "gamma": 3}
    spec = validate_spec(raw)
    supervisor = NodeSupervisor(spec, tmp_path)
    try:
        supervisor.start_all()
        dispatcher = SocketDispatcher(spec)
        t1 = spec.task("t1")
        t2 = spec.task("t2")
        rid = request_id_for("session", t1, 0)
        first = dispatcher.dispatch(t1, 0, rid)
        second = dispatcher.dispatch(t1, 0, rid)
        other = dispatcher.dispatch(t2, 0, request_id_for("session", t2, 0))
        assert first.output_summary == second.output_summary
        assert first.input_digest == second.input_digest
        assert first.output_summary != other.output_summary
        assert first.compute_ms >= 0
        assert first.items_used == sorted(first.items_used)
    finally:
        supervisor.stop_all()


def test_injected_node_error_is_retryable_with_same_request_id(tmp_path):
    raw = one_node_raw()
    raw["nodes"][0]["faults"] = [{"shard_index": 0, "remaining": 1, "mode": "error_reply"}]
    spec = validate_spec(raw)
    supervisor = NodeSupervisor(spec, tmp_path)
    try:
        supervisor.start_all()
        dispatcher = SocketDispatcher(spec)
        task = spec.task("t1")
        rid = request_id_for("session", task, 0)
        with pytest.raises(DispatchError) as exc:
            dispatcher.dispatch(task, 0, rid)
        assert exc.value.code == "E_INJECTED"
        result = dispatcher.dispatch(task, 0, rid)
        assert result.output_summary
    finally:
        supervisor.stop_all()


def test_real_socket_timeout_is_reported(tmp_path):
    raw = one_node_raw(timeout_ms=100)
    raw["nodes"][0]["faults"] = [{"shard_index": 0, "remaining": 1, "mode": "drop_before_reply"}]
    spec = validate_spec(raw)
    supervisor = NodeSupervisor(spec, tmp_path)
    try:
        supervisor.start_all()
        with pytest.raises(DispatchError) as exc:
            SocketDispatcher(spec).dispatch(spec.task("t1"), 0, "req-timeout")
        assert exc.value.code == "E_TIMEOUT"
        assert exc.value.retryable is True
    finally:
        supervisor.stop_all()


def test_occupied_port_reports_e_port_in_use(tmp_path):
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = free_port()
    holder.bind(("127.0.0.1", port))
    holder.listen(1)
    raw = minimal_raw(nodes=1, shards=1, backoff=[1], max_retries=0)
    raw["nodes"][0]["port"] = port
    spec = validate_spec(raw)
    try:
        with pytest.raises(DispatchError) as exc:
            NodeSupervisor(spec, tmp_path).start_all()
        assert exc.value.code == "E_PORT_IN_USE"
        assert exc.value.node_id == "n1"
    finally:
        holder.close()


def test_handshake_failure_is_reported_separately(tmp_path):
    raw = minimal_raw(nodes=1, shards=1, backoff=[1], max_retries=0)
    raw["nodes"][0]["faults"] = [{"shard_index": 0, "remaining": 100, "mode": "drop_before_handshake"}]
    spec = validate_spec(raw)
    with pytest.raises(DispatchError) as exc:
        NodeSupervisor(spec, tmp_path).start_all()
    assert exc.value.code == "E_HANDSHAKE"
    assert exc.value.node_id == "n1"
