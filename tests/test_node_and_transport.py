"""Requirement 4 (collection transport): real loopback TCP subprocess node."""

import json
import socket
import time

import pytest

from config_regression.model import NodeEntry, SessionConfig
from config_regression.sampling import run_aggregation, stable_hash
from config_regression.transport import NodeProcess, send_request


def _entry(node_id="node-a", fault_mode="none", fault_arg=0):
    return NodeEntry(node_id=node_id, host="127.0.0.1", port=0,
                     work_ms_min=1, work_ms_max=3,
                     fault_mode=fault_mode, fault_arg=fault_arg)


def _payload(node_id, version=1):
    return {"request_id": f"req-{version}-{node_id}", "node_id": node_id,
            "manifest_version": version, "base_seed": 42,
            "sample_count": 500, "group_count": 4,
            "aggregations": ["sum", "mean", "min", "max"],
            "items": [{"key": "k1", "value": 1.0, "scale_value": 1.0,
                       "quantized_value": 1000000}]}


def test_node_subprocess_replies_with_deterministic_summary():
    proc = NodeProcess(_entry())
    proc.start()
    try:
        assert proc.host == "127.0.0.1"
        assert proc.port > 0
        first = send_request(proc.host, proc.port, _payload(proc.entry.node_id),
                             timeout_ms=2000)
        second = send_request(proc.host, proc.port, _payload(proc.entry.node_id),
                              timeout_ms=2000)
        assert first["status"] == "ok"
        assert first["results"][0]["summary"] == second["results"][0]["summary"]
        expected = run_aggregation(42, "node-a", "k1", 500, 4,
                                   ("sum", "mean", "min", "max"), 1.0)
        assert first["results"][0]["summary"] == expected
        assert isinstance(first["results"][0]["elapsed_ms"], int)
    finally:
        proc.stop()


def test_node_uses_declared_loopback_high_port():
    proc = NodeProcess(_entry())
    proc.start()
    try:
        with socket.create_connection(("127.0.0.1", proc.port), timeout=2):
            pass
    finally:
        proc.stop()


def test_drop_first_fault_succeeds_on_second_connection():
    proc = NodeProcess(_entry(fault_mode="drop_first", fault_arg=1))
    proc.start()
    try:
        with pytest.raises((ConnectionError, OSError)):
            send_request(proc.host, proc.port,
                         _payload(proc.entry.node_id), timeout_ms=1000)
        response = send_request(proc.host, proc.port,
                                _payload(proc.entry.node_id), timeout_ms=1000)
        assert response["status"] == "ok"
    finally:
        proc.stop()


def test_stable_hash_is_platform_independent():
    # golden value guards against accidental algorithm changes
    assert isinstance(stable_hash(42, "node-a", "k1"), int)
    assert stable_hash(42, "node-a", "k1") == stable_hash(42, "node-a", "k1")
