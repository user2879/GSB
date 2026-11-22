from __future__ import annotations

import pytest

from driftwatch.config import parse_agent_manifest, parse_criteria, parse_snapshot_chain
from driftwatch.errors import DriftError

from .conftest import make_criteria


def _agents(port: int = 40101) -> dict:
    return {
        "agents": [
            {
                "agent_id": "a1",
                "host": "127.0.0.1",
                "port": port,
                "seed": 7,
                "aggregation": {"kind": "median", "window": 5, "version": "v1"},
                "hosts": ["host-a"],
            }
        ]
    }


def test_snapshot_validation_aggregates_violations(snapshot_payload):
    payload = snapshot_payload
    payload["snapshots"][0]["seq"] = 1
    payload["snapshots"][1]["seq"] = 1
    payload["snapshots"][0]["parent_seq"] = 9
    payload["snapshots"][1]["hosts"]["host-b"]["tolerance_bps"] = 10001
    with pytest.raises(DriftError) as caught:
        parse_snapshot_chain(payload)
    error = caught.value
    assert error.code == "E_SNAPSHOT"
    paths = {item["path"] for item in error.violations}
    assert "$.snapshots[1].seq" in paths
    assert any(path.endswith(".parent_seq") for path in paths)
    assert "$.snapshots[1].hosts.host-b.tolerance_bps" in paths


def test_snapshot_cycle_is_rejected(snapshot_payload):
    payload = snapshot_payload
    payload["snapshots"][0]["parent_seq"] = 2
    payload["snapshots"][1]["parent_seq"] = 1
    with pytest.raises(DriftError) as caught:
        parse_snapshot_chain(payload)
    assert caught.value.code == "E_SNAPSHOT"
    assert any("cycle" in item["reason"] for item in caught.value.violations)


def test_agent_validation_aggregates_and_rejects_duplicate_port(snapshot_payload):
    chain = parse_snapshot_chain(snapshot_payload)
    criteria = make_criteria()
    payload = _agents()
    second = dict(payload["agents"][0])
    second["agent_id"] = "a2"
    second["port"] = 40101
    second["host"] = "localhost"
    second["seed"] = True
    payload["agents"].append(second)
    with pytest.raises(DriftError) as caught:
        parse_agent_manifest(payload, chain, criteria)
    assert caught.value.code == "E_AGENT"
    reasons = {(item["path"], item["reason"]) for item in caught.value.violations}
    assert ("$.agents[1].port", "port must be globally unique") in reasons
    assert ("$.agents[1].host", "must be the literal loopback address 127.0.0.1") in reasons
    assert ("$.agents[1].seed", "must be an integer") in reasons


def test_agent_unknown_host_is_rejected(snapshot_payload):
    chain = parse_snapshot_chain(snapshot_payload)
    payload = _agents()
    payload["agents"][0]["hosts"] = ["unknown-host"]
    with pytest.raises(DriftError) as caught:
        parse_agent_manifest(payload, chain, make_criteria())
    assert caught.value.code == "E_AGENT"
    assert any("absent from the snapshot chain" in item["reason"] for item in caught.value.violations)


def make_criteria_tolerance_and_rule_validation():
    with pytest.raises(DriftError) as caught:
        parse_criteria({"scale_exp": 99, "boundary_rule": "unknown", "min_sources": 0})
    assert caught.value.code == "E_AGENT"
    assert len(caught.value.violations) >= 3
