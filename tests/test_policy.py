from __future__ import annotations

import json
from pathlib import Path

import pytest

from qosfdiag.errors import PolicyError
from qosfdiag.policy import load_policy


def valid_policy() -> dict:
    return {
        "policy_version": "v1",
        "max_committed_rate_kbps": 10000,
        "shapers": [
            {
                "shaper_id": "edge",
                "committed_rate_kbps": 2000,
                "burst_bytes": 3000,
                "queue_limit_packets": 8,
                "match_dst_port_prefix": 530,
            },
            {
                "shaper_id": "core",
                "committed_rate_kbps": 1000,
                "burst_bytes": 2000,
                "queue_limit_packets": 4,
                "match_dst_port_prefix": 530,
            },
        ],
        "edges": [{"upstream": "edge", "downstream": "core"}],
    }


def test_valid_policy_and_stable_topology() -> None:
    policy = load_policy(valid_policy())
    assert policy.policy_version == "v1"
    assert policy.roots == ("edge",)
    assert policy.topological_order() == ("edge", "core")
    assert policy.parent_by_downstream == {"core": "edge"}


def test_policy_file_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(valid_policy()), encoding="utf-8")
    assert load_policy(path).policy_version == "v1"


def test_policy_rejects_all_errors_without_partial_load() -> None:
    broken = {
        "policy_version": "",
        "max_committed_rate_kbps": 1000,
        "shapers": [
            {
                "shaper_id": "dup",
                "committed_rate_kbps": 0,
                "burst_bytes": -1,
                "queue_limit_packets": -2,
                "match_dst_port_prefix": 0,
            },
            {
                "shaper_id": "dup",
                "committed_rate_kbps": 1001,
                "burst_bytes": 1,
                "queue_limit_packets": 1,
                "match_dst_port_prefix": 70000,
            },
            {
                "shaper_id": "b",
                "committed_rate_kbps": 10,
                "burst_bytes": 10,
                "queue_limit_packets": 1,
                "match_dst_port_prefix": 500,
            },
            {
                "shaper_id": "a",
                "committed_rate_kbps": 10,
                "burst_bytes": 10,
                "queue_limit_packets": 1,
                "match_dst_port_prefix": 501,
            },
        ],
        "edges": [
            {"upstream": "missing", "downstream": "b"},
            {"upstream": "a", "downstream": "b"},
            {"upstream": "b", "downstream": "a"},
            {"upstream": "b", "downstream": "a"},
        ],
    }
    with pytest.raises(PolicyError) as caught:
        load_policy(broken)
    error = caught.value
    assert error.code == "E_POLICY"
    fields = {(item.shaper_id, item.field) for item in error.violations}
    assert ("<policy>", "policy_version") in fields
    assert ("dup", "committed_rate_kbps") in fields
    assert ("dup", "shaper_id") in fields
    assert ("missing", "cascade") in fields
    assert any(item.field == "cascade" and "cycle detected" in item.message for item in error.violations)
    ids = [(item.shaper_id, item.field, item.message) for item in error.violations]
    assert ids == sorted(ids)


def test_multiple_upstreams_rejected() -> None:
    broken = valid_policy()
    broken["shapers"].append(
        {
            "shaper_id": "other",
            "committed_rate_kbps": 100,
            "burst_bytes": 100,
            "queue_limit_packets": 0,
            "match_dst_port_prefix": 1,
        }
    )
    broken["edges"].append({"upstream": "other", "downstream": "core"})
    with pytest.raises(PolicyError) as caught:
        load_policy(broken)
    assert any(
        item.shaper_id == "core"
        and item.field == "cascade"
        and "at most one upstream" in item.message
        for item in caught.value.violations
    )


def test_bool_is_not_accepted_as_integer() -> None:
    broken = valid_policy()
    broken["shapers"][0]["committed_rate_kbps"] = True
    with pytest.raises(PolicyError):
        load_policy(broken)
