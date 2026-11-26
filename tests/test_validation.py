from __future__ import annotations

import pytest

from reco_shard_diag.errors import BizErrorException
from reco_shard_diag.fixtures import build_demo_config
from reco_shard_diag.models import ProjectConfig
from reco_shard_diag.validation import validate_config


def _codes(raw):
    return [
        (error.code, error.object_id, error.field)
        for error in validate_config(ProjectConfig.from_raw(raw)).errors
    ]


def test_duplicate_node_and_port_detected(raw_copy):
    raw_copy["events"] = []
    raw_copy["nodes"][1]["node_id"] = raw_copy["nodes"][0]["node_id"]
    raw_copy["nodes"][1]["port"] = raw_copy["nodes"][0]["port"]
    codes = _codes(raw_copy)
    assert any(code == "E_NODE_ID_DUPLICATE" for code, _, _ in codes)
    assert any(code == "E_NODE_PORT_DUPLICATE" for code, _, _ in codes)


def test_invalid_capacity_and_lease_limit_are_located(raw_copy):
    raw_copy["events"] = []
    raw_copy["nodes"][0]["capacity_slots"] = 0
    raw_copy["nodes"][1]["lease_limit"] = -2
    with pytest.raises(BizErrorException) as exc_info:
        ProjectConfig.from_raw(raw_copy)
    assert exc_info.value.error.code == "E_NODE_CAPACITY_INVALID"
    assert exc_info.value.error.object_id == "node-a"
    assert exc_info.value.error.field == "capacity_slots"


def test_missing_quota_parent_is_located(raw_copy):
    raw_copy["events"] = []
    new_quota = dict(raw_copy["quota_versions"][-1])
    new_quota["version"] = 5
    new_quota["parent_version"] = 999
    raw_copy["quota_versions"].append(new_quota)
    codes = _codes(raw_copy)
    assert ("E_QUOTA_PARENT_MISSING", "5", "parent_version") in codes


def test_dependency_cycle_and_missing_dependency(raw_copy):
    raw_copy["events"] = []
    raw_copy["shards"][0]["upstream"] = ["shard-b"]
    raw_copy["shards"][1]["upstream"] = ["shard-a"]
    raw_copy["shards"][2]["upstream"] = ["missing-shard"]
    codes = _codes(raw_copy)
    assert any(code == "E_DEP_CYCLE" for code, _, _ in codes)
    assert ("E_DEP_MISSING", "shard-c", "upstream") in codes


def test_event_sequence_must_be_strict(raw_copy):
    raw_copy["events"][1]["seq"] = raw_copy["events"][0]["seq"]
    codes = _codes(raw_copy)
    assert any(code == "E_EVENT_SEQ_NOT_STRICT" for code, _, _ in codes)


def test_valid_demo_config():
    report = validate_config(ProjectConfig.from_raw(build_demo_config(include_events=True)))
    assert report.ok
