"""规格强校验 E_SPEC 测试。"""

import pytest

from edge_cache.errors import SpecError
from edge_cache.spec import validate_spec


def _base():
    return {
        "session": {"session_id": "s", "global_lease_limit": 2,
                    "global_byte_budget": 1024, "session_scale": 1000},
        "nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": 19101,
             "capacity_slots": 2, "per_node_lease_limit": 2,
             "node_seed": 1, "quota_version": 1}],
        "contents": [
            {"content_key": "c", "total_bytes": 512, "shard_count": 2,
             "content_seed": 9}],
        "events": [
            {"seq": 1, "type": "write", "node_id": "n1",
             "content_key": "c", "shard_index": 0, "ttl_ms": 100}],
    }


def test_valid_spec():
    spec = validate_spec(_base())
    assert spec["format_version"] == 1
    assert spec["contents"][0]["shard_byte_size"] == 256


def test_duplicate_node_id():
    d = _base()
    d["nodes"].append(dict(d["nodes"][0], port=19102))
    with pytest.raises(SpecError) as ei:
        validate_spec(d)
    assert ei.value.code == "E_SPEC"
    assert "node_id" in ei.value.location


def test_host_must_be_loopback_literal():
    d = _base()
    d["nodes"][0]["host"] = "10.0.0.1"
    with pytest.raises(SpecError) as ei:
        validate_spec(d)
    assert ei.value.location.endswith(".host")


def test_duplicate_port():
    d = _base()
    d["nodes"].append(dict(d["nodes"][0], node_id="n2"))
    with pytest.raises(SpecError) as ei:
        validate_spec(d)
    assert "port" in ei.value.location


def test_shard_count_must_divide_total():
    d = _base()
    d["contents"][0]["shard_count"] = 3
    with pytest.raises(SpecError) as ei:
        validate_spec(d)
    assert "shard_count" in ei.value.location


def test_event_seq_must_increase():
    d = _base()
    d["events"].append(dict(d["events"][0]))
    with pytest.raises(SpecError) as ei:
        validate_spec(d)
    assert "event#1" in ei.value.location


def test_event_unknown_node_reference():
    d = _base()
    d["events"][0]["node_id"] = "ghost"
    with pytest.raises(SpecError) as ei:
        validate_spec(d)
    assert "node_id" in ei.value.location


def test_shard_index_out_of_range():
    d = _base()
    d["events"][0]["shard_index"] = 9
    with pytest.raises(SpecError) as ei:
        validate_spec(d)
    assert "shard_index" in ei.value.location


def test_positive_integer_enforcement():
    d = _base()
    d["nodes"][0]["capacity_slots"] = 0
    with pytest.raises(SpecError) as ei:
        validate_spec(d)
    assert "capacity_slots" in ei.value.location
