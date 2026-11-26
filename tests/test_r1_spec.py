import json
import pytest

from ckpt_diag.errors import SpecError
from ckpt_diag.samplegen import build_spec_dict
from ckpt_diag.spec import load_spec


def test_valid_spec_loads(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(build_spec_dict()))
    spec = load_spec(path)
    assert len(spec.shards) == 3
    assert {shard.host for shard in spec.shards} == {"127.0.0.1"}


def _expect(tmp_path, payload, location_part, field_name):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SpecError) as exc:
        load_spec(path)
    assert exc.value.code == "E_SPEC"
    assert location_part in exc.value.location
    assert exc.value.field == field_name


def test_duplicate_shard_id(tmp_path):
    payload = build_spec_dict()
    payload["shards"][1]["shard_id"] = payload["shards"][0]["shard_id"]
    _expect(tmp_path, payload, "shards[1]", "shard_id")


def test_bad_host_literal(tmp_path):
    payload = build_spec_dict()
    payload["shards"][0]["host"] = "localhost"
    _expect(tmp_path, payload, "shards[0]", "host")


def test_duplicate_port(tmp_path):
    payload = build_spec_dict()
    payload["shards"][1]["port"] = payload["shards"][0]["port"]
    _expect(tmp_path, payload, "shards[1]", "port")


def test_negative_lease_cap(tmp_path):
    payload = build_spec_dict()
    payload["shards"][0]["lease_cap"] = -1
    _expect(tmp_path, payload, "shards[0]", "lease_cap")


def test_layer_parent_must_exist(tmp_path):
    payload = build_spec_dict()
    payload["shards"][0]["layers"][1]["parent"] = "NOPE"
    _expect(tmp_path, payload, "shards[0].layers[1]", "parent")


def test_layer_cycle(tmp_path):
    payload = build_spec_dict()
    payload["shards"][0]["layers"][0]["parent"] = "L2"
    _expect(tmp_path, payload, "shards[0].layers", "layers")


def test_generation_must_increase(tmp_path):
    payload = build_spec_dict()
    payload["checkpoints"][1]["generation"] = payload["checkpoints"][0]["generation"]
    _expect(tmp_path, payload, "checkpoints[1]", "generation")


def test_rollback_target_must_exist(tmp_path):
    payload = build_spec_dict()
    next(event for event in payload["events"]
         if event["kind"] == "rollback")["target_checkpoint_id"] = "cp-missing"
    _expect(tmp_path, payload, "events", "target_checkpoint_id")


def test_event_seq_strictly_increasing(tmp_path):
    payload = build_spec_dict()
    payload["events"][1]["seq"] = payload["events"][0]["seq"]
    _expect(tmp_path, payload, "events[1]", "seq")


def test_duplicate_checkpoint_id(tmp_path):
    payload = build_spec_dict()
    payload["checkpoints"][1]["checkpoint_id"] = payload["checkpoints"][0]["checkpoint_id"]
    _expect(tmp_path, payload, "checkpoints[1]", "checkpoint_id")


def test_unknown_event_kind(tmp_path):
    payload = build_spec_dict()
    payload["events"][0]["kind"] = "explode"
    _expect(tmp_path, payload, "events[0]", "kind")
