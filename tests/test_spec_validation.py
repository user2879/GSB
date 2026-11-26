from __future__ import annotations

import copy
import json

import pytest

from waterlease.errors import SpecError
from waterlease.spec import load_spec, parse_spec
from tests.support import ROOT


def base_raw() -> dict:
    return json.loads(json.dumps(load_spec(ROOT / "spec").raw))


def test_valid_shipped_spec_loads() -> None:
    spec = load_spec(ROOT / "spec")
    assert len(spec.nodes) == 3
    assert len(spec.shards) == 7
    assert spec.spec_digest


def test_duplicate_node_id_reports_field_path() -> None:
    raw = base_raw()
    raw["nodes"][1]["node_id"] = raw["nodes"][0]["node_id"]
    with pytest.raises(SpecError) as exc:
        parse_spec(raw)
    assert exc.value.code == "E_SPEC"
    assert "$.nodes[1].node_id" in exc.value.path


def test_host_must_be_literal_loopback() -> None:
    raw = base_raw()
    raw["nodes"][0]["host"] = "localhost"
    with pytest.raises(SpecError) as exc:
        parse_spec(raw)
    assert exc.value.path == "$.nodes[0].host"


def test_duplicate_ports_are_rejected() -> None:
    raw = base_raw()
    raw["nodes"][1]["port"] = raw["nodes"][0]["port"]
    with pytest.raises(SpecError) as exc:
        parse_spec(raw)
    assert exc.value.path == "$.nodes[1].port"


@pytest.mark.parametrize("field", ["concurrency_quota", "max_leases"])
def test_quota_fields_must_be_non_negative(field: str) -> None:
    raw = base_raw()
    raw["nodes"][0][field] = -1
    with pytest.raises(SpecError) as exc:
        parse_spec(raw)
    assert exc.value.path == f"$.nodes[0].{field}"


def test_missing_dependency_reference_rejected() -> None:
    raw = base_raw()
    raw["shards"][2]["upstreams"] = ["missing_shard"]
    with pytest.raises(SpecError) as exc:
        parse_spec(raw)
    assert "upstreams" in exc.value.path


def test_dependency_cycle_rejected() -> None:
    raw = base_raw()
    by_id = {item["shard_id"]: item for item in raw["shards"]}
    by_id["s1_raw_a"]["upstreams"] = ["s3_dep_a"]
    with pytest.raises(SpecError) as exc:
        parse_spec(raw)
    assert exc.value.code == "E_SPEC"
    assert exc.value.details["cycle"]


def test_scale_exp_is_limited_to_three() -> None:
    raw = base_raw()
    raw["session"]["scale_exp"] = 4
    with pytest.raises(SpecError) as exc:
        parse_spec(raw)
    assert exc.value.path == "$.session.scale_exp"
