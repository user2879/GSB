from __future__ import annotations

import pytest

from gsb.errors import DependencyError, SpecError
from gsb.spec import parse_spec
from helpers import base_raw_spec


def test_valid_spec_and_stable_topology() -> None:
    spec = parse_spec(base_raw_spec())
    assert spec.topological_order() == ("raw-a", "raw-b", "join-ab", "raw-d", "report")
    assert spec.node_map()["node-c"].fixed_scale == 10_000


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (lambda raw: raw["nodes"][0].__setitem__("host", "localhost"), "$.nodes[0].host"),
        (lambda raw: raw["nodes"][1].__setitem__("port", raw["nodes"][0]["port"]), "$.nodes[1].port"),
        (lambda raw: raw["nodes"][0].__setitem__("concurrency", True), "$.nodes[0].concurrency"),
        (lambda raw: raw.__setitem__("fixed_scale", 999), "$.fixed_scale"),
        (lambda raw: raw["shards"][1].__setitem__("id", "raw-a"), "$.shards[1].id"),
        (lambda raw: raw["nodes"][0].__setitem__("unknown", 1), "$.nodes[0].unknown"),
    ],
)
def test_first_spec_violation(mutate, path: str) -> None:
    raw = base_raw_spec()
    mutate(raw)
    with pytest.raises(SpecError) as captured:
        parse_spec(raw)
    assert captured.value.code == "E_SPEC"
    assert captured.value.path == path


def test_missing_dependency_is_e_depend() -> None:
    raw = base_raw_spec()
    raw["shards"][1]["dependencies"] = ["does-not-exist"]
    with pytest.raises(DependencyError) as captured:
        parse_spec(raw)
    assert captured.value.code == "E_DEPEND"
    assert captured.value.path == "shards.raw-b.dependencies[0]"


def test_cycle_is_e_depend_with_cycle_path() -> None:
    raw = base_raw_spec()
    raw["shards"][0]["dependencies"] = ["report"]
    with pytest.raises(DependencyError) as captured:
        parse_spec(raw)
    assert captured.value.code == "E_DEPEND"
    assert captured.value.details["cycle"][0] == captured.value.details["cycle"][-1]


def test_unknown_node_reference_is_dependency_error() -> None:
    raw = base_raw_spec()
    raw["shards"][0]["node_id"] = "missing-node"
    with pytest.raises(DependencyError) as captured:
        parse_spec(raw)
    assert captured.value.code == "E_DEPEND"
    assert captured.value.path == "$.shards[0].node_id"
