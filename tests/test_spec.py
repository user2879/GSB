from __future__ import annotations

import copy

import pytest

from modeldash.errors import SpecError
from modeldash.spec import validate_spec


def invalid(raw, field_prefix):
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    assert exc.value.code == "E_SPEC"
    assert exc.value.field.startswith(field_prefix), exc.value.field


def test_valid_spec(raw_spec):
    spec = validate_spec(raw_spec)
    assert spec.spec_version == 1
    assert spec.current_version == 2
    assert [node.port for node in spec.nodes] == [node["port"] for node in raw_spec["nodes"]]


@pytest.mark.parametrize(
    ("mutator", "field"),
    [
        (lambda x: x["configs"][1].__setitem__("version", 1), "configs[1].version"),
        (lambda x: x["configs"][1].__setitem__("version", 0), "configs[1].version"),
        (lambda x: x["configs"][0]["items"].__setitem__("bad key", 1), "configs[0].items.bad key"),
        (lambda x: x["configs"][0]["items"].__setitem__("alpha", []), "configs[0].items.alpha"),
        (lambda x: x["configs"][0]["items"].__setitem__("alpha", float("nan")), "configs[0].items.alpha"),
        (lambda x: x["configs"][0].__setitem__("target_nodes", ["missing"]), "configs[0].target_nodes[0]"),
        (lambda x: x["nodes"][0].__setitem__("host", "localhost"), "nodes[0].host"),
        (lambda x: x["nodes"][1].__setitem__("port", x["nodes"][0]["port"]), "nodes[1].port"),
        (lambda x: x["nodes"][0].__setitem__("quota", 0), "nodes[0].quota"),
        (lambda x: x["configs"][0].__setitem__("shards", 0), "configs[0].shards"),
        (lambda x: x["tasks"][0].__setitem__("priority", -1), "tasks[0].priority"),
        (lambda x: x["tasks"][0].__setitem__("config_version", 99), "tasks[0].config_version"),
        (lambda x: x["tasks"][0].__setitem__("node", "missing"), "tasks[0].node"),
    ],
)
def test_invalid_specs(raw_spec, mutator, field):
    value = copy.deepcopy(raw_spec)
    mutator(value)
    invalid(value, field)


def test_node_must_be_a_config_target(raw_spec):
    value = copy.deepcopy(raw_spec)
    value["configs"][0]["target_nodes"] = ["n2", "n3"]
    value["tasks"][0]["node"] = "n1"
    invalid(value, "tasks[0].node")


def test_retry_limit_cannot_exceed_declared_sequence(raw_spec):
    value = copy.deepcopy(raw_spec)
    value["budget"]["max_retries"] = 3
    invalid(value, "budget.max_retries")
