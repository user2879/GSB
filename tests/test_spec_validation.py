from __future__ import annotations

import copy

import pytest

from lease_arbitrator.errors import SpecError
from lease_arbitrator.spec import validate_spec


def valid_spec() -> dict:
    return {
        "format_version": 1,
        "scale": 2,
        "priority_range": [0, 10],
        "global_budget": 10,
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": 23001,
                "quota": 4,
                "max_active": 2,
                "seed": 7,
                "quota_version": 1,
            },
            {
                "node_id": "n2",
                "host": "127.0.0.1",
                "port": 23002,
                "quota": 4,
                "max_active": 1,
                "seed": 9,
                "quota_version": 1,
            },
        ],
        "leases": [
            {
                "lease_id": "l1",
                "node_id": "n1",
                "units": 2,
                "priority": 5,
                "hold_ms": 100,
                "quota_version": 1,
            }
        ],
        "quota_versions": [
            {
                "quota_version": 1,
                "global_budget": 10,
                "nodes": {
                    "n1": {"quota": 4, "max_active": 2},
                    "n2": {"quota": 4, "max_active": 1},
                },
            }
        ],
    }


def test_valid_spec_passes() -> None:
    spec = validate_spec(valid_spec())
    assert [node["node_id"] for node in spec["nodes"]] == ["n1", "n2"]


@pytest.mark.parametrize(
    ("mutate", "path_fragment"),
    [
        (lambda s: s["nodes"][0].__setitem__("node_id", ""), "$.nodes[0].node_id"),
        (lambda s: s["nodes"][0].__setitem__("host", "localhost"), "$.nodes[0].host"),
        (lambda s: s["nodes"][1].__setitem__("port", 23001), "$.nodes[1].port"),
        (lambda s: s["nodes"][0].__setitem__("quota", -1), "$.nodes[0].quota"),
        (lambda s: s["nodes"][0].__setitem__("max_active", True), "$.nodes[0].max_active"),
        (lambda s: s["leases"][0].__setitem__("units", 0), "$.leases[0].units"),
        (lambda s: s["leases"][0].__setitem__("priority", 99), "$.leases[0].priority"),
        (
            lambda s: s["quota_versions"].append({"quota_version": 0}),
            "$.quota_versions[1].quota_version",
        ),
    ],
)
def test_invalid_spec_reports_first_path(mutate, path_fragment: str) -> None:
    raw = valid_spec()
    mutate(raw)
    with pytest.raises(SpecError) as exc_info:
        validate_spec(raw)
    assert exc_info.value.code == "E_SPEC"
    assert exc_info.value.path == path_fragment


def test_duplicate_lease_id_path() -> None:
    raw = valid_spec()
    raw["leases"].append(copy.deepcopy(raw["leases"][0]))
    with pytest.raises(SpecError) as exc_info:
        validate_spec(raw)
    assert exc_info.value.path == "$.leases[1].lease_id"
