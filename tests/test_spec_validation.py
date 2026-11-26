import pytest
from pathlib import Path

from dispatch_tokens.errors import SpecError
from dispatch_tokens.spec import load_spec, parse_spec

BASE = {
    "nodes": [
        {"node_id": "n1", "host": "127.0.0.1", "port": 49201, "rate": 5,
         "burst": 4, "bucket_key": 1, "seed": 1, "param_version": 1},
    ],
    "shards": [
        {"shard_id": "s1", "upstream": [], "sample_rows": 10, "weight": 1,
         "priority": 0},
    ],
    "revisions": [
        {"version": 1, "parent": None, "global_rate": None,
         "global_burst": None, "node_rates": {}, "node_bursts": {},
         "shard_deps": {}, "scale": 100},
    ],
    "events": [],
}


def test_valid_spec_loads():
    spec = parse_spec(BASE)
    assert spec.nodes[0].node_id == "n1"
    assert spec.revisions[0].scale == 100


@pytest.mark.parametrize("mutator,location,field", [
    (lambda d: d["nodes"][0].__setitem__("host", "0.0.0.0"),
     "nodes[0]", "host"),
    (lambda d: d["nodes"][0].__setitem__("port", 0), "nodes[0]", "port"),
    (lambda d: d["nodes"][0].__setitem__("rate", -1), "nodes[0]", "rate"),
    (lambda d: d["nodes"][0].__setitem__("rate", 1.5), "nodes[0]", "rate"),
    (lambda d: d["nodes"].append(dict(d["nodes"][0], port=49202)),
     "nodes[1]", "node_id"),
    (lambda d: d["nodes"].append(dict(d["nodes"][0], node_id="n2")),
     "nodes[1]", "port"),
    (lambda d: d["shards"][0].__setitem__("upstream", ["ghost"]),
     "shards[s1]", "upstream"),
    (lambda d: d["revisions"][0].__setitem__("scale", 50),
     "revisions[0]", "scale"),
])
def test_first_violation_reported(mutator, location, field):
    data = {"nodes": [dict(n) for n in BASE["nodes"]],
            "shards": [dict(s) for s in BASE["shards"]],
            "revisions": [dict(r) for r in BASE["revisions"]],
            "events": []}
    mutator(data)
    with pytest.raises(SpecError) as exc:
        parse_spec(data)
    assert exc.value.code == "E_SPEC"
    assert exc.value.location == location
    assert exc.value.field == field


def test_revision_parent_must_exist():
    data = {"nodes": [dict(n) for n in BASE["nodes"]],
            "shards": [dict(s) for s in BASE["shards"]],
            "revisions": [
                dict(BASE["revisions"][0]),
                {"version": 2, "parent": 99, "global_rate": None,
                 "global_burst": None, "node_rates": {}, "node_bursts": {},
                 "shard_deps": {}, "scale": None},
            ], "events": []}
    with pytest.raises(SpecError) as exc:
        parse_spec(data)
    assert exc.value.location == "revisions[1]"
    assert exc.value.field == "parent"


def test_revision_version_monotonic():
    data = {"nodes": [dict(n) for n in BASE["nodes"]],
            "shards": [dict(s) for s in BASE["shards"]],
            "revisions": [
                dict(BASE["revisions"][0]),
                {"version": 1, "parent": 1, "global_rate": None,
                 "global_burst": None, "node_rates": {}, "node_bursts": {},
                 "shard_deps": {}, "scale": None},
            ], "events": []}
    with pytest.raises(SpecError) as exc:
        parse_spec(data)
    assert exc.value.field == "version"


def test_shard_dependency_cycle():
    data = {"nodes": [dict(n) for n in BASE["nodes"]],
            "shards": [
                {"shard_id": "a", "upstream": ["b"], "sample_rows": 1,
                 "weight": 1, "priority": 0},
                {"shard_id": "b", "upstream": ["a"], "sample_rows": 1,
                 "weight": 1, "priority": 0},
            ],
            "revisions": [dict(BASE["revisions"][0])], "events": []}
    with pytest.raises(SpecError) as exc:
        parse_spec(data)
    assert exc.value.field == "upstream"


def test_event_unknown_kind_and_ordering():
    data = {"nodes": [dict(n) for n in BASE["nodes"]],
            "shards": [dict(s) for s in BASE["shards"]],
            "revisions": [dict(BASE["revisions"][0])],
            "events": [
                {"seq": 1, "kind": "nope", "ts_ms": 0, "shard_id": "s1",
                 "node_id": "n1", "version": None},
            ]}
    with pytest.raises(SpecError) as exc:
        parse_spec(data)
    assert exc.value.field == "kind"


def test_demo_spec_file_loads():
    spec = load_spec(Path(__file__).resolve().parents[1]
                     / "samples" / "demo_spec.json")
    assert len(spec.nodes) == 3
    assert len(spec.events) >= 10
