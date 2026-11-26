from __future__ import annotations

import json

import pytest

from workbench.errors import SpecError
from workbench.spec import load_spec
from workbench.testkit import write_test_spec


def test_example_spec_is_valid(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec = write_test_spec(spec_path)
    assert spec.topo_order[:2] == ("s1", "s2")
    assert spec.digest


@pytest.mark.parametrize(
    "mutate,field",
    [
        (lambda s: s["nodes"][0].__setitem__("node_id", "n2"), "nodes.node_id"),
        (lambda s: s["nodes"][0].__setitem__("host", "localhost"), "nodes.host"),
        (lambda s: s["nodes"][0].__setitem__("port", 70000), "nodes.port"),
        (lambda s: s["nodes"][0].__setitem__("concurrency_quota", -1), "nodes.concurrency_quota"),
        (lambda s: s["shards"][0].__setitem__("sample_rows", 0), "shards.sample_rows"),
        (lambda s: s["shards"][0].__setitem__("resource_units", 0), "shards.resource_units"),
        (lambda s: s["capacity_chain"][1].__setitem__("version", 1), "capacity_chain.version"),
        (lambda s: s["events"][1].__setitem__("seq", 1), "events.seq"),
    ],
)
def test_spec_first_violation_has_field_and_location(tmp_path, mutate, field):
    path = tmp_path / "bad.json"
    spec = write_test_spec(path)
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw))
    with pytest.raises(SpecError) as exc:
        load_spec(path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.field == field
    assert exc.value.location.startswith("$")


def test_duplicate_ports_are_rejected(tmp_path):
    path = tmp_path / "ports.json"
    raw = write_test_spec(path).raw
    raw["nodes"][1]["port"] = raw["nodes"][0]["port"]
    path.write_text(json.dumps(raw))
    with pytest.raises(SpecError) as exc:
        load_spec(path)
    assert exc.value.field == "nodes.port"
    assert "[1]" in exc.value.location


def test_missing_upstream_and_cycle_are_rejected(tmp_path):
    path = tmp_path / "dep.json"
    raw = write_test_spec(path).raw
    raw["shards"][0]["upstreams"] = ["missing"]
    path.write_text(json.dumps(raw))
    with pytest.raises(SpecError) as exc:
        load_spec(path)
    assert exc.value.field == "shards.upstreams"

    raw = write_test_spec(path).raw
    by_id = {s["shard_id"]: s for s in raw["shards"]}
    by_id["s1"]["upstreams"] = ["s2"]
    by_id["s2"]["upstreams"] = ["s1"]
    path.write_text(json.dumps(raw))
    with pytest.raises(SpecError) as exc:
        load_spec(path)
    assert "cycle" in exc.value.message


def test_bad_json_reports_line_location(tmp_path):
    path = tmp_path / "syntax.json"
    path.write_text('{\n  "nodes": [\n', encoding="utf-8")
    with pytest.raises(SpecError) as exc:
        load_spec(path)
    assert "line" in exc.value.location
