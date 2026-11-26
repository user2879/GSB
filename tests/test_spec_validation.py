import json
from pathlib import Path

import pytest

from cacheflow.errors import SpecError
from cacheflow.spec import load_bundle, load_nodes_spec


ROOT = Path(__file__).resolve().parents[1]


def example_bundle():
    return load_bundle(ROOT / "examples/nodes.json", ROOT / "examples/artifacts.json", ROOT / "examples/events.json")


def test_example_specs_are_valid():
    bundle = example_bundle()
    assert len(bundle.nodes) == 3
    assert len(bundle.artifacts) == 4
    assert len(bundle.events) == 29
    assert len({node.port for node in bundle.nodes}) == 3


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def base_nodes():
    return json.loads((ROOT / "examples/nodes.json").read_text())


def test_duplicate_node_id_reports_first_field(tmp_path):
    doc = base_nodes()
    doc["nodes"][1]["node_id"] = "node-a"
    path = write_json(tmp_path / "nodes.json", doc)
    with pytest.raises(SpecError) as exc:
        load_nodes_spec(path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.field == "node_id"
    assert "nodes[1]" in str(exc.value.entry)


def test_host_must_be_literal_loopback(tmp_path):
    doc = base_nodes()
    doc["nodes"][0]["host"] = "localhost"
    path = write_json(tmp_path / "nodes.json", doc)
    with pytest.raises(SpecError) as exc:
        load_nodes_spec(path)
    assert exc.value.field == "host"


def test_duplicate_port_rejected(tmp_path):
    doc = base_nodes()
    doc["nodes"][1]["port"] = doc["nodes"][0]["port"]
    path = write_json(tmp_path / "nodes.json", doc)
    with pytest.raises(SpecError) as exc:
        load_nodes_spec(path)
    assert exc.value.field == "port"


@pytest.mark.parametrize("field", ["capacity_slots", "max_leases"])
def test_quota_fields_positive_int(tmp_path, field):
    doc = base_nodes()
    doc["nodes"][0][field] = 0
    path = write_json(tmp_path / "nodes.json", doc)
    with pytest.raises(SpecError) as exc:
        load_nodes_spec(path)
    assert exc.value.field == field


def test_event_sequence_must_increase(tmp_path):
    nodes = ROOT / "examples/nodes.json"
    artifacts = ROOT / "examples/artifacts.json"
    events_doc = json.loads((ROOT / "examples/events.json").read_text())
    events_doc["events"][1]["seq"] = 1
    events_path = write_json(tmp_path / "events.json", events_doc)
    with pytest.raises(SpecError) as exc:
        load_bundle(nodes, artifacts, events_path)
    assert exc.value.field == "seq"


def test_shard_ranges_must_be_contiguous(tmp_path):
    artifacts_doc = json.loads((ROOT / "examples/artifacts.json").read_text())
    artifacts_doc["artifacts"][0]["shards"][1]["start"] = 5
    artifacts_path = write_json(tmp_path / "artifacts.json", artifacts_doc)
    with pytest.raises(SpecError) as exc:
        load_bundle(ROOT / "examples/nodes.json", artifacts_path, ROOT / "examples/events.json")
    assert exc.value.code == "E_SPEC"
    assert exc.value.field in {"start", "end"}


def test_shard_must_not_exceed_total_bytes(tmp_path):
    artifacts_doc = json.loads((ROOT / "examples/artifacts.json").read_text())
    artifacts_doc["artifacts"][0]["shards"][-1]["end"] = 99
    artifacts_path = write_json(tmp_path / "artifacts.json", artifacts_doc)
    with pytest.raises(SpecError) as exc:
        load_bundle(ROOT / "examples/nodes.json", artifacts_path, ROOT / "examples/events.json")
    assert exc.value.field == "end"
