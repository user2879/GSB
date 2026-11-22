import copy
import json

import pytest

from edge_lease.errors import CycleError, ManifestError
from edge_lease.manifest import (
    load_manifests,
    load_registry,
    load_shards,
    topological_order,
)
from tests.conftest import write_json


def test_valid_inputs_load(sample_workspace):
    registry = load_registry(sample_workspace / "nodes.json")
    shards = load_shards(sample_workspace / "shards.json")
    manifests = load_manifests(
        sample_workspace / "manifest.json", registry, shards
    )
    assert [m.version for m in manifests] == [1, 2]
    assert len(manifests[0].entries) == 3


def test_version_must_be_strictly_increasing(sample_workspace):
    doc = json.loads((sample_workspace / "manifest.json").read_text())
    doc["versions"][1]["version"] = 1
    write_json(sample_workspace / "bad.json", doc)
    registry = load_registry(sample_workspace / "nodes.json")
    shards = load_shards(sample_workspace / "shards.json")
    with pytest.raises(ManifestError) as exc:
        load_manifests(sample_workspace / "bad.json", registry, shards)
    assert exc.value.code == "E_MANIFEST"
    assert "versions[1].version" in exc.value.location


def test_unknown_target_node_rejected(sample_workspace):
    doc = json.loads((sample_workspace / "manifest.json").read_text())
    doc["versions"][0]["entries"][0]["targets"] = ["edge-zzz"]
    write_json(sample_workspace / "bad.json", doc)
    registry = load_registry(sample_workspace / "nodes.json")
    shards = load_shards(sample_workspace / "shards.json")
    with pytest.raises(ManifestError) as exc:
        load_manifests(sample_workspace / "bad.json", registry, shards)
    assert "targets[0]" in exc.value.location


def test_shard_gap_rejected(sample_workspace):
    doc = json.loads((sample_workspace / "shards.json").read_text())
    doc["paths"][0]["shards"][1]["start"] = 50
    write_json(sample_workspace / "bad_shards.json", doc)
    with pytest.raises(ManifestError) as exc:
        load_shards(sample_workspace / "bad_shards.json")
    assert exc.value.code == "E_MANIFEST"
    assert "shards" in exc.value.location


def test_derived_source_must_exist(sample_workspace):
    doc = json.loads((sample_workspace / "manifest.json").read_text())
    doc["versions"][0]["entries"][2]["derived"]["sources"][0][0] = "nope.bin"
    write_json(sample_workspace / "bad.json", doc)
    registry = load_registry(sample_workspace / "nodes.json")
    shards = load_shards(sample_workspace / "shards.json")
    with pytest.raises(ManifestError):
        load_manifests(sample_workspace / "bad.json", registry, shards)


def test_derived_cycle_raises_e_cycle(sample_workspace):
    doc = json.loads((sample_workspace / "manifest.json").read_text())
    entries = doc["versions"][0]["entries"]
    # intro -> body -> intro cycle
    entries[0]["derived"] = {"sources": [["media/body.bin", 0.5]]}
    entries[1]["derived"] = {"sources": [["media/intro.bin", 0.5]]}
    doc["versions"] = [doc["versions"][0]]
    write_json(sample_workspace / "cycle.json", doc)
    registry = load_registry(sample_workspace / "nodes.json")
    shards = load_shards(sample_workspace / "shards.json")
    with pytest.raises(CycleError) as exc:
        load_manifests(sample_workspace / "cycle.json", registry, shards)
    assert exc.value.code == "E_CYCLE"
    assert exc.value.context["cycle"]


def test_topological_order_is_stable(sample_workspace):
    registry = load_registry(sample_workspace / "nodes.json")
    shards = load_shards(sample_workspace / "shards.json")
    manifests = load_manifests(
        sample_workspace / "manifest.json", registry, shards
    )
    entries = manifests[0].entry_map()
    order = topological_order(entries)
    assert order.index("media/index.json") > max(
        order.index("media/intro.bin"), order.index("media/body.bin")
    )
    # shuffled input still yields the same order
    shuffled = copy.deepcopy(manifests[0])
    object.__setattr__(
        shuffled, "entries", tuple(reversed(shuffled.entries))
    )
    assert topological_order(shuffled.entry_map()) == order
