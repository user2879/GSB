import json

import pytest

from kvdiag.errors import SpecError
from kvdiag.spec import load_all_rows, load_spec


def test_valid_tree_loads(tree):
    spec = load_spec(tree["spec"])
    assert spec.shard_ids() == ["shard-0", "shard-1", "shard-2", "shard-3"]
    rows = load_all_rows(spec)
    assert set(rows) == set(spec.shard_ids())


def _mutate(tree, fn):
    path = tree["spec"]
    decl = json.loads(path.read_text(encoding="utf-8"))
    fn(decl)
    path.write_text(json.dumps(decl, indent=2), encoding="utf-8")
    return path


def test_duplicate_shard_id_rejected(tree):
    def dup(decl):
        decl["shards"][1]["shard_id"] = "shard-0"
    path = _mutate(tree, dup)
    with pytest.raises(SpecError) as exc:
        load_spec(path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.shard_id == "shard-0"
    assert exc.value.field == "shard_id"


def test_overlapping_ranges_rejected(tree):
    def overlap(decl):
        decl["shards"][1]["range"][0] -= 1
    path = _mutate(tree, overlap)
    with pytest.raises(SpecError) as exc:
        load_spec(path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.shard_id == "shard-1"
    assert exc.value.field == "range"


def test_non_contiguous_ranges_rejected(tree):
    def gap(decl):
        decl["shards"][1]["range"][0] += 10
        decl["shards"][1]["range"][1] += 10
        decl["shards"][2]["range"][0] += 10
        decl["shards"][2]["range"][1] += 10
        decl["shards"][3]["range"][0] += 10
    path = _mutate(tree, gap)
    with pytest.raises(SpecError) as exc:
        load_spec(path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.field == "range"


def test_lease_fields_validated(tree):
    def bad_ttl(decl):
        decl["shards"][2]["lease"]["ttl_ms"] = 0
    with pytest.raises(SpecError) as exc:
        load_spec(_mutate(tree, bad_ttl))
    assert exc.value.shard_id == "shard-2"
    assert exc.value.field == "lease.ttl_ms"

    def bad_interval(decl):
        decl["shards"][1]["lease"]["renew_interval_ms"] = \
            decl["shards"][1]["lease"]["ttl_ms"]
    with pytest.raises(SpecError) as exc:
        load_spec(_mutate(tree, bad_interval))
    assert exc.value.shard_id == "shard-1"
    assert exc.value.field == "lease.renew_interval_ms"

    def bad_inflight(decl):
        decl["shards"][0]["lease"]["max_inflight_renewals"] = -1
    with pytest.raises(SpecError) as exc:
        load_spec(_mutate(tree, bad_inflight))
    assert exc.value.shard_id == "shard-0"
    assert exc.value.field == "lease.max_inflight_renewals"


def test_index_rule_missing_column_rejected(tree):
    def bad_rule(decl):
        decl["shards"][0]["index_rules"] = [
            {"column": "nosuch", "kind": "prefix", "prefix_len": 2}]
    path = _mutate(tree, bad_rule)
    spec = load_spec(path)
    with pytest.raises(SpecError) as exc:
        load_all_rows(spec)
    assert exc.value.shard_id == "shard-0"
    assert exc.value.field == "index_rules.column"


def test_sample_key_out_of_range_rejected(tree):
    spec = load_spec(tree["spec"])
    sample = tree["root"] / "samples" / "shard-1.json"
    doc = json.loads(sample.read_text(encoding="utf-8"))
    doc["rows"][0]["key"] = doc["rows"][0]["key"] + (1 << 16)
    sample.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SpecError) as exc:
        load_all_rows(spec)
    assert exc.value.code == "E_SPEC"
    assert exc.value.shard_id == "shard-1"
    assert exc.value.field == "rows.key"


def test_no_partial_load(tree):
    spec = load_spec(tree["spec"])
    sample = tree["root"] / "samples" / "shard-3.json"
    sample.write_text("{ not json", encoding="utf-8")
    with pytest.raises(SpecError):
        load_all_rows(spec)
