import random

import pytest

from kvdiag.errors import SpecError
from kvdiag.index import build_index_entries, entries_bytes
from kvdiag.spec import load_all_rows, load_spec


def test_entries_deterministic_under_row_shuffle(tree):
    spec = load_spec(tree["spec"])
    rows = load_all_rows(spec)
    shard = spec.shard("shard-0")
    base = build_index_entries(shard, rows["shard-0"], index_version=1)
    for seed in range(5):
        shuffled = list(rows["shard-0"])
        random.Random(seed).shuffle(shuffled)
        again = build_index_entries(shard, shuffled, index_version=1)
        assert entries_bytes(again) == entries_bytes(base)


def test_entry_fields_and_positions(tree):
    spec = load_spec(tree["spec"])
    rows = load_all_rows(spec)
    shard = spec.shard("shard-1")
    entries = build_index_entries(shard, rows["shard-1"], index_version=7)
    assert entries, "expected entries"
    for e in entries:
        assert e["shard_id"] == "shard-1"
        assert len(e["key_digest"]) == 16
        assert e["index_version"] == 7
    by_rule = {}
    for e in entries:
        by_rule.setdefault(e["rule"], []).append(e["position"])
    for rule, positions in by_rule.items():
        assert positions == list(range(len(positions)))


def test_key_out_of_shard_rejected(tree):
    spec = load_spec(tree["spec"])
    rows = load_all_rows(spec)
    shard = spec.shard("shard-0")
    bad = dict(rows["shard-0"][0])
    bad["key"] = shard.hi + 100
    with pytest.raises(SpecError) as exc:
        build_index_entries(shard, rows["shard-0"] + [bad], index_version=1)
    assert exc.value.code == "E_SPEC"
    assert exc.value.shard_id == "shard-0"


def test_index_version_changes_bytes(tree):
    spec = load_spec(tree["spec"])
    rows = load_all_rows(spec)
    shard = spec.shard("shard-2")
    v1 = build_index_entries(shard, rows["shard-2"], index_version=1)
    v2 = build_index_entries(shard, rows["shard-2"], index_version=2)
    assert entries_bytes(v1) != entries_bytes(v2)
