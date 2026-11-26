import copy
import json
from pathlib import Path

import pytest

from mvcc_lab.errors import SpecError
from mvcc_lab.loader import build_spec, load_spec, spec_digests

SAMPLE = ROOT_SAMPLE = None


@pytest.fixture
def raw():
    path = Path(__file__).parents[1] / "samples" / "sample_spec.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_loads_deterministically(raw):
    spec = build_spec(raw)
    assert len(spec.transactions) == 5
    assert spec.max_seq == 8
    assert set(spec.row_chains) == {"R-A", "R-B", "R-C"}


def test_duplicate_transaction_id(raw):
    raw["transactions"][1]["txn_id"] = raw["transactions"][0]["txn_id"]
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.code == "E_SPEC"
    assert info.value.field == "txn_id"
    assert info.value.object_id == "T1"


def test_negative_begin_seq(raw):
    raw["transactions"][0]["begin_seq"] = -1
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.field == "begin_seq"
    assert info.value.object_id == "T1"


def test_bool_rejected_as_integer(raw):
    raw["transactions"][0]["begin_seq"] = True
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.field == "begin_seq"


def test_bad_outcome(raw):
    raw["transactions"][0]["outcome"] = "maybe"
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.field == "outcome"


def test_commit_seq_only_for_committed(raw):
    raw["transactions"][2]["commit_seq"] = 7
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.object_id == "T3"
    assert info.value.field == "commit_seq"


def test_committed_requires_commit_seq(raw):
    raw["transactions"][0]["commit_seq"] = None
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.field == "commit_seq"


def test_missing_prev_reference(raw):
    raw["records"][1]["prev_version"] = 99
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.object_id == "R-A"
    assert info.value.field == "prev_version"


def test_chain_cycle(raw):
    # 让头版本回指链内版本，制造环。
    raw["records"][0]["prev_version"] = 2
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.field == "prev_version"


def test_snapshot_out_of_range(raw):
    raw["read_points"][0]["snapshot_seq"] = 100
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.object_id == "RP-1"
    assert info.value.field == "snapshot_seq"


def test_query_unknown_readpoint(raw):
    raw["queries"][0]["read_point_id"] = "NOPE"
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.field == "read_point_id"


def test_first_error_is_reported_in_declaration_order(raw):
    # 同时制造两处错误，应报告声明顺序更靠前的对象。
    raw["transactions"][0]["begin_seq"] = -1
    raw["transactions"][3]["outcome"] = "bogus"
    with pytest.raises(SpecError) as info:
        build_spec(raw)
    assert info.value.object_id == "T1"


def test_spec_digests_stable(raw):
    spec_a = build_spec(copy.deepcopy(raw))
    spec_b = build_spec(copy.deepcopy(raw))
    assert spec_digests(spec_a) == spec_digests(spec_b)
    raw["transactions"][3]["outcome"] = "committed"
    raw["transactions"][3]["commit_seq"] = 9
    changed = build_spec(raw)
    assert (spec_digests(changed)["transactions_digest"]
            != spec_digests(spec_a)["transactions_digest"])
