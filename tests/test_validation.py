from __future__ import annotations

import json
from pathlib import Path

import pytest

from walcrash.errors import BusinessError
from walcrash.validation import load_sample

from .conftest import ALT_SAMPLE, SAMPLE, make_sample_variant


def test_valid_sample_and_deterministic_targets() -> None:
    sample = load_sample(SAMPLE)
    assert [page.page_id for page in sample.pages] == ["p1", "p2", "p3"]
    assert len(sample.entries) == 12
    assert sample.crash_point_lsn == 10
    assert sample.committed_txids == ("tx-001", "tx-002", "tx-003")
    versions = {entry.lsn: entry.target_version for entry in sample.entries}
    assert versions == {1: 2, 2: 4, 3: 5, 4: 3, 5: 1, 6: 4, 7: 6, 8: 2, 9: 5, 10: 7, 11: 6, 12: 3}
    assert [entry.visible for entry in sample.entries] == [True] * 10 + [False, False]
    assert load_sample(ALT_SAMPLE).crash_point_lsn == 4


def test_page_errors_are_e_page_and_page_table_is_unique(tmp_path: Path) -> None:
    with pytest.raises(BusinessError) as invalid_size:
        load_sample("samples/invalid_page_size.json")
    assert invalid_size.value.code == "E_PAGE"
    assert invalid_size.value.context["field"] == "size_bytes"

    document = json.loads(SAMPLE.read_text(encoding="utf-8"))
    document["log_file"] = str((SAMPLE.parent / document["log_file"]).resolve())
    document["pages"] = [dict(document["pages"][0]), dict(document["pages"][0])]
    path = tmp_path / "duplicate_page.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(BusinessError) as duplicate:
        load_sample(path)
    assert duplicate.value.code == "E_PAGE"
    assert duplicate.value.context["field"] == "page_id"
    assert duplicate.value.context["page_index"] == 1


def test_order_error_is_e_order_with_first_bad_lsn() -> None:
    with pytest.raises(BusinessError) as error:
        load_sample("samples/invalid_order.json")
    assert error.value.code == "E_ORDER"
    assert error.value.context["field"] == "lsn"
    assert error.value.context["first_invalid_lsn"] == 1
    assert error.value.context["expected_lsn"] == 2


def test_unknown_page_and_digest_mismatch_are_all_or_nothing(tmp_path: Path) -> None:
    unknown_page = make_sample_variant(tmp_path, mutations={1: {"page_id": "missing-page"}})
    with pytest.raises(BusinessError) as error:
        load_sample(unknown_page)
    assert error.value.code == "E_LOG"
    assert error.value.context["first_invalid_lsn"] == 1
    assert error.value.context["field"] == "page_id"

    bad_digest = make_sample_variant(tmp_path, mutations={2: {"content_digest": "0" * 64}})
    with pytest.raises(BusinessError) as digest_error:
        load_sample(bad_digest)
    assert digest_error.value.code == "E_LOG"
    assert digest_error.value.context["first_invalid_lsn"] == 2
    assert digest_error.value.context["field"] == "content_digest"


def test_transactions_must_remain_grouped(tmp_path: Path) -> None:
    path = make_sample_variant(
        tmp_path,
        mutations={2: {"txid": "tx-999"}, 3: {"txid": "tx-001"}},
    )
    with pytest.raises(BusinessError) as error:
        load_sample(path)
    assert error.value.code == "E_LOG"
    assert error.value.context["first_invalid_lsn"] == 3
    assert error.value.context["field"] == "txid"


def test_checkpoint_interval_and_page_budget_are_validated(tmp_path: Path) -> None:
    path = make_sample_variant(
        tmp_path,
        budget={
            "global_slots": 3,
            "per_tx_quota": 2,
            "page_byte_budget": 1,
            "checkpoint_interval": 4,
        },
    )
    with pytest.raises(BusinessError) as error:
        load_sample(path)
    assert error.value.code == "E_BUDGET"
    assert error.value.context["field"] == "page_byte_budget"

