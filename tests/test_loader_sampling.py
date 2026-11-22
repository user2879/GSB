from __future__ import annotations

from collections import Counter

from corpus_workbench.codes import (
    E_DUP_DOC_ID,
    E_ENCODING_INVALID,
    E_SOURCE_UNKNOWN,
)
from corpus_workbench.config import load_balance_config
from corpus_workbench.demo_data import generate_demo_workspace
from corpus_workbench.loader import load_documents
from corpus_workbench.sampling import allocate_source_counts, sample_documents


def test_loader_business_codes_and_injection_trace(tmp_path):
    generate_demo_workspace(tmp_path)
    balance = load_balance_config(tmp_path / "configs" / "balance.json")
    loaded = load_documents(tmp_path / "data" / "corpus", balance.source_weights)

    assert len(loaded.documents) == 72
    assert len(loaded.failures) == 4
    assert {failure.code for failure in loaded.failures} == {
        E_DUP_DOC_ID,
        E_ENCODING_INVALID,
        E_SOURCE_UNKNOWN,
    }
    injection_ids = {failure.injection_id for failure in loaded.failures}
    assert {
        "inj-empty-text",
        "inj-non-utf8",
        "inj-unknown-source",
        "inj-duplicate-doc-id",
    } <= injection_ids
    assert loaded.injection_traces["inj-non-utf8"]["code"] == E_ENCODING_INVALID


def test_sampling_is_exactly_repeatable_and_weighted(tmp_path):
    generate_demo_workspace(tmp_path)
    balance = load_balance_config(tmp_path / "configs" / "balance.json")
    loaded = load_documents(tmp_path / "data" / "corpus", balance.source_weights)

    first, first_failures = sample_documents(loaded.documents, balance)
    second, second_failures = sample_documents(loaded.documents, balance)

    assert first_failures == []
    assert second_failures == []
    assert first == second
    assert len(first) == balance.target_sample_count
    assert [sample.order for sample in first] == list(range(len(first)))
    assert [(sample.source, sample.sample_index) for sample in first] == sorted(
        (sample.source, sample.sample_index) for sample in first
    )
    counts = Counter(sample.source for sample in first)
    expected = allocate_source_counts(
        balance.source_weights,
        balance.target_sample_count,
        {"books", "papers", "web"},
    )
    assert dict(counts) == expected
    assert max(counts.values()) > min(counts.values())
    assert len({(sample.source, sample.doc_id) for sample in first}) < len(first)

