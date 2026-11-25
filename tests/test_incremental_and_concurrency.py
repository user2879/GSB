from __future__ import annotations

from search_proto.engine import SearchEngine
from search_proto.loader import EditCommand
from search_proto.model import Document, QuerySpec
from search_proto.report import comparison_projection

from .conftest import make_config, make_stopwords, make_vocab


def _docs() -> list[Document]:
    return [
        Document.create("D1", 0, {"title": ["alpha"], "body": ["beta"]}),
        Document.create("D2", 0, {"title": ["gamma"], "body": ["delta"]}),
        Document.create("D3", 0, {"title": ["alpha"], "body": ["gamma"]}),
    ]


def _queries() -> list[QuerySpec]:
    return [
        QuerySpec.create("Q_A", ["alpha"], ["title", "body"]),
        QuerySpec.create("Q_G", ["gamma"], ["title", "body"]),
    ]


def _final_full(edit: EditCommand) -> SearchEngine:
    config = make_config(top_k=3)
    vocab = make_vocab()
    stopwords = make_stopwords()
    pre = SearchEngine(config, stopwords, vocab, _docs(), _queries())
    pre.apply_edits([edit])
    full = SearchEngine(
        pre.config,
        pre.stopwords,
        pre.vocabulary,
        pre.ordered_documents,
        pre.ordered_queries,
        failures=pre.sorted_failures(),
    )
    full.materialize()
    full.run_batches()
    return full


def test_targeted_invalidation_and_exact_reuse() -> None:
    config = make_config(top_k=3)
    vocab = make_vocab()
    stopwords = make_stopwords()
    engine = SearchEngine(config, stopwords, vocab, _docs(), _queries())
    engine.materialize()
    engine.run_batches()

    replacement = Document.create(
        "D2", 0, {"title": ["gamma"], "body": ["beta"]}
    )
    edit = EditCommand(
        op="replace_doc", line=1, payload={"doc": replacement}
    )
    engine.apply_edit(edit)
    engine.materialize()
    engine.run_batches()

    ledger = {
        (entry["scope"], entry["entity"]): entry
        for entry in engine.cache_ledger_entries()
    }
    assert ledger[("query_ranking", "Q_A")]["decision"] == "reused"
    assert ledger[("query_ranking", "Q_G")]["decision"] == "rebuilt"
    assert engine.report()["cache_entry_counts"]["reused"] > 0

    full = _final_full(edit)
    assert comparison_projection(engine) == comparison_projection(full)


def test_threaded_and_serial_engines_are_field_identical() -> None:
    config = make_config(top_k=3, max_workers=4)
    vocab = make_vocab()
    stopwords = make_stopwords()
    threaded = SearchEngine(config, stopwords, vocab, _docs(), _queries())
    serial = SearchEngine(
        config, stopwords, vocab, _docs(), _queries(), max_workers_override=1
    )
    for engine in (threaded, serial):
        engine.materialize()
        engine.run_batches()

    assert comparison_projection(threaded) == comparison_projection(serial)
    assert threaded.report()["concurrency"]["max_active_observed"] <= config.max_workers
    assert all(
        observation["max_active_observed"] <= config.max_workers
        for observation in threaded.concurrency_observations
    )

