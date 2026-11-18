"""Requirement 3: extraction window, citation backfill and E_CITATION_MISSING."""

from __future__ import annotations

from rag_app.engine_keys import citation_key, snippets_fingerprint
from rag_app.errors import E_CITATION_MISSING
from rag_app.extraction import assemble_citations, extract_snippets
from rag_app.models import Document, RetrievalConfig
from rag_app.retrieval import BM25Index
from rag_app.rules import load_rule_table


def _rules():
    return load_rule_table(
        {
            "version": "r1",
            "synonyms": {},
            "category_expand": {},
            "templates": {
                "standard": (
                    "Q:{query}|S:{top_snippet}|N:{citation_count}|D:{top_doc_id}"
                ),
                "concise": "{query}->{top_doc_id}:{citation_count}",
            },
        }
    )


def _engine_parts(min_citations: int = 4):
    docs = [
        Document(
            "D1",
            "phone guide",
            "phone battery policy. phone shipping terms. phone warranty text.",
            "catalog",
        ),
        Document(
            "D2",
            "phone review",
            "phone battery tested. phone screen checked.",
            "review",
        ),
    ]
    config = RetrievalConfig.from_dict(
        {
            "normalization": {
                "fold_full_half": True,
                "case_fold": True,
                "whitespace_merge": True,
            },
            "bm25": {"k1": 1.5, "b": 0.75},
            "recall_top_k": 2,
            "answer_window": 2,
            "min_citations": min_citations,
            "fusion_weights": {
                "bm25_weight": 1.0,
                "title_weight": 0.1,
                "overlap_weight": 0.1,
                "sentence_weight": 1.0,
                "doc_balance_weight": 0.2,
            },
            "first_response_budget_ms": 100,
            "max_concurrency": 2,
            "snapshot_commit_interval": 1,
        }
    )
    rules = _rules()
    index = BM25Index(docs, config, rules)
    retrieval = index.search("rk", "phone battery", config.recall_top_k)
    return config, rules, index, retrieval


def test_window_limits_snippets_and_backfill_meets_minimum():
    config, rules, index, retrieval = _engine_parts(min_citations=4)
    extraction = extract_snippets("ek", index, retrieval, config, rules)
    assert len(extraction.snippets) == config.answer_window
    citation = assemble_citations(
        "ck",
        extraction,
        retrieval,
        index,
        config,
        rules,
        "standard",
        "S1#0001",
    )
    assert citation.answer_available
    assert len(citation.citations) >= 4
    assert any(item["reason"] == "score_desc_backfill" for item in citation.backfilled)
    assert citation.citations[0].backfilled is False


def test_missing_citation_returns_business_code_and_unavailable_answer():
    config, rules, index, retrieval = _engine_parts()
    empty_retrieval = index.search("rk2", "zzx-oov-token", config.recall_top_k)
    extraction = extract_snippets("ek2", index, empty_retrieval, config, rules)
    citation = assemble_citations(
        "ck2",
        extraction,
        empty_retrieval,
        index,
        config,
        rules,
        "standard",
        "S1#0009",
    )
    assert citation.answer_available is False
    assert citation.answer == ""
    assert citation.failure is not None
    assert citation.failure.code == E_CITATION_MISSING


def test_template_change_only_changes_citation_key():
    config, rules, index, retrieval = _engine_parts()
    extraction = extract_snippets("ek", index, retrieval, config, rules)
    fp = snippets_fingerprint(extraction.snippets)
    key_a = citation_key(
        parent_key="ek",
        snippet_fingerprint=fp,
        config=config,
        rules=rules,
        template_name="standard",
    )
    key_b = citation_key(
        parent_key="ek",
        snippet_fingerprint=fp,
        config=config,
        rules=rules,
        template_name="concise",
    )
    assert key_a != key_b
