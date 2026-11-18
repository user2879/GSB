"""Requirements 2: normalization, BM25 ranking and tie-break determinism."""

from __future__ import annotations

from rag_app.models import Document, RetrievalConfig
from rag_app.normalize import fold_full_half, normalize_text, tokenize
from rag_app.retrieval import BM25Index
from rag_app.rules import load_rule_table


def test_full_width_folding_round_trip():
    text = "ＮＩＭＢＵＳ　ＰＨＯＮＥ"
    assert fold_full_half(text) == "NIMBUS PHONE"
    rules = {"fold_full_half": True, "case_fold": True, "whitespace_merge": True}
    assert normalize_text("  Nimbus\u3000Phone\n", rules) == "nimbus phone"
    assert tokenize("A  B   C", rules) == ["a", "b", "c"]


def _config():
    return RetrievalConfig.from_dict(
        {
            "normalization": {
                "fold_full_half": True,
                "case_fold": True,
                "whitespace_merge": True,
            },
            "bm25": {"k1": 1.5, "b": 0.75},
            "recall_top_k": 3,
            "answer_window": 2,
            "min_citations": 2,
            "fusion_weights": {
                "bm25_weight": 1.0,
                "title_weight": 0.0,
                "overlap_weight": 0.0,
                "sentence_weight": 1.0,
                "doc_balance_weight": 0.2,
            },
            "first_response_budget_ms": 100,
            "max_concurrency": 2,
            "snapshot_commit_interval": 1,
        }
    )


def _rules():
    return load_rule_table(
        {
            "version": "t",
            "synonyms": {},
            "category_expand": {},
            "templates": {"standard": "{query}|{top_snippet}|{citation_count}"},
        }
    )


def test_bm25_is_repeatable_and_lexicographic_tie_break():
    docs = [
        Document("D01", "phone guide", "phone battery phone phone", "s"),
        Document("D02", "phone guide", "phone battery phone phone", "s"),
        Document("D03", "other", "laptop memory warranty text", "s"),
    ]
    index = BM25Index(docs, _config(), _rules())
    first = index.search("k1", "phone battery", top_k=3)
    second = index.search("k2", "phone battery", top_k=3)
    ids_first = [c.doc_id for c in first.candidates]
    ids_second = [c.doc_id for c in second.candidates]
    assert ids_first == ids_second
    # D01/D02 are identical content: doc_id lexicographic order wins.
    assert ids_first[:2] == ["D01", "D02"]
    assert first.candidates[0].score == second.candidates[0].score


def test_recall_top_k_is_enforced_and_zero_score_excluded():
    docs = [
        Document(f"D{i:02d}", "title", f"phone battery text {i}", "s")
        for i in range(10)
    ]
    config = _config()
    index = BM25Index(docs, config, _rules())
    result = index.search("k", "phone battery", top_k=4)
    assert len(result.candidates) == 4
    none = index.search("k2", "unseen-vocab-token", top_k=4)
    assert none.candidates == ()


def test_synonym_and_category_expansion_feed_retrieval():
    docs = [
        Document("D1", "phone", "smartphone battery life details", "s"),
        Document("D2", "laptop", "workstation memory details", "s"),
    ]
    rules = load_rule_table(
        {
            "version": "t",
            "synonyms": {"mobile": "phone"},
            "category_expand": {"phone": ["smartphone"]},
            "templates": {"standard": "{query}"},
        }
    )
    index = BM25Index(docs, _config(), rules)
    result = index.search("k", "mobile", top_k=2)
    assert result.doc_set == ("D1",)
