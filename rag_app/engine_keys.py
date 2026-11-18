"""Content-fingerprint cache keys for the three pipeline stages."""

from __future__ import annotations

from .fingerprint import cache_key, fingerprint
from .models import RetrievalConfig
from .retrieval import bm25_param_fingerprint
from .rules import RuleTable


def retrieval_key(
    *,
    effective_query: str,
    config: RetrievalConfig,
    rules: RuleTable,
    index_fingerprint: str,
) -> str:
    return cache_key(
        "retrieval",
        [
            {"q": effective_query},
            {"params": bm25_param_fingerprint(config)},
            {"index": index_fingerprint},
            {"rules_version": rules.version},
        ],
    )


def extraction_key(
    *,
    parent_key: str,
    retrieval_doc_set: tuple[str, ...],
    query_terms: tuple[str, ...],
    config: RetrievalConfig,
    rules: RuleTable,
) -> str:
    return cache_key(
        "extraction",
        [
            {"parent": parent_key},
            {"doc_set": list(retrieval_doc_set)},
            {"query_terms": list(query_terms)},
            {"window": config.answer_window},
            {"fusion": config.fusion_weights},
            {"rules_version": rules.version},
        ],
    )


def citation_key(
    *,
    parent_key: str,
    snippet_fingerprint: str,
    config: RetrievalConfig,
    rules: RuleTable,
    template_name: str,
) -> str:
    return cache_key(
        "citation",
        [
            {"parent": parent_key},
            {"snippets": snippet_fingerprint},
            {"min_citations": config.min_citations},
            {"rules_version": rules.version},
            {"template": template_name},
            {"template_body": rules.template(template_name)},
        ],
    )


def snippets_fingerprint(snippets) -> str:
    return fingerprint(
        [
            {
                "doc_id": snippet.doc_id,
                "sentence_index": snippet.sentence_index,
                "score": round(snippet.score, 6),
            }
            for snippet in snippets
        ]
    )
