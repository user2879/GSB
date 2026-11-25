from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

from .config import PrototypeConfig
from .model import FIELD_ORDER, Document, StopwordTable
from .util import sha256_json


T = TypeVar("T")
INDEX_SCHEMA = "inverted-index-v1"


class ConcurrencyProbe:
    """Thread-safe observation of actual task concurrency and task keys."""

    def __init__(self, maximum_allowed: int):
        self.maximum_allowed = maximum_allowed
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.task_keys: list[str] = []

    def run(self, key: str, function: Callable[[], T]) -> T:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.task_keys.append(key)
        try:
            return function()
        finally:
            with self._lock:
                self._active -= 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_allowed": self.maximum_allowed,
            "max_active_observed": self.max_active,
            "task_count": len(self.task_keys),
            "task_keys": sorted(self.task_keys),
        }


def document_stats_cache_key(
    document: Document,
    config: PrototypeConfig,
    stopwords: StopwordTable,
) -> str:
    return sha256_json(
        {
            "schema": INDEX_SCHEMA,
            "document_fingerprint": document.content_fingerprint,
            "fields": list(config.fields),
            "stopword_fingerprint": stopwords.content_fingerprint,
        }
    )


def compute_document_stats(
    document: Document,
    config: PrototypeConfig,
    stopwords: StopwordTable,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    excluded_all: list[dict[str, Any]] = []
    for field_name in FIELD_ORDER:
        if field_name not in config.fields:
            continue
        raw_terms = document.fields.get(field_name, ())
        marked_positions = set(document.stopword_marks.get(field_name, ()))
        effective_positions: list[int] = []
        field_terms: dict[str, list[int]] = {}
        excluded: list[dict[str, Any]] = []
        for position, term in enumerate(raw_terms):
            reasons: list[str] = []
            if stopwords.is_stopword(field_name, term):
                reasons.append("stopword_table")
            if position in marked_positions:
                reasons.append("document_mark")
            if reasons:
                item = {
                    "doc_id": document.doc_id,
                    "field": field_name,
                    "position": position,
                    "term": term,
                    "reasons": sorted(set(reasons)),
                }
                excluded.append(item)
                excluded_all.append(item)
                continue
            effective_positions.append(position)
            field_terms.setdefault(term, []).append(position)

        fields[field_name] = {
            "length": len(effective_positions),
            "effective_positions": effective_positions,
            "excluded_positions": excluded,
            "terms": {
                term: {"tf": len(positions), "positions": positions}
                for term, positions in sorted(field_terms.items())
            },
        }

    stats = {
        "doc_id": document.doc_id,
        "version": document.version,
        "content_fingerprint": document.content_fingerprint,
        "fields": fields,
        "excluded_positions": sorted(
            excluded_all,
            key=lambda item: (
                FIELD_ORDER.index(item["field"]),
                item["position"],
                item["term"],
            ),
        ),
    }
    stats["field_stats_fingerprint"] = sha256_json(stats)
    return stats


def posting_record_from_stats(stats: dict[str, Any], term: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field_name in FIELD_ORDER:
        field_stats = stats["fields"].get(field_name)
        if field_stats is None:
            continue
        term_stats = field_stats["terms"].get(term)
        if term_stats is None:
            continue
        fields[field_name] = {
            "tf": term_stats["tf"],
            "positions": list(term_stats["positions"]),
            "field_length": field_stats["length"],
        }
    return {
        "doc_id": stats["doc_id"],
        "version": stats["version"],
        "content_fingerprint": stats["content_fingerprint"],
        "field_stats_fingerprint": stats["field_stats_fingerprint"],
        "fields": fields,
    }


def document_posting_terms(stats: dict[str, Any]) -> list[str]:
    if not stats or "fields" not in stats:
        return []
    terms: set[str] = set()
    for field_name in FIELD_ORDER:
        field_stats = stats["fields"].get(field_name)
        if field_stats is not None:
            terms.update(field_stats["terms"])
    return sorted(terms)


def term_stats_from_postings(
    term: str,
    term_postings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    doc_ids = sorted(term_postings)
    docs: list[dict[str, Any]] = []
    df_by_field = {field_name: 0 for field_name in FIELD_ORDER}
    totals = {field_name: 0 for field_name in FIELD_ORDER}
    counts = {field_name: 0 for field_name in FIELD_ORDER}
    for doc_id in doc_ids:
        posting = term_postings[doc_id]
        fields = posting["fields"]
        for field_name, field_value in fields.items():
            df_by_field[field_name] += 1
            totals[field_name] += field_value["field_length"]
            counts[field_name] += 1
        docs.append(
            {
                "doc_id": posting["doc_id"],
                "version": posting["version"],
                "content_fingerprint": posting["content_fingerprint"],
                "field_stats_fingerprint": posting["field_stats_fingerprint"],
                "fields": {
                    field_name: {
                        "tf": value["tf"],
                        "positions": list(value["positions"]),
                        "field_length": value["field_length"],
                    }
                    for field_name, value in sorted(
                        fields.items(), key=lambda pair: FIELD_ORDER.index(pair[0])
                    )
                },
            }
        )

    stats = {
        "term": term,
        "df": len(doc_ids),
        "df_by_field": {
            field_name: df_by_field[field_name]
            for field_name in FIELD_ORDER
            if df_by_field[field_name]
        },
        "average_field_lengths": {
            field_name: {
                "total_length": totals[field_name],
                "count": counts[field_name],
            }
            for field_name in FIELD_ORDER
            if counts[field_name]
        },
        "documents": docs,
    }
    stats["term_stats_fingerprint"] = sha256_json(stats)
    return stats


def content_fingerprint(documents: Iterable[Document]) -> str:
    return sha256_json(
        [
            {
                "doc_id": document.doc_id,
                "version": document.version,
                "content_fingerprint": document.content_fingerprint,
            }
            for document in sorted(documents, key=lambda item: item.doc_id)
        ]
    )


def index_fingerprint(
    config: PrototypeConfig,
    stopwords: StopwordTable,
    documents_content_fingerprint: str,
) -> str:
    return sha256_json(
        {
            "schema": INDEX_SCHEMA,
            "content_fingerprint": documents_content_fingerprint,
            "config_fingerprint": config.semantic_fingerprint,
            "stopword_fingerprint": stopwords.content_fingerprint,
        }
    )


@dataclass(frozen=True)
class BuiltIndex:
    document_stats: dict[str, dict[str, Any]]
    postings: dict[str, dict[str, dict[str, Any]]]
    term_stats: dict[str, dict[str, Any]]
    config_fingerprint: str
    stopword_fingerprint: str
    content_fingerprint: str
    index_fingerprint: str
    concurrency: dict[str, Any]

    @property
    def document_count(self) -> int:
        return len(self.document_stats)

    def to_dict(self) -> dict[str, Any]:
        excluded = []
        for doc_id in sorted(self.document_stats):
            excluded.extend(self.document_stats[doc_id]["excluded_positions"])
        return {
            "schema": INDEX_SCHEMA,
            "index_fingerprint": self.index_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "stopword_fingerprint": self.stopword_fingerprint,
            "document_count": self.document_count,
            "documents": [
                self.document_stats[doc_id]
                for doc_id in sorted(self.document_stats)
            ],
            "postings": [
                {
                    "term": term,
                    "documents": [
                        self.postings[term][doc_id]
                        for doc_id in sorted(self.postings[term])
                    ],
                }
                for term in sorted(self.postings)
            ],
            "term_stats": [
                self.term_stats[term] for term in sorted(self.term_stats)
            ],
            "excluded_positions": sorted(
                excluded,
                key=lambda item: (
                    item["doc_id"],
                    FIELD_ORDER.index(item["field"]),
                    item["position"],
                    item["term"],
                ),
            ),
        }


def build_index(
    documents: Iterable[Document],
    config: PrototypeConfig,
    stopwords: StopwordTable,
    *,
    max_workers: int | None = None,
) -> BuiltIndex:
    ordered_documents = sorted(documents, key=lambda item: item.doc_id)
    workers = min(
        max(1, config.max_workers if max_workers is None else max_workers),
        max(1, len(ordered_documents)),
    )
    probe = ConcurrencyProbe(maximum_allowed=config.max_workers)

    def task(document: Document) -> tuple[str, dict[str, Any]]:
        return document.doc_id, compute_document_stats(document, config, stopwords)

    tasks = [
        (f"document:{document.doc_id}:field_stats", document)
        for document in ordered_documents
    ]
    document_stats: dict[str, dict[str, Any]] = {}
    if workers <= 1:
        for key, document in tasks:
            doc_id, stats = probe.run(key, lambda document=document: task(document))
            document_stats[doc_id] = stats
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="index") as pool:
            futures = [
                pool.submit(probe.run, key, lambda document=document: task(document))
                for key, document in tasks
            ]
            for future in futures:
                doc_id, stats = future.result()
                document_stats[doc_id] = stats

    postings: dict[str, dict[str, dict[str, Any]]] = {}
    for doc_id in sorted(document_stats):
        stats = document_stats[doc_id]
        for term in document_posting_terms(stats):
            postings.setdefault(term, {})[doc_id] = posting_record_from_stats(stats, term)
    term_stats = {
        term: term_stats_from_postings(term, postings[term])
        for term in sorted(postings)
    }
    docs_content_fp = content_fingerprint(ordered_documents)
    return BuiltIndex(
        document_stats=document_stats,
        postings=postings,
        term_stats=term_stats,
        config_fingerprint=config.semantic_fingerprint,
        stopword_fingerprint=stopwords.content_fingerprint,
        content_fingerprint=docs_content_fp,
        index_fingerprint=index_fingerprint(config, stopwords, docs_content_fp),
        concurrency=probe.to_dict(),
    )
