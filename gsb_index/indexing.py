"""Deterministic shard-side inverted index construction."""

from __future__ import annotations

import base64
import hashlib
from collections import Counter
from typing import Any, Iterable

from .canonical import fingerprint
from .models import CompressionProfile, Document, ShardSpec, Vocabulary


def term_bucket(term: str, bucket_count: int) -> int:
    digest = hashlib.sha256(term.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % bucket_count


def indexed_fields(document: Document, vocabulary: Vocabulary) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for field_name in ("title", "body"):
        terms = document.fields[field_name]
        if document.field_stopwords[field_name]:
            terms = tuple(term for term in terms if term not in vocabulary.stopwords)
        result[field_name] = terms
    return result


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint input must be non-negative")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _zigzag(value: int) -> int:
    return (value << 1) if value >= 0 else ((-value << 1) - 1)


def _encode_int_stream(values: Iterable[int], encoding: str) -> tuple[str, int]:
    numbers = [int(value) for value in values]
    if encoding in {"plain-int-v1", "plain"}:
        payload = "|".join(str(value) for value in numbers).encode("ascii")
    elif encoding in {"zigzag-varint-v1", "varint"}:
        payload = b"".join(_varint(_zigzag(value)) for value in numbers)
    elif encoding in {"delta-int-v1", "delta"}:
        deltas: list[int] = []
        previous = 0
        for value in numbers:
            deltas.append(value - previous)
            previous = value
        payload = b"".join(_varint(_zigzag(value)) for value in deltas)
    elif encoding in {"bucket-fixed-v1", "fixed-int-v1"}:
        payload = b"".join(value.to_bytes(8, "big", signed=True) for value in numbers)
    else:
        # Unknown encodings are still deterministic and part of the content
        # fingerprint, rather than silently falling back to an incompatible one.
        text = f"{encoding}:" + "|".join(str(value) for value in numbers)
        payload = text.encode("utf-8")
    return base64.b64encode(payload).decode("ascii"), len(payload)


def _bucket_fingerprint(bucket: dict[str, Any]) -> str:
    return fingerprint(
        {
            "bucket_index": bucket["bucket_index"],
            "weight": bucket["weight"],
            "encoding": bucket["encoding"],
            "terms": bucket["terms"],
        }
    )


def bucket_fingerprint(bucket: dict[str, Any]) -> str:
    """Public wrapper used when mixing reused and recomputed buckets."""

    return _bucket_fingerprint(bucket)


def shard_fingerprint(
    shard_id: str,
    revision: int,
    profile: CompressionProfile,
    buckets: Iterable[dict[str, Any]],
    documents: Iterable[Document],
) -> str:
    return fingerprint(
        {
            "shard_id": shard_id,
            "revision": revision,
            "compression_fingerprint": fingerprint(profile.to_dict()),
            "documents": sorted(
                ({"doc_id": doc.doc_id, "version": doc.version, "fingerprint": fingerprint(doc.to_dict())} for doc in documents),
                key=lambda item: (item["doc_id"], item["version"]),
            ),
            "buckets": sorted(
                (
                    {
                        "bucket_index": bucket["bucket_index"],
                        "weight": bucket["weight"],
                        "encoding": bucket["encoding"],
                        "fingerprint": bucket.get("fingerprint") or _bucket_fingerprint(bucket),
                    }
                    for bucket in buckets
                ),
                key=lambda item: item["bucket_index"],
            ),
        }
    )


def compute_shard_summary(
    shard: ShardSpec,
    profile: CompressionProfile,
    vocabulary: Vocabulary,
    documents: Iterable[Document],
    revision: int,
    *,
    entry_limit: int | None = None,
) -> dict[str, Any]:
    docs = sorted(tuple(documents), key=lambda doc: (doc.doc_id, doc.version))
    if len({doc.doc_id for doc in docs}) != len(docs):
        raise ValueError("duplicate doc_id in shard batch")

    bucket_count = profile.bucket_count
    capacity_per_bucket = shard.term_bucket_capacity // bucket_count
    stats: dict[int, dict[str, dict[str, Any]]] = {
        index: {} for index in range(bucket_count)
    }
    for document in docs:
        fields = indexed_fields(document, vocabulary)
        title_counts = Counter(fields["title"])
        body_counts = Counter(fields["body"])
        document_terms = set(title_counts) | set(body_counts)
        for term in sorted(document_terms):
            bucket_index = term_bucket(term, bucket_count)
            term_stats = stats[bucket_index].setdefault(
                term,
                {
                    "term": term,
                    "title_tf": 0,
                    "body_tf": 0,
                    "title_df": 0,
                    "body_df": 0,
                    "df": 0,
                    "postings": [],
                },
            )
            title_tf = int(title_counts.get(term, 0))
            body_tf = int(body_counts.get(term, 0))
            if title_tf < 0 or body_tf < 0:
                raise ValueError("negative term frequency")
            term_stats["title_tf"] += title_tf
            term_stats["body_tf"] += body_tf
            term_stats["title_df"] += int(title_tf > 0)
            term_stats["body_df"] += int(body_tf > 0)
            term_stats["df"] += 1
            term_stats["postings"].append(
                {
                    "doc_id": document.doc_id,
                    "version": document.version,
                    "title_tf": title_tf,
                    "body_tf": body_tf,
                }
            )

    buckets: list[dict[str, Any]] = []
    entries_count = 0
    for bucket_index in range(bucket_count):
        term_items = []
        for term in sorted(stats[bucket_index]):
            item = stats[bucket_index][term]
            item["postings"] = sorted(item["postings"], key=lambda posting: (posting["doc_id"], posting["version"]))
            term_items.append(item)
        if len(term_items) > capacity_per_bucket:
            raise OverflowError(
                f"bucket {bucket_index} has {len(term_items)} terms, capacity is {capacity_per_bucket}"
            )
        entries_count += len(term_items)
        values: list[int] = []
        for item in term_items:
            values.extend(
                [item["title_tf"], item["body_tf"], item["title_df"], item["body_df"], item["df"]]
            )
        encoded_terms, encoded_bytes = _encode_int_stream(values, profile.encoding)
        bucket = {
            "bucket_index": bucket_index,
            "weight": float(profile.bucket_weights[bucket_index]),
            "encoding": profile.encoding,
            "term_count": len(term_items),
            "entry_count": len(term_items),
            "terms": term_items,
            "encoded_terms": encoded_terms,
            "encoded_bytes": encoded_bytes,
        }
        bucket["fingerprint"] = _bucket_fingerprint(bucket)
        buckets.append(bucket)

    if entry_limit is not None and entries_count > entry_limit:
        limit_error = OverflowError(
            f"shard {shard.shard_id} requires {entries_count} entries, limit is {entry_limit}"
        )
        limit_error.required_entries = entries_count  # type: ignore[attr-defined]
        limit_error.entry_limit = entry_limit  # type: ignore[attr-defined]
        raise limit_error

    document_fingerprints = sorted(
        (
            {"doc_id": doc.doc_id, "version": doc.version, "fingerprint": fingerprint(doc.to_dict())}
            for doc in docs
        ),
        key=lambda item: (item["doc_id"], item["version"]),
    )
    summary = {
        "shard_id": shard.shard_id,
        "revision": revision,
        "doc_count": len(docs),
        "bucket_count": bucket_count,
        "bucket_weights": [float(weight) for weight in profile.bucket_weights],
        "encoding": profile.encoding,
        "compression_fingerprint": fingerprint(profile.to_dict()),
        "buckets": buckets,
        "entries_count": entries_count,
        "document_fingerprints": document_fingerprints,
    }
    summary["shard_fingerprint"] = shard_fingerprint(
        shard.shard_id,
        revision,
        profile,
        buckets,
        docs,
    )
    return summary


def summarize_document_terms(
    documents: Iterable[Document], vocabulary: Vocabulary, bucket_count: int
) -> dict[str, set[int]]:
    """Return bucket indices touched by each document under a profile."""

    result: dict[str, set[int]] = {}
    for document in documents:
        fields = indexed_fields(document, vocabulary)
        terms = set(fields["title"]) | set(fields["body"])
        result[document.doc_id] = {term_bucket(term, bucket_count) for term in terms}
    return result


def estimate_entry_count(
    documents: Iterable[Document], vocabulary: Vocabulary, bucket_count: int
) -> int:
    """Count unique terms after per-field stopword filtering."""

    terms: set[str] = set()
    for document in documents:
        fields = indexed_fields(document, vocabulary)
        terms.update(fields["title"])
        terms.update(fields["body"])
    return len(terms)
