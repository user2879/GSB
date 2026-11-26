from __future__ import annotations

from copy import deepcopy
from typing import Any

from .hashing import canonical_hash, composite_key
from .vocab import Vocabulary


def field_term_stats(terms: list[str]) -> dict[str, dict[str, Any]]:
    positions: dict[str, list[int]] = {}
    for position, term in enumerate(terms):
        positions.setdefault(term, []).append(position)
    return {
        term: {"tf": len(term_positions), "positions": term_positions}
        for term, term_positions in sorted(positions.items())
    }


def document_forward(document: dict[str, Any]) -> dict[str, Any]:
    forward: dict[str, Any] = {
        "doc_id": document["doc_id"],
        "version": document["version"],
        "document_fingerprint": document["document_fingerprint"],
        "views": {},
    }
    for view_name, view in document["views"].items():
        view_record: dict[str, Any] = {
            "language": view["language"],
            "fields": {},
            "term_stats": {},
        }
        for field_name, terms in view["fields"].items():
            stats = field_term_stats(terms)
            field_content_fingerprint = canonical_hash(
                {"field": field_name, "terms": terms, "stats": stats}
            )
            view_record["fields"][field_name] = list(terms)
            view_record["term_stats"][field_name] = stats
            view_record["fields"][f"{field_name}_fingerprint"] = field_content_fingerprint
        forward["views"][view_name] = view_record
    forward["corpus_document_fingerprint"] = canonical_hash(
        {"doc_id": forward["doc_id"], "version": forward["version"]}
    )
    return forward


def field_tf_records(
    forward: dict[str, Any], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    doc_id = forward["doc_id"]
    for view_name in config["view_names"]:
        view = forward["views"][view_name]
        for field_name in config["field_order"]:
            stats = view["term_stats"][field_name]
            content_fingerprint = view["fields"][f"{field_name}_fingerprint"]
            record_id = f"{doc_id}:{view_name}:{field_name}"
            records[record_id] = {
                "record_id": record_id,
                "doc_id": doc_id,
                "view": view_name,
                "field": field_name,
                "term_stats": stats,
                "content_fingerprint": content_fingerprint,
                "cache_key": composite_key(
                    "fieldtf", content_fingerprint, config["index_config_fingerprint"]
                ),
            }
    return records


def index_config_fingerprint(config: dict[str, Any]) -> str:
    return composite_key(
        "idxcfg",
        canonical_hash(
            {
                "views": config["views"],
                "field_order": config["field_order"],
                "vocab_fingerprint": config.get("vocab_fingerprint"),
                "schema_version": config.get("schema_version", "1.0"),
            }
        ),
    )


def bucket_id(view: str, term_id: str) -> str:
    return f"{view}:{term_id}"


def _posting_for_doc(
    document: dict[str, Any], forward: dict[str, Any], view_name: str, term_id: str
) -> dict[str, Any] | None:
    fields: dict[str, Any] = {}
    total_tf = 0
    view = forward["views"][view_name]
    for field_name, stats in view["term_stats"].items():
        if term_id in stats:
            stat = stats[term_id]
            fields[field_name] = {"tf": stat["tf"], "positions": list(stat["positions"])}
            total_tf += stat["tf"]
    if not fields:
        return None
    return {
        "doc_id": document["doc_id"],
        "version": document["version"],
        "term_frequency": total_tf,
        "fields": fields,
    }


def _recompute_bucket(
    bucket: dict[str, Any], config: dict[str, Any], vocab: Vocabulary
) -> dict[str, Any]:
    postings = sorted(bucket["postings"], key=lambda item: item["doc_id"])
    bucket["postings"] = postings
    bucket["document_frequency"] = len(postings)
    bucket["doc_ids"] = [posting["doc_id"] for posting in postings]
    term_id = bucket["term_id"]
    view_name = bucket["view"]
    bucket["surface_term"] = vocab.surface_form(term_id, view_name)
    fingerprint_material = {
        "term_id": term_id,
        "view": view_name,
        "postings": postings,
    }
    bucket["bucket_fingerprint"] = canonical_hash(fingerprint_material)
    bucket["cache_key"] = composite_key(
        "bucket", bucket["bucket_fingerprint"], config["index_config_fingerprint"]
    )
    return bucket


def empty_index(config: dict[str, Any], vocab: Vocabulary) -> dict[str, Any]:
    return {
        "forward": {},
        "field_tf": {},
        "buckets": {},
        "bucket_order": [],
        "corpus_fingerprint": "",
        "document_count": 0,
        "config_fingerprint": config["full_config_fingerprint"],
        "vocab_fingerprint": vocab.fingerprint,
    }


def corpus_fingerprint(forward: dict[str, dict[str, Any]]) -> str:
    rows = [
        {
            "doc_id": doc_id,
            "version": record["version"],
            "document_fingerprint": record["document_fingerprint"],
        }
        for doc_id, record in sorted(forward.items())
    ]
    return composite_key("corpus", canonical_hash(rows))


def _finalize_index(index: dict[str, Any]) -> dict[str, Any]:
    index["bucket_order"] = sorted(index["buckets"])
    index["document_count"] = len(index["forward"])
    index["corpus_fingerprint"] = corpus_fingerprint(index["forward"])
    return index


def build_index(
    documents: list[dict[str, Any]], config: dict[str, Any], vocab: Vocabulary
) -> dict[str, Any]:
    index = empty_index(config, vocab)
    for document in sorted(documents, key=lambda item: item["doc_id"]):
        _upsert_document_into_index(index, document, config, vocab)
    return _finalize_index(index)


def _remove_doc_from_buckets(
    index: dict[str, Any], doc_id: str
) -> tuple[set[str], set[str]]:
    affected: set[str] = set()
    removed: set[str] = set()
    old_forward = index["forward"].get(doc_id)
    if old_forward is None:
        return affected, removed
    candidate_bucket_ids: set[str] = set()
    for view_name, view in old_forward["views"].items():
        for field_stats in view.get("term_stats", {}).values():
            for term_id in field_stats:
                candidate_bucket_ids.add(bucket_id(view_name, term_id))
    for current_bucket_id in candidate_bucket_ids:
        bucket = index["buckets"].get(current_bucket_id)
        if bucket is None:
            continue
        affected.add(current_bucket_id)
        bucket["postings"] = [
            posting for posting in bucket["postings"] if posting["doc_id"] != doc_id
        ]
        if not bucket["postings"]:
            removed.add(current_bucket_id)
            del index["buckets"][current_bucket_id]
    return affected, removed


def _upsert_document_into_index(
    index: dict[str, Any],
    document: dict[str, Any],
    config: dict[str, Any],
    vocab: Vocabulary,
) -> dict[str, list[str]]:
    doc_id = document["doc_id"]
    old_affected: set[str] = set()
    removed_buckets: set[str] = set()
    if doc_id in index["forward"]:
        old_affected, removed_buckets = _remove_doc_from_buckets(index, doc_id)

    forward = document_forward(document)
    index["forward"][doc_id] = forward
    new_field_records = field_tf_records(forward, config)

    new_affected: set[str] = set()
    for view_name in config["view_names"]:
        for field_name in config["field_order"]:
            index["field_tf"].update(
                {
                    record_id: record
                    for record_id, record in new_field_records.items()
                    if record_id.endswith(f":{view_name}:{field_name}")
                }
            )
        for field_stats in forward["views"][view_name]["term_stats"].values():
            for term_id in field_stats:
                current_bucket_id = bucket_id(view_name, term_id)
                new_affected.add(current_bucket_id)
                bucket = index["buckets"].setdefault(
                    current_bucket_id,
                    {
                        "bucket_id": current_bucket_id,
                        "term_id": term_id,
                        "view": view_name,
                        "postings": [],
                    },
                )
                posting = _posting_for_doc(document, forward, view_name, term_id)
                if posting is not None:
                    bucket["postings"] = [
                        item for item in bucket["postings"] if item["doc_id"] != doc_id
                    ]
                    bucket["postings"].append(posting)

    affected = old_affected | new_affected
    recomputed: list[str] = []
    for current_bucket_id in sorted(affected):
        bucket = index["buckets"].get(current_bucket_id)
        if bucket is not None:
            _recompute_bucket(bucket, config, vocab)
            recomputed.append(current_bucket_id)
    _finalize_index(index)
    final_removed_buckets = sorted(
        current_bucket_id
        for current_bucket_id in removed_buckets
        if current_bucket_id not in index["buckets"]
    )
    return {
        "affected_buckets": sorted(affected),
        "recomputed_buckets": recomputed,
        "removed_buckets": final_removed_buckets,
        "field_tf_recomputed": sorted(new_field_records),
    }


def upsert_document(
    index: dict[str, Any],
    document: dict[str, Any],
    config: dict[str, Any],
    vocab: Vocabulary,
) -> dict[str, list[str]]:
    return _upsert_document_into_index(index, document, config, vocab)


def delete_document(
    index: dict[str, Any],
    doc_id: str,
    config: dict[str, Any],
    vocab: Vocabulary,
) -> dict[str, list[str]] | None:
    if doc_id not in index["forward"]:
        return None
    old_field_records = sorted(
        record_id
        for record_id in index["field_tf"]
        if index["field_tf"][record_id]["doc_id"] == doc_id
    )
    for record_id in old_field_records:
        del index["field_tf"][record_id]
    affected, removed = _remove_doc_from_buckets(index, doc_id)
    del index["forward"][doc_id]
    recomputed: list[str] = []
    for current_bucket_id in sorted(affected - removed):
        bucket = index["buckets"].get(current_bucket_id)
        if bucket is not None:
            _recompute_bucket(bucket, config, vocab)
            recomputed.append(current_bucket_id)
    _finalize_index(index)
    return {
        "affected_buckets": sorted(affected),
        "recomputed_buckets": recomputed,
        "removed_buckets": sorted(removed),
        "field_tf_recomputed": [],
        "field_tf_removed": old_field_records,
    }


def serialize_index(index: dict[str, Any]) -> dict[str, Any]:
    return {
        "forward": index["forward"],
        "field_tf": index["field_tf"],
        "buckets": {bucket_id: index["buckets"][bucket_id] for bucket_id in index["bucket_order"]},
        "bucket_order": index["bucket_order"],
        "corpus_fingerprint": index["corpus_fingerprint"],
        "document_count": index["document_count"],
        "config_fingerprint": index["config_fingerprint"],
        "vocab_fingerprint": index["vocab_fingerprint"],
    }


def clone_index(index: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(index)
