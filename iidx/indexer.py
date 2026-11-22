from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from .cache import CacheStore
from .compression import (
    DecodeError,
    block_fingerprint,
    compare_decoded_block,
    decode_block,
    encode_block,
)
from .config import AppConfig
from .loader import NormalizedDoc, QueryDef
from .util import failure, fixed_point_score, format_fixed, sha256_json


class ConcurrencyTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.maximum = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def exit(self) -> None:
        with self._lock:
            self.active -= 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"active": self.active, "max_observed": self.maximum}


def _run_parallel(tasks: Iterable[tuple[str, Any]], workers: int, tracker: ConcurrencyTracker) -> list[Any]:
    def wrapped(stable_key: str, function: Any) -> tuple[str, Any]:
        tracker.enter()
        try:
            return stable_key, function()
        finally:
            tracker.exit()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="iidx") as executor:
        futures = {
            stable_key: executor.submit(wrapped, stable_key, function)
            for stable_key, function in tasks
        }
        results = {stable_key: future.result() for stable_key, future in futures.items()}
    return [results[stable_key][1] for stable_key in sorted(results)]


def _build_shard_blocks(
    shard: str,
    docs: tuple[NormalizedDoc, ...],
    config: AppConfig,
    cache: CacheStore,
) -> dict[str, Any]:
    postings: dict[str, dict[str, list[int]]] = {}
    for doc in docs:
        for position, term in enumerate(doc.token_terms):
            postings.setdefault(term, {}).setdefault(doc.doc_id, []).append(position)
    sorted_terms = sorted(postings)
    blocks: list[dict[str, Any]] = []
    for local_block_no, start in enumerate(range(0, len(sorted_terms), config.chunk_size)):
        term_names = sorted_terms[start : start + config.chunk_size]
        terms: list[dict[str, Any]] = []
        unique_docs: set[str] = set()
        position_count = 0
        for term in term_names:
            term_postings = []
            for doc_id in sorted(postings[term]):
                positions = sorted(postings[term][doc_id])
                unique_docs.add(doc_id)
                position_count += len(positions)
                term_postings.append({"doc_id": doc_id, "positions": positions})
            terms.append(
                {
                    "term": term,
                    "doc_freq": len(term_postings),
                    "postings": term_postings,
                }
            )
        fingerprint = block_fingerprint(terms)
        content_cache_key = cache.block_key(shard, fingerprint, config.index_config_hash)
        cached = cache.lookup_block(content_cache_key)
        content_reused = False
        core_block: dict[str, Any] | None = None
        if cached is not None:
            cached_terms = cached.get("terms")
            if isinstance(cached_terms, list) and block_fingerprint(cached_terms) == fingerprint:
                core_block = {
                    "terms": cached_terms,
                    "doc_count": len(unique_docs),
                    "position_count": position_count,
                    "fingerprint": fingerprint,
                }
                content_reused = True
        if core_block is None:
            core_block = {
                "terms": terms,
                "doc_count": len(unique_docs),
                "position_count": position_count,
                "fingerprint": fingerprint,
            }
            cache.store_block(
                content_cache_key,
                {
                    "content_fingerprint": fingerprint,
                    "terms": terms,
                    "doc_count": len(unique_docs),
                    "position_count": position_count,
                },
            )
        blocks.append(
            {
                "shard": shard,
                "local_block_no": local_block_no,
                "terms": core_block["terms"],
                "doc_count": core_block["doc_count"],
                "position_count": core_block["position_count"],
                "fingerprint": fingerprint,
                "content_cache_key": content_cache_key,
                "content_reused": content_reused,
            }
        )
    return {
        "shard": shard,
        "blocks": blocks,
        "task_key": f"{shard}:00000000:build",
        "document_count": len(docs),
        "term_count": len(sorted_terms),
    }


def build_blocks(
    documents_by_shard: dict[str, tuple[NormalizedDoc, ...]],
    accepted_doc_ids: set[str],
    config: AppConfig,
    cache: CacheStore,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted_by_shard: dict[str, tuple[NormalizedDoc, ...]] = {}
    for shard in sorted(documents_by_shard):
        docs = tuple(doc for doc in documents_by_shard[shard] if doc.doc_id in accepted_doc_ids)
        if docs:
            accepted_by_shard[shard] = docs
    tracker = ConcurrencyTracker()
    tasks = [
        (
            f"{shard}:00000000:build",
            lambda shard=shard, docs=docs: _build_shard_blocks(shard, docs, config, cache),
        )
        for shard, docs in accepted_by_shard.items()
    ]
    shard_results = _run_parallel(tasks, config.thread_pool_size, tracker)
    blocks: list[dict[str, Any]] = []
    for shard_result in shard_results:
        blocks.extend(shard_result["blocks"])
    blocks.sort(key=lambda item: (item["shard"], item["local_block_no"]))
    for block_no, block in enumerate(blocks):
        block["block_no"] = block_no
    metrics = {
        "thread_pool_size": config.thread_pool_size,
        "concurrency_quota": config.concurrency_quota,
        "build": tracker.snapshot(),
        "shards": len(accepted_by_shard),
        "blocks": len(blocks),
    }
    return blocks, metrics


def _cached_decode_mismatch_failure(block_no: int, tier: int, location: str | None) -> dict[str, Any]:
    return failure(
        "compression",
        "E_DECODE_MISMATCH",
        f"block:{block_no:08d}:tier:{tier}",
        "cached compressed artifact failed decode verification; the block is re-encoded",
        block_no=block_no,
        tier=tier,
        first_mismatch_location=location,
    )


def compress_blocks(
    blocks: list[dict[str, Any]],
    config: AppConfig,
    cache: CacheStore,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    tracker = ConcurrencyTracker()
    failures: list[dict[str, Any]] = []
    decisions: dict[tuple[int, int], dict[str, Any]] = {}
    for block in blocks:
        tier, reason, matched = config.selected_tier(block["block_no"], sum(
            entry["doc_freq"] for entry in block["terms"]
        ))
        decisions[(block["block_no"], tier)] = {"reason": reason, "matched_switch_point": matched}

    def compress_one(block: dict[str, Any], tier: int) -> dict[str, Any]:
        block_no = block["block_no"]
        encoding_key = cache.encoding_key(
            block["content_cache_key"],
            block["fingerprint"],
            tier,
            config.encoding_parameters_hash(tier),
        )
        cached = cache.lookup_encoding(encoding_key)
        data: bytes | None = None
        reused = False
        decode_stats: dict[str, Any] | None = None
        decoded_block: dict[str, Any] | None = None
        if cached is not None:
            cached_metadata, cached_data = cached
            try:
                candidate_block, candidate_stats = decode_block(cached_data)
                ok, location, _, _ = compare_decoded_block(block, candidate_block)
                if ok and candidate_stats.get("fingerprint_matches"):
                    data = cached_data
                    decoded_block = candidate_block
                    decode_stats = candidate_stats
                    reused = True
                else:
                    local_failures.append(
                        _cached_decode_mismatch_failure(block_no, tier, location)
                    )
            except DecodeError as exc:
                local_failures.append(
                    _cached_decode_mismatch_failure(block_no, tier, f"offset:{exc.offset}")
                )
        if data is None:
            cache.increment_encoding_calls()
            data, encoded_stats = encode_block(block, tier, config.low_frequency_threshold)
            decoded_block, decode_stats = decode_block(data)
            ok, location, left, right = compare_decoded_block(block, decoded_block)
            if not ok or not decode_stats.get("fingerprint_matches"):
                failures.append(
                    failure(
                        "compression",
                        "E_DECODE_MISMATCH",
                        f"block:{block_no:08d}:tier:{tier}",
                        "fresh compression roundtrip mismatched the source block",
                        block_no=block_no,
                        tier=tier,
                        first_mismatch_location=location,
                        expected=left,
                        actual=right,
                    )
                )
            cache.store_encoding(
                encoding_key,
                {
                    "block_key": block["content_cache_key"],
                    "block_fingerprint": block["fingerprint"],
                    "tier": tier,
                    "encoding_parameters_hash": config.encoding_parameters_hash(tier),
                    "stats": encoded_stats,
                },
                data,
            )
        assert decoded_block is not None and decode_stats is not None
        return {
            "block_no": block_no,
            "shard": block["shard"],
            "local_block_no": block["local_block_no"],
            "tier": tier,
            "data": data,
            "encoding_cache_key": encoding_key,
            "reused": reused,
            "encoded_bytes": len(data),
            "decoded": decoded_block,
            "decode_stats": decode_stats,
            "decode_ok": True,
            "decoded_checksum": decode_stats["decoded_checksum"],
            "filtered_count": decode_stats["filtered_terms"],
            "task_key": f"{block['shard']}:{block_no:08d}:compress:tier-{tier}",
        }

    local_failures: list[dict[str, Any]] = []
    tasks = [
        (
            f"{block['shard']}:{block['block_no']:08d}:compress:tier-{tier}",
            lambda block=block, tier=tier: compress_one(block, tier),
        )
        for block in blocks
        for tier in (1, 2)
    ]
    results = _run_parallel(tasks, config.thread_pool_size, tracker)
    failures.extend(local_failures)
    by_block: dict[int, dict[int, dict[str, Any]]] = {}
    for result in results:
        by_block.setdefault(result["block_no"], {})[result["tier"]] = result

    block_records: list[dict[str, Any]] = []
    retrieval_blocks: dict[int, dict[str, Any]] = {}
    compression_decisions: list[dict[str, Any]] = []
    for block in blocks:
        selected_tier, reason, matched = config.selected_tier(
            block["block_no"], sum(entry["doc_freq"] for entry in block["terms"])
        )
        tier_results = by_block[block["block_no"]]
        length1 = tier_results[1]["encoded_bytes"]
        length2 = tier_results[2]["encoded_bytes"]
        lengths_differ = length1 != length2
        if not lengths_differ:
            failures.append(
                failure(
                    "compression",
                    "E_COMPRESSION_LENGTH_COLLISION",
                    f"block:{block['block_no']:08d}",
                    "tier-1 and tier-2 encodings must have different byte lengths",
                    block_no=block["block_no"],
                    tier1_bytes=length1,
                    tier2_bytes=length2,
                )
            )
        selected = tier_results[selected_tier]
        retrieval_blocks[block["block_no"]] = {
            **block,
            "terms": selected["decoded"]["terms"],
            "compression_tier": selected_tier,
        }
        block_records.append(
            {
                "block_no": block["block_no"],
                "shard": block["shard"],
                "local_block_no": block["local_block_no"],
                "terms": block["terms"],
                "doc_count": block["doc_count"],
                "position_count": block["position_count"],
                "fingerprint": block["fingerprint"],
                "content_cache_key": block["content_cache_key"],
                "content_reused": block["content_reused"],
                "compression": {
                    "selected_tier": selected_tier,
                    "selection_reason": reason,
                    "matched_switch_point": matched,
                    "tier1": {
                        "encoded_bytes": length1,
                        "encoding_cache_key": tier_results[1]["encoding_cache_key"],
                        "reused": tier_results[1]["reused"],
                        "decoded_checksum": tier_results[1]["decoded_checksum"],
                        "filtered_count": tier_results[1]["filtered_count"],
                        "decode_ok": tier_results[1]["decode_ok"],
                    },
                    "tier2": {
                        "encoded_bytes": length2,
                        "encoding_cache_key": tier_results[2]["encoding_cache_key"],
                        "reused": tier_results[2]["reused"],
                        "decoded_checksum": tier_results[2]["decoded_checksum"],
                        "filtered_count": tier_results[2]["filtered_count"],
                        "decode_ok": tier_results[2]["decode_ok"],
                    },
                    "selected_encoded_bytes": selected["encoded_bytes"],
                    "selected_encoding_cache_key": selected["encoding_cache_key"],
                    "selected_reused": selected["reused"],
                    "decoded_checksum": selected["decoded_checksum"],
                    "filtered_count": selected["filtered_count"],
                    "decode_ok": selected["decode_ok"],
                    "lengths_differ": lengths_differ,
                },
            }
        )
        compression_decisions.append(
            {
                "block_no": block["block_no"],
                "shard": block["shard"],
                "selected_tier": selected_tier,
                "reason": reason,
                "matched_switch_point": matched,
                "tier1_bytes": length1,
                "tier2_bytes": length2,
                "byte_length_delta": length2 - length1,
                "filtered_count": selected["filtered_count"],
            }
        )
    metrics = {
        "compression": tracker.snapshot(),
        "encodings": len(results),
        "reused_encodings": sum(1 for item in results if item["reused"]),
        "fresh_encodings": sum(1 for item in results if not item["reused"]),
        "cache_metrics": cache.snapshot_metrics(),
    }
    return block_records, metrics, retrieval_blocks, failures


def run_queries(
    queries: dict[str, QueryDef],
    retrieval_blocks: dict[int, dict[str, Any]],
    documents: dict[str, NormalizedDoc],
    config: AppConfig,
    cache: CacheStore,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    term_to_blocks: dict[str, list[int]] = {}
    for block_no, block in retrieval_blocks.items():
        for entry in block["terms"]:
            term_to_blocks.setdefault(entry["term"], []).append(block_no)
    for term in term_to_blocks:
        term_to_blocks[term] = sorted(set(term_to_blocks[term]))

    tracker = ConcurrencyTracker()

    def evaluate(query: QueryDef) -> dict[str, Any]:
        referenced = sorted(
            {
                block_no
                for term in query.terms
                for block_no in term_to_blocks.get(term, [])
            }
        )
        referenced_payload = [
            {
                "block_no": block_no,
                "shard": retrieval_blocks[block_no]["shard"],
                "fingerprint": retrieval_blocks[block_no]["fingerprint"],
                "tier": retrieval_blocks[block_no]["compression_tier"],
            }
            for block_no in referenced
        ]
        document_lengths: dict[str, int] = {}
        for block_no in referenced:
            for entry in retrieval_blocks[block_no]["terms"]:
                if entry["term"] not in query.terms:
                    continue
                for posting in entry["postings"]:
                    doc = documents.get(posting["doc_id"])
                    if doc is not None:
                        document_lengths[doc.doc_id] = doc.token_count
        cache_key = cache.query_key(
            query.as_dict(),
            referenced_payload,
            config.query_config_hash,
            document_lengths,
        )
        cached = cache.lookup_query(cache_key)
        if cached is not None:
            cached_result = cached.get("result")
            if (
                isinstance(cached_result, dict)
                and cached_result.get("query_id") == query.query_id
                and cached_result.get("query_terms") == list(query.terms)
                and cached_result.get("query_stream_version") == query.version
                and cached_result.get("top_k") == query.top_k
                and cached_result.get("referenced_blocks") == referenced
                and cached_result.get("result_checksum")
                == sha256_json(cached_result.get("results", []))
            ):
                return {
                    **cached_result,
                    "cache_key": cache_key,
                    "reused": True,
                    "task_key": f"query:{query.query_id}:recall",
                }
        scores: dict[str, int] = {}
        matched_terms: dict[str, set[str]] = {}
        query_terms = set(query.terms)
        for block_no in referenced:
            block = retrieval_blocks[block_no]
            for entry in block["terms"]:
                term = entry["term"]
                if term not in query_terms:
                    continue
                for posting in entry["postings"]:
                    doc_id = posting["doc_id"]
                    doc = documents.get(doc_id)
                    if doc is None:
                        continue
                    tf = len(posting["positions"])
                    component = (tf * 1_000_000) // (tf + doc.token_count + 1)
                    scores[doc_id] = scores.get(doc_id, 0) + component
                    matched_terms.setdefault(doc_id, set()).add(term)
        quantum = config.quantization_quantum
        ranked: list[dict[str, Any]] = []
        for doc_id, raw_score in scores.items():
            score_fixed = fixed_point_score(raw_score, config.fixed_decimal_bits, quantum)
            doc = documents[doc_id]
            ranked.append(
                {
                    "doc_id": doc_id,
                    "shard": doc.shard,
                    "score_fixed": score_fixed,
                    "score": format_fixed(score_fixed, config.fixed_decimal_bits),
                    "matched_terms": sorted(matched_terms[doc_id]),
                }
            )
        ranked.sort(key=lambda item: (-item["score_fixed"], item["doc_id"]))
        top_results = ranked[: query.top_k]
        result_payload = {
            "query_id": query.query_id,
            "query_terms": list(query.terms),
            "query_stream_version": query.version,
            "top_k": query.top_k,
            "results": top_results,
            "recall_count": len(ranked),
            "referenced_blocks": referenced,
            "fixed_point": {
                "bits": config.fixed_decimal_bits,
                "quantization_tier": config.quantization_tier,
                "quantum": quantum,
            },
            "result_checksum": sha256_json(top_results),
        }
        cache.store_query(cache_key, {"result": result_payload})
        return {
            **result_payload,
            "cache_key": cache_key,
            "reused": False,
            "task_key": f"query:{query.query_id}:recall",
        }

    tasks = [
        (f"query:{query_id}:recall", lambda query=queries[query_id]: evaluate(query))
        for query_id in sorted(queries)
    ]
    results = _run_parallel(tasks, config.thread_pool_size, tracker)
    results.sort(key=lambda item: item["query_id"])
    metrics = {
        "query": tracker.snapshot(),
        "queries": len(results),
        "reused_queries": sum(1 for item in results if item["reused"]),
        "fresh_queries": sum(1 for item in results if not item["reused"]),
        "cache_metrics": cache.snapshot_metrics(),
    }
    return results, metrics


def build_invalidation_ledger(
    previous_report: dict[str, Any] | None,
    blocks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    previous_blocks = previous_report.get("blocks", []) if previous_report else []
    previous_queries = previous_report.get("queries", []) if previous_report else []
    previous_block_by_key = {
        (item.get("shard"), item.get("fingerprint")): item
        for item in previous_blocks
        if isinstance(item, dict)
    }
    previous_block_numbers = {
        (item.get("shard"), item.get("local_block_no")): item
        for item in previous_blocks
        if isinstance(item, dict)
    }
    reused_blocks: list[dict[str, Any]] = []
    recomputed_blocks: list[dict[str, Any]] = []
    for block in sorted(blocks, key=lambda item: item["block_no"]):
        common = {
            "block_no": block["block_no"],
            "shard": block["shard"],
            "local_block_no": block["local_block_no"],
            "fingerprint": block["fingerprint"],
            "content_cache_key": block["content_cache_key"],
        }
        if block.get("content_reused"):
            previous = previous_block_by_key.get((block["shard"], block["fingerprint"]))
            reused_blocks.append(
                {
                    **common,
                    "previous_block_no": previous.get("block_no") if previous else None,
                }
            )
        else:
            previous = previous_block_numbers.get((block["shard"], block["local_block_no"]))
            if previous is None:
                reason = "new_block_or_layout_changed"
            elif previous.get("fingerprint") != block["fingerprint"]:
                reason = "content_changed"
            else:
                reason = "cache_missing"
            recomputed_blocks.append({**common, "reason": reason})
    new_logical = {(item.get("shard"), item.get("local_block_no")) for item in blocks}
    new_fingerprints = {(item.get("shard"), item.get("fingerprint")) for item in blocks}
    invalidated_blocks: list[dict[str, Any]] = []
    for old in sorted(previous_blocks, key=lambda item: int(item.get("block_no", 0))):
        logical = (old.get("shard"), old.get("local_block_no"))
        fingerprint = (old.get("shard"), old.get("fingerprint"))
        if fingerprint not in new_fingerprints or logical not in new_logical:
            invalidated_blocks.append(
                {
                    "block_no": old.get("block_no"),
                    "shard": old.get("shard"),
                    "local_block_no": old.get("local_block_no"),
                    "fingerprint": old.get("fingerprint"),
                    "reason": "removed_or_content_changed",
                }
            )

    previous_query_by_id = {
        item.get("query_id"): item for item in previous_queries if isinstance(item, dict)
    }
    reused_queries: list[dict[str, Any]] = []
    recomputed_queries: list[dict[str, Any]] = []
    for query in sorted(queries, key=lambda item: item["query_id"]):
        previous = previous_query_by_id.get(query["query_id"])
        common = {
            "query_id": query["query_id"],
            "cache_key": query["cache_key"],
            "referenced_blocks": query["referenced_blocks"],
        }
        if query.get("reused"):
            reused_queries.append(common)
        else:
            if previous is None:
                reason = "new_query"
            elif previous.get("referenced_blocks") != query.get("referenced_blocks"):
                reason = "referenced_blocks_changed"
            else:
                reason = "query_or_compression_config_changed"
            recomputed_queries.append({**common, "reason": reason})
    current_query_ids = {item["query_id"] for item in queries}
    invalidated_queries = [
        {
            "query_id": item.get("query_id"),
            "cache_key": item.get("cache_key"),
            "reason": "query_removed_or_cache_key_changed",
        }
        for item in sorted(previous_queries, key=lambda entry: str(entry.get("query_id")))
        if item.get("query_id") not in current_query_ids
        or item.get("cache_key") != next(
            (new.get("cache_key") for new in queries if new["query_id"] == item.get("query_id")), None
        )
    ]
    total_blocks = max(1, len(blocks))
    total_queries = max(1, len(queries))
    return {
        "reused_blocks": reused_blocks,
        "recomputed_blocks": recomputed_blocks,
        "invalidated_blocks": invalidated_blocks,
        "reused_queries": reused_queries,
        "recomputed_queries": recomputed_queries,
        "invalidated_queries": invalidated_queries,
        "counts": {
            "reused_blocks": len(reused_blocks),
            "recomputed_blocks": len(recomputed_blocks),
            "invalidated_blocks": len(invalidated_blocks),
            "reused_queries": len(reused_queries),
            "recomputed_queries": len(recomputed_queries),
            "invalidated_queries": len(invalidated_queries),
            "block_reuse_ratio": f"{len(reused_blocks) / total_blocks:.6f}",
            "query_reuse_ratio": f"{len(reused_queries) / total_queries:.6f}",
        },
        "provenance_note": (
            "Cache provenance depends on the prior state; canonical index/query/ledger projections are "
            "the fields compared against a clean full rebuild."
        ),
    }


def canonical_projection(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": report.get("config"),
        "summary": {
            "accepted_documents": report.get("summary", {}).get("accepted_documents"),
            "accepted_queries": report.get("summary", {}).get("accepted_queries"),
            "completed_shards": report.get("summary", {}).get("completed_shards"),
            "partial_shards": report.get("summary", {}).get("partial_shards"),
            "pending_shards": report.get("summary", {}).get("pending_shards"),
            "quota_exhausted_shards": report.get("summary", {}).get("quota_exhausted_shards"),
            "block_count": report.get("summary", {}).get("block_count"),
            "selected_tier_counts": report.get("summary", {}).get("selected_tier_counts"),
            "consumed_token_budget": report.get("summary", {}).get("consumed_token_budget"),
            "remaining_token_budget": report.get("summary", {}).get("remaining_token_budget"),
            "pending_queue": report.get("summary", {}).get("pending_queue"),
            "quota_exhausted_queue": report.get("summary", {}).get("quota_exhausted_queue"),
        },
        "blocks": [
            {
                "block_no": item["block_no"],
                "shard": item["shard"],
                "local_block_no": item["local_block_no"],
                "terms": item["terms"],
                "doc_count": item["doc_count"],
                "position_count": item["position_count"],
                "fingerprint": item["fingerprint"],
                "content_cache_key": item["content_cache_key"],
                "compression": {
                    "selected_tier": item["compression"]["selected_tier"],
                    "selection_reason": item["compression"]["selection_reason"],
                    "matched_switch_point": item["compression"]["matched_switch_point"],
                    "tier1_bytes": item["compression"]["tier1"]["encoded_bytes"],
                    "tier2_bytes": item["compression"]["tier2"]["encoded_bytes"],
                    "tier1_encoding_cache_key": item["compression"]["tier1"]["encoding_cache_key"],
                    "tier2_encoding_cache_key": item["compression"]["tier2"]["encoding_cache_key"],
                    "selected_encoding_cache_key": item["compression"]["selected_encoding_cache_key"],
                    "decoded_checksum": item["compression"]["decoded_checksum"],
                    "filtered_count": item["compression"]["filtered_count"],
                    "decode_ok": item["compression"]["decode_ok"],
                },
            }
            for item in sorted(report.get("blocks", []), key=lambda entry: entry["block_no"])
        ],
        "queries": [
            {
                "query_id": item["query_id"],
                "query_terms": item["query_terms"],
                "query_stream_version": item["query_stream_version"],
                "cache_key": item["cache_key"],
                "top_k": item["top_k"],
                "results": item["results"],
                "referenced_blocks": item["referenced_blocks"],
                "result_checksum": item["result_checksum"],
            }
            for item in sorted(report.get("queries", []), key=lambda entry: entry["query_id"])
        ],
        "compression_decisions": report.get("compression_decisions", []),
        "budget_ledger": report.get("budget_ledger", []),
        "failures": report.get("failures", []),
    }


def canonical_digest(report: dict[str, Any]) -> str:
    return sha256_json(canonical_projection(report))
