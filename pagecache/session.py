from __future__ import annotations

from typing import Any

from .aggregate import FixedPointAggregator
from .cache import PageCache
from .canonical import canonical_bytes, canonical_json, sha256_hex


def checkpoint_record(
    *,
    access_index: int,
    page_id: str,
    cache: PageCache,
    aggregator: FixedPointAggregator,
) -> dict[str, Any]:
    cache_state = cache.state()
    aggregation_state = aggregator.state()
    return {
        "access_index": access_index,
        "page_id": page_id,
        "overflow_sequence": list(cache_state["overflow_sequence"]),
        "swap_in_sequence": list(cache_state["swap_in_sequence"]),
        "swap_in_count": int(cache_state["swap_in_count"]),
        "resident_page_ids": [row["page_id"] for row in cache_state["resident_pages"]],
        "cache_state_sha256": sha256_hex(canonical_json(cache_state).encode("utf-8")),
        "aggregation_state": aggregation_state,
        "aggregation_state_sha256": sha256_hex(canonical_json(aggregation_state).encode("utf-8")),
    }


def run_reference(
    *,
    cache: PageCache,
    aggregator: FixedPointAggregator,
    scan_order: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    for index, page_id in enumerate(scan_order):
        page = cache.get_page(page_id)
        aggregator.consume_page(page)
        trace.append(checkpoint_record(access_index=index, page_id=page_id, cache=cache, aggregator=aggregator))
    output = aggregator.output()
    output_bytes = canonical_bytes(output)
    return {
        "trace": trace,
        "final_output": output,
        "final_output_sha256": sha256_hex(output_bytes),
        "final_cache_state": cache.state(),
        "final_aggregation_state": aggregator.state(),
    }


def run_prefix(
    *,
    cache: PageCache,
    aggregator: FixedPointAggregator,
    scan_order: list[str] | tuple[str, ...],
    position: int,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for index, page_id in enumerate(scan_order[:position]):
        page = cache.get_page(page_id)
        aggregator.consume_page(page)
        trace.append(checkpoint_record(access_index=index, page_id=page_id, cache=cache, aggregator=aggregator))
    return trace
