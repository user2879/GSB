from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config import RuntimeConfig
from .model import CandidateRow, Shard
from .util import stable_hash


def _authority_key(row: CandidateRow) -> tuple[float, int, int, int]:
    return (-row.score, -row.shard_seq, -row.coverage_start_ms, -row.line_no)


def _occurrence(row: CandidateRow, shard: Shard) -> dict[str, Any]:
    return {
        "shard_id": shard.shard_id,
        "shard_seq": shard.shard_seq,
        "coverage_start_ms": shard.coverage_start_ms,
        "coverage_end_ms": shard.coverage_end_ms,
        "score": row.score,
        "line_no": row.line_no,
        "shard_fingerprint": shard.fingerprint,
    }


def merge_candidates(
    shards: list[Shard],
    cfg: RuntimeConfig,
    previous_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Merge and deduplicate within request+channel, retaining cross-channel copies."""
    previous = previous_cache or {}
    scan_rows: list[tuple[CandidateRow, Shard, int, str]] = []
    scan_index = 0
    for shard in sorted(
        shards,
        key=lambda item: (
            item.channel,
            item.coverage_start_ms,
            item.coverage_end_ms,
            item.shard_seq,
            item.source,
        ),
    ):
        for row in sorted(
            shard.rows,
            key=lambda item: (
                item.request_id,
                item.line_no,
                item.item_id,
            ),
        ):
            scan_rows.append((row, shard, scan_index, row.request_id))
            scan_index += 1

    grouped: dict[tuple[str, str, str], list[tuple[CandidateRow, Shard, int]]] = defaultdict(list)
    for row, shard, index, _request in scan_rows:
        grouped[(row.request_id, row.channel, row.item_id)].append((row, shard, index))

    cache: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    reuse_count = 0
    compute_count = 0
    for (request_id, channel, item_id), occurrences in grouped.items():
        occurrences.sort(key=lambda item: _authority_key(item[0]))
        authority_row, authority_shard, _ = occurrences[0]
        references = sorted(
            {
                (shard.shard_id, shard.fingerprint, shard.coverage_start_ms, shard.coverage_end_ms)
                for _row, shard, _index in occurrences
            },
            key=lambda item: (item[2], item[3], item[0]),
        )
        first = min(occurrences, key=lambda item: item[2])
        key_payload = {
            "stage": "merge",
            "request_id": request_id,
            "channel": channel,
            "item_id": item_id,
            "alignment_ms": cfg.alignment_ms,
            "references": references,
            "rows": sorted(
                [
                    {
                        **row.semantic_dict(),
                        "line_no": row.line_no,
                        "shard_id": shard.shard_id,
                        "shard_fingerprint": shard.fingerprint,
                    }
                    for row, shard, _index in occurrences
                ],
                key=lambda item: stable_hash(item),
            ),
        }
        cache_key = stable_hash(key_payload)
        if cache_key in previous:
            record = previous[cache_key]
            reuse_count += 1
        else:
            overridden = []
            for row, shard, _index in occurrences[1:]:
                overlaps_authority = (
                    shard.coverage_start_ms < authority_shard.coverage_end_ms
                    and authority_shard.coverage_start_ms < shard.coverage_end_ms
                )
                overridden.append(
                    {
                        **_occurrence(row, shard),
                        "overlaps_authority": overlaps_authority,
                    }
                )
            record = {
                "cache_key": cache_key,
                "request_id": request_id,
                "channel": channel,
                "item_id": item_id,
                "authority": {
                    **_occurrence(authority_row, authority_shard),
                    "features": list(authority_row.features),
                    "request_id": request_id,
                    "channel": channel,
                    "item_id": item_id,
                },
                "occurrences_count": len(occurrences),
                "max_score": max(row.score for row, _shard, _index in occurrences),
                "overridden": overridden,
                "references": {
                    "shards": [entry[0] for entry in references],
                    "channels": [channel],
                    "shard_fingerprints": {
                        entry[0]: entry[1] for entry in references
                    },
                },
                "order": {
                    "channel": first[1].channel,
                    "coverage_start_ms": first[1].coverage_start_ms,
                    "coverage_end_ms": first[1].coverage_end_ms,
                    "shard_seq": first[1].shard_seq,
                    "line_no": first[0].line_no,
                },
            }
            compute_count += 1
        cache[cache_key] = record
        records.append(record)

    records.sort(
        key=lambda record: (
            record["request_id"],
            record["order"]["channel"],
            record["order"]["coverage_start_ms"],
            record["order"]["coverage_end_ms"],
            record["order"]["shard_seq"],
            record["order"]["line_no"],
            record["item_id"],
            record["channel"],
        )
    )
    telemetry = {
        "computed": compute_count,
        "reused": reuse_count,
        "active_keys": sorted(cache),
    }
    return records, cache, telemetry


def group_by_request_item(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["request_id"], record["item_id"])].append(record)
    for values in grouped.values():
        values.sort(key=lambda record: record["channel"])
    return dict(grouped)
