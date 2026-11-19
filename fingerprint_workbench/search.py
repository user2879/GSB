from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from .audio import audio_content_hash, read_wav_bytes
from .config import BudgetConfig, SAMPLE_RATE
from .errors import WorkbenchError
from .fingerprint import extract_fingerprint
from .index import shard_id_for_hash
from .workbench import Workbench


EXACT_WEIGHT_NUMERATOR = 5
EXACT_WEIGHT_DENOMINATOR = 4
ROBUST_WEIGHT_NUMERATOR = 1
ROBUST_WEIGHT_DENOMINATOR = 4
CONTEXT_WEIGHT_NUMERATOR = 3
CONTEXT_WEIGHT_DENOMINATOR = 1
SEQUENCE_WEIGHT_NUMERATOR = 5
SEQUENCE_WEIGHT_DENOMINATOR = 1
ALIGNMENT_TOLERANCE_FRAMES = 10


def _weighted_votes(
    exact_votes: int,
    robust_votes: int,
    context_votes: int,
    sequence_votes: int,
) -> float:
    return (
        exact_votes * EXACT_WEIGHT_NUMERATOR / EXACT_WEIGHT_DENOMINATOR
        + robust_votes * ROBUST_WEIGHT_NUMERATOR / ROBUST_WEIGHT_DENOMINATOR
        + context_votes * CONTEXT_WEIGHT_NUMERATOR / CONTEXT_WEIGHT_DENOMINATOR
        + sequence_votes * SEQUENCE_WEIGHT_NUMERATOR / SEQUENCE_WEIGHT_DENOMINATOR
    )


class ShardCache:
    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self.items: OrderedDict[str, tuple[dict[str, Any], int]] = OrderedDict()
        self.size_bytes = 0
        self.decisions: list[dict[str, Any]] = []

    def get(self, shard: str) -> dict[str, Any] | None:
        if shard not in self.items:
            return None
        document, size = self.items.pop(shard)
        self.items[shard] = (document, size)
        return document

    def put(self, shard: str, document: dict[str, Any], size: int) -> None:
        if shard in self.items:
            old_document, old_size = self.items.pop(shard)
            self.size_bytes -= old_size
        if size > self.capacity_bytes:
            self.decisions.append(
                {
                    "priority": 2,
                    "action": "bypass_partition_cache",
                    "reason": "shard_exceeds_memory_budget",
                    "shard": shard,
                    "bytes": size,
                    "capacity_bytes": self.capacity_bytes,
                }
            )
            return
        while self.items and self.size_bytes + size > self.capacity_bytes:
            evicted_shard, (_, evicted_size) = self.items.popitem(last=False)
            self.size_bytes -= evicted_size
            self.decisions.append(
                {
                    "priority": 2,
                    "action": "evict_partition_cache",
                    "reason": "partition_memory_budget",
                    "shard": evicted_shard,
                    "bytes": evicted_size,
                    "capacity_bytes": self.capacity_bytes,
                }
            )
        self.items[shard] = (document, size)
        self.size_bytes += size


def _query_groups(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        shard = shard_id_for_hash(record["hash"])
        grouped.setdefault(shard, {}).setdefault(record["hash"], []).append(record)
    return grouped


def _match_shard(
    shard: str,
    shard_document: dict[str, Any],
    query_hashes: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[int, list[int]]], int]:
    entries = {entry["hash"]: entry["occurrences"] for entry in shard_document["entries"]}
    hits: dict[str, dict[int, list[int]]] = {}
    work_units = 0
    for fingerprint_hash in sorted(query_hashes):
        occurrences = entries.get(fingerprint_hash)
        if not occurrences:
            continue
        query_records = query_hashes[fingerprint_hash]
        for query_record in query_records:
            for occurrence in occurrences:
                work_units += 1
                if occurrence["kind"] != query_record["kind"]:
                    continue
                track_hits = hits.setdefault(occurrence["track_id"], {})
                delta = int(occurrence["anchor_frame"]) - int(query_record["anchor_frame"])
                bucket = track_hits.setdefault(delta, [0, 0, 0, 0])
                if query_record["kind"] == "exact":
                    bucket[0] += 1
                elif query_record["kind"] == "robust":
                    bucket[1] += 1
                elif query_record["kind"] == "context":
                    bucket[2] += 1
                else:
                    bucket[3] += 1
    return hits, work_units


def _merge_hits(
    merged: dict[str, dict[int, list[int]]],
    incoming: dict[str, dict[int, list[int]]],
) -> None:
    for track_id in sorted(incoming):
        target_track = merged.setdefault(track_id, {})
        for delta in sorted(incoming[track_id]):
            exact, robust, context, sequence = incoming[track_id][delta]
            bucket = target_track.setdefault(delta, [0, 0, 0, 0])
            bucket[0] += exact
            bucket[1] += robust
            bucket[2] += context
            bucket[3] += sequence


def _candidates_from_hits(
    hits: dict[str, dict[int, list[int]]],
    *,
    query_weight: float,
    hop_samples: int,
    query_duration_samples: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    tolerance = ALIGNMENT_TOLERANCE_FRAMES
    for track_id in sorted(hits):
        histogram = hits[track_id]
        deltas = sorted(histogram)
        best: tuple[float, int, int, int, int, int] | None = None
        for delta in deltas:
            exact = 0
            robust = 0
            context = 0
            sequence = 0
            for neighbor in range(delta - tolerance, delta + tolerance + 1):
                values = histogram.get(neighbor)
                if values is not None:
                    exact += values[0]
                    robust += values[1]
                    context += values[2]
                    sequence += values[3]
            weighted = _weighted_votes(exact, robust, context, sequence)
            key = (weighted, sequence, context, exact, -abs(delta), -delta)
            if best is None or key > best:
                best = key
                best_values = (weighted, exact, robust, context, sequence, delta)
        assert best is not None
        weighted, exact, robust, context, sequence, delta = best_values
        score = min(1.0, weighted / query_weight) if query_weight > 0.0 else 0.0
        candidates.append(
            {
                "track_id": track_id,
                "score": round(score, 6),
                "alignment_frame": int(delta),
                "alignment_start_sample": int(delta) * hop_samples,
                "alignment_start_seconds": round(int(delta) * hop_samples / SAMPLE_RATE, 6),
                "duration_seconds": round(query_duration_samples / SAMPLE_RATE, 6),
                "votes": round(weighted, 6),
                "exact_votes": int(exact),
                "robust_votes": int(robust),
                "context_votes": int(context),
                "sequence_votes": int(sequence),
                "score_components": {
                    "histogram_peak_weight": round(weighted, 6),
                    "query_weight": round(query_weight, 6),
                    "normalizer": "sum(exact*1.25 + robust*0.25 + context*3.0 + sequence*5.0)",
                    "alignment_tolerance_frames": tolerance,
                    "histogram_peak_frame": int(delta),
                },
            }
        )
    candidates.sort(
        key=lambda candidate: (
            -candidate["score"],
            -candidate["votes"],
            candidate["alignment_frame"],
            candidate["track_id"],
        )
    )
    return candidates


def search(
    workbench: Workbench,
    query_path: str | Path,
    *,
    budget: BudgetConfig | None = None,
    version: int | None = None,
    query_id: str | None = None,
) -> dict[str, Any]:
    active_budget = budget or workbench.budget
    state = workbench.current if version is None else workbench.state_at(version)
    data, samples = read_wav_bytes(query_path)
    query_fingerprint = extract_fingerprint(
        samples,
        state.fingerprint_config,
        audio_sha256=audio_content_hash(data),
    )
    document = query_fingerprint.document
    if document["config_digest"] != state.fingerprint_config.digest:
        raise WorkbenchError(
            "E_QUERY_DRIFT",
            "query fingerprint configuration differs from the selected state",
            {
                "query_config_digest": document["config_digest"],
                "state_config_digest": state.fingerprint_config.digest,
            },
        )

    manifest = workbench.index.load_manifest(state.manifest_hash)
    query_by_shard = _query_groups(document["fingerprints"])
    needed_shards = sorted(set(query_by_shard) & set(manifest.get("shards", {})))
    capacity_bytes = int(active_budget.partition_memory_mb * 1024 * 1024)
    references = manifest.get("shards", {})
    sizes = [int(references[shard]["size_bytes"]) for shard in needed_shards]
    average_size = max(1, sum(sizes) // len(sizes)) if sizes else 1
    memory_slots = max(1, capacity_bytes // average_size)
    cpu_slots = max(1, os.cpu_count() or 1)
    threads_used = min(
        active_budget.query_threads,
        memory_slots,
        max(1, len(needed_shards)),
        cpu_slots,
    )
    decisions: list[dict[str, Any]] = []
    if threads_used < active_budget.query_threads:
        decisions.append(
            {
                "priority": 1,
                "action": "serial_query" if threads_used == 1 else "reduce_query_threads",
                "reason": "partition_memory_or_work_budget",
                "requested_threads": active_budget.query_threads,
                "threads_used": threads_used,
                "memory_slots": memory_slots,
            }
        )

    cache = ShardCache(capacity_bytes)
    shard_documents: dict[str, dict[str, Any]] = {}
    for shard in needed_shards:
        document_cached = cache.get(shard)
        if document_cached is None:
            document_cached = workbench.index.load_shard(references[shard], shard)
            cache.put(shard, document_cached, int(references[shard]["size_bytes"]))
        shard_documents[shard] = document_cached
    decisions.extend(cache.decisions)

    if threads_used == 1:
        results = [
            _match_shard(shard, shard_documents[shard], query_by_shard[shard])
            for shard in needed_shards
        ]
    else:
        with ThreadPoolExecutor(max_workers=threads_used) as executor:
            futures = [
                executor.submit(_match_shard, shard, shard_documents[shard], query_by_shard[shard])
                for shard in needed_shards
            ]
            results = [future.result() for future in futures]

    merged: dict[str, dict[int, list[int]]] = {}
    work_units = 0
    for shard, (shard_hits, shard_work) in zip(needed_shards, results):
        work_units += shard_work
        _merge_hits(merged, shard_hits)

    exact_query_count = sum(1 for record in document["fingerprints"] if record["kind"] == "exact")
    robust_query_count = sum(1 for record in document["fingerprints"] if record["kind"] == "robust")
    context_query_count = sum(1 for record in document["fingerprints"] if record["kind"] == "context")
    sequence_query_count = sum(1 for record in document["fingerprints"] if record["kind"] == "sequence")
    query_weight = _weighted_votes(
        exact_query_count,
        robust_query_count,
        context_query_count,
        sequence_query_count,
    )
    candidates = _candidates_from_hits(
        merged,
        query_weight=query_weight,
        hop_samples=state.fingerprint_config.hop_samples,
        query_duration_samples=int(document["duration_samples"]),
    )

    allowed_work_units = max(1, active_budget.query_time_budget_ms * 1200)
    budget_pressure = work_units > allowed_work_units
    candidate_cap = active_budget.max_candidates
    if budget_pressure:
        candidate_cap = min(candidate_cap, 4)
    before_cap = candidates
    candidates = candidates[:candidate_cap]
    if len(candidates) < len(before_cap):
        skipped = [
            {
                "track_id": candidate["track_id"],
                "score": candidate["score"],
                "votes": candidate["votes"],
                "score_components": candidate["score_components"],
            }
            for candidate in before_cap[candidate_cap:]
        ]
        decisions.append(
            {
                "priority": 3,
                "action": "skip_low_score_candidates",
                "reason": "deterministic_query_work_budget",
                "work_units": work_units,
                "allowed_work_units": allowed_work_units,
                "candidate_cap": candidate_cap,
                "before": len(before_cap),
                "after": len(candidates),
                "skipped": skipped,
            }
        )
    candidates = [
        candidate
        for candidate in candidates
        if candidate["score"] >= state.similarity_threshold
    ]
    decisions.sort(
        key=lambda decision: (
            decision["priority"],
            decision["action"],
            decision.get("shard", ""),
        )
    )
    return {
        "schema": 1,
        "query_id": query_id,
        "state_version": state.version,
        "state_digest": state.state_digest,
        "index_hash": state.manifest_hash,
        "similarity_threshold": state.similarity_threshold,
        "fingerprint_config_digest": state.fingerprint_config.digest,
        "query_audio_sha256": document["audio_sha256"],
        "query_fingerprint_hash": query_fingerprint.content_hash,
        "query": {
            "duration_samples": int(document["duration_samples"]),
            "duration_seconds": round(document["duration_samples"] / SAMPLE_RATE, 6),
            "frame_count": int(document["frame_count"]),
            "peak_count": int(document["peak_count"]),
            "fingerprint_count": int(document["fingerprint_count"]),
            "exact_count": exact_query_count,
            "robust_count": robust_query_count,
            "context_count": context_query_count,
            "sequence_count": sequence_query_count,
            "weight": round(query_weight, 6),
        },
        "budget": active_budget.to_dict(),
        "execution": {
            "threads_used": threads_used,
            "work_units": work_units,
            "allowed_work_units": allowed_work_units,
            "shards_scanned": len(needed_shards),
            "cache_resident_shards": sorted(cache.items),
            "cache_resident_bytes": cache.size_bytes,
        },
        "degradation_decisions": decisions,
        "candidates": candidates,
    }


def compare_retrieval_candidates(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> None:
    left = expected.get("candidates", [])
    right = actual.get("candidates", [])
    for index in range(max(len(left), len(right))):
        if index >= len(left) or index >= len(right):
            raise WorkbenchError(
                "E_QUERY_DRIFT",
                "retrieval candidate count drifted after state rollback",
                {
                    "first_divergence_index": index,
                    "expected_present": index < len(left),
                    "actual_present": index < len(right),
                    "expected_candidate": left[index] if index < len(left) else None,
                    "actual_candidate": right[index] if index < len(right) else None,
                },
            )
        left_candidate = left[index]
        right_candidate = right[index]
        if left_candidate != right_candidate:
            raise WorkbenchError(
                "E_QUERY_DRIFT",
                "retrieval candidate drifted after state rollback",
                {
                    "first_divergence_index": index,
                    "expected_candidate": left_candidate,
                    "actual_candidate": right_candidate,
                    "expected_score_components": left_candidate.get("score_components"),
                    "actual_score_components": right_candidate.get("score_components"),
                },
            )


def find_first_candidate_divergence(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, Any] | None:
    left = expected.get("candidates", [])
    right = actual.get("candidates", [])
    for index in range(max(len(left), len(right))):
        left_candidate = left[index] if index < len(left) else None
        right_candidate = right[index] if index < len(right) else None
        if left_candidate != right_candidate:
            return {
                "index": index,
                "expected": left_candidate,
                "actual": right_candidate,
                "expected_score_components": (
                    left_candidate.get("score_components") if left_candidate else None
                ),
                "actual_score_components": (
                    right_candidate.get("score_components") if right_candidate else None
                ),
            }
    return None
