"""Batch ownership normalization and duplicate-delivery adjudication."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .errors import E_FRAME_INVALID, Failure
from .jsonutil import MISSING, decode_missing, hash_json, is_missing, json_ready
from .models import (
    ALL_RECEIPT_FIELDS,
    CandidateDecision,
    ConfigBundle,
    MergeRules,
    OwnershipDecision,
    RankConflict,
    Receipt,
    RequestSpec,
    BatchNormalized,
)


@dataclass(frozen=True)
class NormalizationResult:
    decisions: tuple[OwnershipDecision, ...]
    batches: tuple[BatchNormalized, ...]
    failures: tuple[Failure, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": [decision.as_dict() for decision in self.decisions],
            "batches": [batch.as_dict() for batch in self.batches],
            "failures": [failure.to_dict() for failure in self.failures],
        }


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def receipt_semantic_hash(receipt: Receipt) -> dict[str, Any]:
    return {
        "version": receipt.format_version,
        "source_version": receipt.source_version,
        "source_file": receipt.source_file,
        "source_index": receipt.source_index,
        "fields": receipt.as_fields(include_all=True),
    }


def batch_input_hash(requests: Sequence[RequestSpec], receipts: Sequence[Receipt]) -> str:
    request_payload = [request.as_dict() for request in sorted(requests, key=lambda item: item.key)]
    receipt_payload = sorted(hash_json(receipt_semantic_hash(receipt)) for receipt in receipts)
    return hash_json({"requests": request_payload, "receipts": receipt_payload})


def _effective_worker(receipt: Receipt, batch_worker: str) -> str:
    if is_missing(receipt.worker_id):
        return batch_worker
    return str(receipt.worker_id)


def _version_richness(receipt: Receipt) -> int:
    return {"r1": 1, "r2": 2, "r3": 3}.get(receipt.format_version, 0)


def _receipt_sort_key(receipt: Receipt, effective_worker: str) -> tuple[Any, ...]:
    rank = receipt.stable_rank
    pv = receipt.parameter_version
    duration = receipt.duration_ms
    return (
        0 if not is_missing(rank) else 1,
        int(rank) if _strict_int(rank) else 2**62,
        -_version_richness(receipt),
        effective_worker,
        -int(pv) if _strict_int(pv) else 2**62,
        int(duration) if _strict_int(duration) else 2**62,
        receipt.uid,
    )


def normalize_batch(
    batch_id: str,
    requests: Sequence[RequestSpec],
    receipts: Sequence[Receipt],
    *,
    merge_rules: MergeRules,
    input_hash: str,
    cache_hit: bool = False,
) -> tuple[BatchNormalized, list[Failure]]:
    failures: list[Failure] = []
    request_lookup = {request.request_id: request for request in requests}
    batch_worker = requests[0].worker_id if requests else ""
    by_request: dict[str, list[Receipt]] = {}
    for receipt in receipts:
        if receipt.request_id not in request_lookup:
            failures.append(
                Failure(
                    E_FRAME_INVALID,
                    f"receipt references request {receipt.request_id!r} not present in batch {batch_id}",
                    receipt.source_file,
                    receipt.source_index,
                    item=receipt.uid,
                )
            )
            continue
        by_request.setdefault(receipt.request_id, []).append(receipt)

    selected: list[Receipt] = []
    conflicts: list[RankConflict] = []
    for request_id in sorted(by_request):
        candidates = sorted(
            by_request[request_id],
            key=lambda receipt: _receipt_sort_key(receipt, _effective_worker(receipt, batch_worker)),
        )
        workers: dict[str, list[Receipt]] = {}
        for receipt in candidates:
            workers.setdefault(_effective_worker(receipt, batch_worker), []).append(receipt)
        worker_ids = sorted(workers)
        if len(worker_ids) > 1:
            ranked = []
            for worker_id in worker_ids:
                worker_receipts = sorted(
                    workers[worker_id],
                    key=lambda receipt: _receipt_sort_key(receipt, worker_id),
                )
                first = worker_receipts[0]
                if _strict_int(first.stable_rank):
                    ranked.append((worker_id, int(first.stable_rank), first))
            for index, (worker_a, rank_a, receipt_a) in enumerate(ranked):
                for worker_b, rank_b, _ in ranked[index + 1 :]:
                    if rank_a != rank_b:
                        winner = min(worker_a, worker_b)
                        conflicts.append(
                            RankConflict(
                                batch_id=batch_id,
                                request_id=request_id,
                                worker_a=worker_a,
                                worker_b=worker_b,
                                rank_a=rank_a,
                                rank_b=rank_b,
                                winner_worker=winner,
                            )
                        )
        winner = min(
            candidates,
            key=lambda receipt: (
                _effective_worker(receipt, batch_worker),
                _receipt_sort_key(receipt, _effective_worker(receipt, batch_worker)),
            ),
        )
        selected.append(winner)

    selected.sort(key=lambda receipt: receipt.log_key)
    conflicts.sort(key=lambda item: (item.batch_id, item.request_id, item.worker_a, item.worker_b))
    param_version: int | Any = MISSING
    for receipt in selected:
        if not is_missing(receipt.parameter_version):
            param_version = int(receipt.parameter_version)
            break
    return (
        BatchNormalized(
            batch_id=batch_id,
            worker_id=batch_worker,
            param_version=param_version,
            receipts=tuple(sorted(receipts, key=lambda receipt: receipt.log_key)),
            candidates=tuple(selected),
            conflicts=tuple(conflicts),
            input_hash=input_hash,
            cache_hit=cache_hit,
            task_key=f"normalize|{batch_id}|{batch_worker}",
        ),
        failures,
    )


def _candidate_refs(receipt: Receipt, effective_worker: str) -> set[str]:
    refs = {receipt.uid, receipt.delivery_key}
    refs.add(f"{receipt.batch_id}:{receipt.request_id}")
    refs.add(f"{receipt.batch_id}/{receipt.request_id}/{effective_worker}")
    return refs


def _has_reference(receipt: Receipt) -> bool:
    return not is_missing(receipt.overwritten_ref) and receipt.overwritten_ref is not None


def _pick_without_overwrites(
    candidates: Sequence[tuple[Receipt, str]],
    merge_rules: MergeRules,
) -> tuple[Receipt, str, dict[str, Any]]:
    def key(item: tuple[Receipt, str]) -> tuple[Any, ...]:
        receipt, worker_id = item
        pv = receipt.parameter_version
        duration = receipt.duration_ms
        return (
            0 if _strict_int(pv) else 1,
            -int(pv) if _strict_int(pv) else 0,
            receipt.batch_id,
            worker_id,
            int(receipt.stable_rank) if _strict_int(receipt.stable_rank) else 2**62,
            int(duration) if _strict_int(duration) else 2**62,
            merge_rules.worker_priority_index(worker_id),
            receipt.uid,
        )

    winner, worker_id = min(candidates, key=key)
    basis = {
        "rule": "parameter_version_desc+batch_id_asc",
        "candidate_uids": [receipt.uid for receipt, _ in candidates],
        "winner_parameter_version": json_ready(winner.parameter_version),
        "winner_batch_id": winner.batch_id,
        "winner_worker_id": worker_id,
    }
    return winner, worker_id, basis


def _pick_with_overwrites(
    candidates: Sequence[tuple[Receipt, str]],
    merge_rules: MergeRules,
) -> tuple[Receipt, str, dict[str, Any]]:
    ref_lookup: dict[str, tuple[Receipt, str]] = {}
    for candidate in candidates:
        for ref in _candidate_refs(candidate[0], candidate[1]):
            ref_lookup[ref] = candidate
    targeted: dict[str, str] = {}
    edges: list[dict[str, str]] = []
    for receipt, _ in candidates:
        if not _has_reference(receipt):
            continue
        target = ref_lookup.get(str(receipt.overwritten_ref))
        if target is None:
            continue
        target_receipt, _ = target
        targeted[target_receipt.uid] = receipt.uid
        edges.append({"from": target_receipt.uid, "to": receipt.uid})

    if not targeted:
        return _pick_without_overwrites(candidates, merge_rules)

    terminals = [candidate for candidate in candidates if candidate[0].uid not in targeted]
    if not terminals:
        winner, worker_id, basis = _pick_without_overwrites(candidates, merge_rules)
        basis = {
            **basis,
            "rule": "overwrite_cycle_fallback",
            "overwrite_edges": edges,
        }
        return winner, worker_id, basis

    def terminal_key(item: tuple[Receipt, str]) -> tuple[Any, ...]:
        receipt, worker_id = item
        duration = receipt.duration_ms
        return (
            0 if _strict_int(duration) else 1,
            int(duration) if _strict_int(duration) else 2**62,
            receipt.batch_id,
            worker_id,
            receipt.uid,
        )

    winner, worker_id = min(terminals, key=terminal_key)
    basis = {
        "rule": "overwrite_reference+lowest_duration_ms",
        "candidate_uids": [receipt.uid for receipt, _ in candidates],
        "overwritten_uids": sorted(targeted),
        "overwrite_edges": edges,
        "winner_duration_ms": json_ready(winner.duration_ms),
        "winner_worker_id": worker_id,
    }
    return winner, worker_id, basis


def resolve_ownership(
    batches: Sequence[BatchNormalized],
    requests: Sequence[RequestSpec],
    merge_rules: MergeRules,
) -> NormalizationResult:
    request_by_id: dict[str, list[RequestSpec]] = {}
    request_by_batch: dict[tuple[str, str], RequestSpec] = {}
    for request in requests:
        request_by_id.setdefault(request.request_id, []).append(request)
        request_by_batch[(request.batch_id, request.request_id)] = request

    candidates_by_request: dict[str, list[tuple[Receipt, str, BatchNormalized]]] = {}
    batch_worker: dict[str, str] = {}
    for batch in batches:
        batch_worker[batch.batch_id] = batch.worker_id
        for receipt in batch.candidates:
            worker = _effective_worker(receipt, batch.worker_id)
            candidates_by_request.setdefault(receipt.request_id, []).append((receipt, worker, batch))

    conflicts_by_request: dict[str, list[RankConflict]] = {}
    for batch in batches:
        for conflict in batch.conflicts:
            conflicts_by_request.setdefault(conflict.request_id, []).append(conflict)

    decisions: list[OwnershipDecision] = []
    failures: list[Failure] = []
    for request_id in sorted(candidates_by_request):
        candidates = candidates_by_request[request_id]
        contexts = request_by_id.get(request_id, [])
        if not contexts:
            failures.append(
                Failure(
                    E_FRAME_INVALID,
                    f"receipt request {request_id!r} has no request-stream context",
                    candidates[0][0].source_file,
                    candidates[0][0].source_index,
                    item=request_id,
                )
            )
            continue
        kinds = {context.kind for context in contexts}
        if len(kinds) > 1:
            failures.append(
                Failure(
                    E_FRAME_INVALID,
                    f"request {request_id!r} has conflicting kinds {sorted(kinds)}",
                    candidates[0][0].source_file,
                    candidates[0][0].source_index,
                    item=request_id,
                )
            )
            continue
        simple_candidates = [(receipt, worker) for receipt, worker, _ in candidates]
        has_overwrites = any(_has_reference(receipt) for receipt, _ in simple_candidates)
        if has_overwrites:
            winner, winner_worker, basis = _pick_with_overwrites(simple_candidates, merge_rules)
        else:
            winner, winner_worker, basis = _pick_without_overwrites(simple_candidates, merge_rules)
        context = request_by_batch.get((winner.batch_id, request_id))
        if context is None:
            failures.append(
                Failure(
                    E_FRAME_INVALID,
                    f"authoritative receipt for {request_id!r} has no batch context",
                    winner.source_file,
                    winner.source_index,
                    item=winner.uid,
                )
            )
            continue
        all_conflicts = tuple(
            sorted(
                conflicts_by_request.get(request_id, []),
                key=lambda item: (item.batch_id, item.worker_a, item.worker_b),
            )
        )
        if all_conflicts:
            basis = {**basis, "rank_conflict_count": len(all_conflicts)}
        decisions.append(
            OwnershipDecision(
                request_id=request_id,
                request_sequence=min(item.sequence for item in contexts),
                kind=context.kind,
                batch_id=winner.batch_id,
                worker_id=winner_worker,
                authoritative_uid=winner.uid,
                candidates=tuple(
                    CandidateDecision.from_receipt(receipt)
                    for receipt in sorted((item[0] for item in candidates), key=lambda receipt: receipt.log_key)
                ),
                basis=json_ready(basis),
                rank_conflicts=all_conflicts,
                duration_ms=winner.duration_ms,
                parameter_version=winner.parameter_version,
                output_digest=winner.output_digest,
            )
        )

    decisions.sort(key=lambda decision: decision.sort_key)
    failures.sort(key=Failure.sort_key)
    return NormalizationResult(tuple(decisions), tuple(sorted(batches, key=lambda item: (item.batch_id, item.worker_id))), tuple(failures))


def receipt_from_ledger_dict(raw: Mapping[str, Any]) -> Receipt:
    fields = {name: decode_missing(raw.get(name, MISSING)) for name in ALL_RECEIPT_FIELDS}
    provenance = raw.get("provenance", {})
    return Receipt(
        **fields,
        format_version=str(raw.get("format_version", "r3")),
        source_version=str(raw.get("source_version", raw.get("format_version", "r3"))),
        uid=str(raw.get("receipt_uid", "")),
        source_file=str(raw.get("source_file", "")),
        source_index=int(raw.get("source_index", 0)),
        provenance=dict(provenance) if isinstance(provenance, dict) else {},
    )
