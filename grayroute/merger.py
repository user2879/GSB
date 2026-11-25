"""Cross-version receipt merge and invalidation boundary propagation."""
from __future__ import annotations

from dataclasses import dataclass
from .canonical import content_hash
from .errors import VersionConflictError
from .receipt import NormalizedReceipt
from .spec import MergeRule
from .batching import Batch, assert_single_version


@dataclass(frozen=True)
class MergeDecision:
    adopted: NormalizedReceipt
    rejected: tuple[tuple[NormalizedReceipt, str], ...]


def _sort_key(receipt: NormalizedReceipt, merge_seq: int,
              merge_rule: MergeRule) -> tuple:
    values = {
        "param_version_seq": receipt.param_version_seq,
        "format_version": receipt.format_version,
        "merge_seq": merge_seq,
    }
    key: list[int] = []
    for column, direction in zip(merge_rule.order, merge_rule.direction):
        value = values[column]
        key.append(-value if direction == "desc" else value)
    return tuple(key)


def merge_receipts(receipts: list[tuple[int, NormalizedReceipt]],
                   merge_rule: MergeRule) -> MergeDecision:
    """Select the adopted receipt; keep the rest with explicit reject reasons."""
    if not receipts:
        raise ValueError("cannot merge an empty receipt set")
    ordered = sorted(receipts, key=lambda pair: _sort_key(pair[1], pair[0],
                                                          merge_rule))
    winner_seq, winner = ordered[0]
    rejected = []
    for merge_seq, candidate in ordered[1:]:
        reason = (
            f"outranked by adopted receipt "
            f"param_version_seq={winner.param_version_seq} "
            f"format_version=r{winner.format_version} merge_seq={winner_seq}"
        )
        rejected.append((candidate, reason))
    return MergeDecision(adopted=winner, rejected=tuple(rejected))


def check_batch_versions(batch: Batch,
                         receipts: list[NormalizedReceipt]) -> None:
    """Reject mixed parameter versions inside one batch (E_VERSION)."""
    seqs = {r.param_version_seq for r in receipts}
    summaries = {r.param_version_seq: r.param_summary_snapshot
                 for r in receipts}
    assert_single_version(batch, seqs, summaries)


def propagate_invalidation(batches: tuple[Batch, ...],
                           adopted_by_batch: dict[str, list[NormalizedReceipt]],
                           invalid_worker_versions: set[tuple[str, int]]
                           ) -> list[dict]:
    """Walk worker/param-version -> batch -> request and mark the boundary.

    Output rows are ordered by worker id, parameter version, batch index and
    request id so the propagation trajectory is stable.
    """
    rows: list[dict] = []
    for batch in sorted(batches, key=lambda b: (b.worker_id,
                                                b.param_version_seq,
                                                b.batch_index)):
        if (batch.worker_id, batch.param_version_seq) not in \
                invalid_worker_versions:
            continue
        for request_id in sorted(batch.request_ids):
            rows.append({
                "worker_id": batch.worker_id,
                "param_version_seq": batch.param_version_seq,
                "batch_id": batch.batch_id,
                "request_id": request_id,
                "reason": (
                    f"worker parameter version "
                    f"{batch.worker_id}@v{batch.param_version_seq} invalidated"
                ),
            })
    return rows


def invalidated_requests(propagation: list[dict]) -> set[str]:
    return {row["request_id"] for row in propagation}


def stable_receipt_summary(receipt: NormalizedReceipt) -> str:
    """Summary that must stay byte-stable for non-invalidated batches."""
    return receipt.content_summary()
