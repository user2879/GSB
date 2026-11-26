"""Adoption (merge) decisions across repeated uploads and mixed formats.

Winner rule for the same (object key, shard index): highest metadata version,
then highest source format version, then smallest stable merge sequence, then
lexicographic merge id as a final total-order tiebreaker. All ordering is
explicit, so results are identical for any request dispatch / dict iteration
order. Rejected shards are retained with an explicit reason.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import (
    AdoptionDecision,
    MergeCandidate,
    RejectedShard,
)


def decide_object(candidates: list[MergeCandidate]) -> list[AdoptionDecision]:
    """Return one adoption decision per shard index for a single object key."""
    grouped: dict[int, list[MergeCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.shard_index].append(candidate)

    decisions: list[AdoptionDecision] = []
    for shard_index in sorted(grouped):
        members = sorted(grouped[shard_index], key=lambda item: item.canonical_tuple())
        winner = members[0]
        rejected: list[RejectedShard] = []
        for candidate in members[1:]:
            reason = _rejection_reason(candidate, winner)
            rejected.append(
                RejectedShard(candidate=candidate, reason=reason,
                              winner_merge_id=winner.merge_id)
            )
        decisions.append(
            AdoptionDecision(
                object_key=winner.object_key,
                shard_index=shard_index,
                winner=winner,
                rejected=tuple(rejected),
            )
        )
    return decisions


def _rejection_reason(candidate: MergeCandidate, winner: MergeCandidate) -> str:
    if candidate.meta_version < winner.meta_version:
        return "lower-meta-version"
    if candidate.meta_version > winner.meta_version:  # pragma: no cover - winner sorts first
        return "tie-order"
    if candidate.source_format < winner.source_format:
        return "lower-source-format"
    if candidate.source_format > winner.source_format:  # pragma: no cover
        return "tie-order"
    if candidate.merge_seq > winner.merge_seq:
        return "later-merge-sequence"
    if candidate.merge_id != winner.merge_id and candidate.shard_digest == winner.shard_digest:
        return "identical-digest-duplicate"
    return "stable-tiebreak-merge-id"


def decide_all(candidates: Iterable[MergeCandidate]) -> dict[str, list[AdoptionDecision]]:
    """Group candidates by object key (deterministic) and decide each."""
    by_key: dict[str, list[MergeCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_key[candidate.object_key].append(candidate)
    return {key: decide_object(by_key[key]) for key in sorted(by_key)}


def decisions_canonical_blob(decisions: dict[str, list[AdoptionDecision]]) -> bytes:
    """Byte-identifiable serialization of the whole adoption conclusion."""
    import json

    payload = {
        key: [
            {
                "shard_index": decision.shard_index,
                "winner_merge_id": decision.winner.merge_id,
                "winner_digest": decision.winner.shard_digest,
                "winner_content_check": decision.winner.content_check,
                "rejected": [
                    {
                        "merge_id": item.candidate.merge_id,
                        "reason": item.reason,
                        "winner_merge_id": item.winner_merge_id,
                    }
                    for item in decision.rejected
                ],
            }
            for decision in decisions[key]
        ]
        for key in sorted(decisions)
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = ["decide_object", "decide_all", "decisions_canonical_blob"]
