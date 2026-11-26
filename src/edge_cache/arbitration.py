"""Authoritative receipt arbitration and dependency-aware conclusions.

Arbitration priority for one shard's candidate receipts, in order:

1. c3 override reference: a receipt that explicitly overrides a duplicate
   delivery wins over the overridden candidate.
2. quota version number descending (newer quota wins).
3. monotonic-clock millisecond duration descending (later completion wins).
4. node id lexicographic ascending (deterministic final tie-break).

If every discriminator ties the result is flagged as a tied arbitration and
*all* candidates are retained in the ledger.  Conclusions then propagate
along the upstream DAG: any shard whose upstream is concluded unavailable is
itself marked unavailable instead of being padded with a stale receipt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import BatchConfig
from .errors import E_SHARD_UNAVAILABLE, Failure
from .receipts import Receipt


@dataclass
class ArbitrationResult:
    shard_id: str
    authority: Receipt | None
    tied: bool
    candidates: list[Receipt] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    failure: Failure | None = None


def _override_rank(receipt: Receipt) -> int:
    # A receipt carrying an override reference represents the superseding
    # duplicate delivery; the referenced predecessor loses.
    return 1 if receipt.duplicate_ref else 0


def arbitrate(shard_id: str, candidates: list[Receipt],
              quota_seq: dict[str, int]) -> ArbitrationResult:
    if not candidates:
        return ArbitrationResult(
            shard_id=shard_id,
            authority=None,
            tied=False,
            candidates=[],
            rationale=["no candidate receipts"],
            failure=Failure(E_SHARD_UNAVAILABLE, "no receipt candidates",
                            shard_id=shard_id),
        )

    def sort_key(receipt: Receipt) -> tuple[Any, ...]:
        return (
            -_override_rank(receipt),
            -quota_seq.get(receipt.quota_version, 0),
            -receipt.clock_ms,
            receipt.node_id,
        )

    ordered = sorted(candidates, key=sort_key)
    best = ordered[0]
    runner = ordered[1] if len(ordered) > 1 else None
    tied = runner is not None and sort_key(runner) == sort_key(best)
    rationale = [
        "override-reference-first",
        "quota-version-desc",
        "clock-ms-desc",
        "node-id-asc",
    ]
    if tied:
        rationale.append("all-discriminators-tied")
    return ArbitrationResult(
        shard_id=shard_id,
        authority=best,
        tied=tied,
        candidates=ordered,
        rationale=rationale,
    )


@dataclass
class ShardConclusion:
    shard_id: str
    available: bool
    authority_node: str | None
    quota_version: str | None
    digest: str | None
    check_rate_permille: int | None
    tied: bool
    upstream_unavailable: tuple[str, ...] = ()
    candidate_node_ids: tuple[str, ...] = ()
    failure_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "available": self.available,
            "authority_node": self.authority_node,
            "quota_version": self.quota_version,
            "digest": self.digest,
            "check_rate_permille": self.check_rate_permille,
            "tied": self.tied,
            "upstream_unavailable": list(self.upstream_unavailable),
            "candidate_node_ids": list(self.candidate_node_ids),
            "failure_code": self.failure_code,
        }


def merge_conclusions(
    config: BatchConfig,
    arbitration: dict[str, ArbitrationResult],
) -> dict[str, ShardConclusion]:
    """Conclude every shard in topological order, propagating upstream loss."""

    conclusions: dict[str, ShardConclusion] = {}
    visiting: set[str] = set()

    def resolve(shard_id: str) -> ShardConclusion:
        if shard_id in conclusions:
            return conclusions[shard_id]
        if shard_id in visiting:  # defensive; config loader guarantees a DAG
            raise ValueError(f"dependency cycle at {shard_id}")
        visiting.add(shard_id)
        shard = config.shard(shard_id)
        blocked: list[str] = []
        for parent_id in shard.upstream:
            parent = resolve(parent_id)
            if not parent.available:
                blocked.append(parent_id)
        result = arbitration.get(shard_id)
        if blocked:
            conclusion = ShardConclusion(
                shard_id=shard_id,
                available=False,
                authority_node=None,
                quota_version=None,
                digest=None,
                check_rate_permille=None,
                tied=False,
                upstream_unavailable=tuple(blocked),
                candidate_node_ids=tuple(
                    sorted(candidate.node_id for candidate in (result.candidates if result else []))
                ),
                failure_code=E_SHARD_UNAVAILABLE,
            )
        elif result is None or result.authority is None:
            conclusion = ShardConclusion(
                shard_id=shard_id,
                available=False,
                authority_node=None,
                quota_version=None,
                digest=None,
                check_rate_permille=None,
                tied=False,
                candidate_node_ids=tuple(
                    sorted(candidate.node_id for candidate in (result.candidates if result else []))
                ),
                failure_code=E_SHARD_UNAVAILABLE,
            )
        else:
            authority = result.authority
            conclusion = ShardConclusion(
                shard_id=shard_id,
                available=True,
                authority_node=authority.node_id,
                quota_version=authority.quota_version,
                digest=authority.digest,
                check_rate_permille=authority.check_rate_permille,
                tied=result.tied,
                candidate_node_ids=tuple(
                    sorted(candidate.node_id for candidate in result.candidates)
                ),
            )
        visiting.discard(shard_id)
        conclusions[shard_id] = conclusion
        return conclusion

    for shard in sorted(config.shards, key=lambda item: item.index):
        resolve(shard.shard_id)
    return conclusions
