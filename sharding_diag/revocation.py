"""Tolerance-aware revocation adjudication (requirement 5).

Given competing candidate conclusions for the same shard (e.g. the current
lease versus a recomputed replacement), the adjudicator returns exactly one
winning candidate and labels the cause:

* ``UPSTREAM_UNDETERMINED`` - shard is frozen; never enters revoke statistics
* ``LEASE_VERSION_CONFLICT``  - candidate versions disagree with the expected lease
* ``TOLERANCE_INDISTINGUISHABLE`` - fixed-point diff is inside the tolerance;
  the stable tie-break (shard id order, then descending lease version) wins
  and the decision is explicitly flagged as tolerance-affected
* ``DISTINGUISHABLE`` - diff exceeds tolerance; the higher fixed-point total wins
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

UPSTREAM_UNDETERMINED = "UPSTREAM_UNDETERMINED"
LEASE_VERSION_CONFLICT = "LEASE_VERSION_CONFLICT"
TOLERANCE_INDISTINGUISHABLE = "TOLERANCE_INDISTINGUISHABLE"
DISTINGUISHABLE = "DISTINGUISHABLE"

# Frozen shards are excluded from every revocation statistic.
REVOKE_STAT_CAUSES = (LEASE_VERSION_CONFLICT, TOLERANCE_INDISTINGUISHABLE, DISTINGUISHABLE)


@dataclass
class Candidate:
    shard_id: str
    node_id: str
    version: int
    capacity_version: int
    fixed_total: int
    summary_digest: str

    @classmethod
    def from_dict(cls, row):
        return cls(
            shard_id=row["shard_id"],
            node_id=row["node_id"],
            version=int(row["version"]),
            capacity_version=int(row["capacity_version"]),
            fixed_total=int(row["fixed_total"]),
            summary_digest=row["summary_digest"],
        )

    def as_dict(self):
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "version": self.version,
            "capacity_version": self.capacity_version,
            "fixed_total": self.fixed_total,
            "summary_digest": self.summary_digest,
        }


def _stable_pick(candidates: List[Candidate]) -> Candidate:
    # Explicit stable rule: shard id ascending, then lease version descending,
    # then node id as a final deterministic tie-break.
    return sorted(candidates, key=lambda c: (c.shard_id, -c.version, c.node_id))[0]


def adjudicate_shard(
    shard_id: str,
    candidates: List[Candidate],
    expected_version: Optional[int],
    is_frozen: bool,
    tolerance: int,
) -> dict:
    if is_frozen:
        return {
            "shard_id": shard_id,
            "cause": UPSTREAM_UNDETERMINED,
            "tolerance_affected": False,
            "winner": None,
            "candidates": [c.as_dict() for c in candidates],
            "diff_fp": None,
        }

    ordered = sorted(candidates, key=lambda c: (c.shard_id, -c.version, c.node_id))
    # The incumbent is the candidate whose lease version matches the active lease.
    incumbent = next((c for c in ordered if c.version == expected_version), None)
    if expected_version is not None and incumbent is None:
        winner = _stable_pick(ordered)
        return {
            "shard_id": shard_id,
            "cause": LEASE_VERSION_CONFLICT,
            "tolerance_affected": False,
            "winner": winner.as_dict(),
            "candidates": [c.as_dict() for c in ordered],
            "diff_fp": None,
            "expected_version": expected_version,
        }

    if len(ordered) < 2:
        winner = ordered[0]
        return {
            "shard_id": shard_id,
            "cause": DISTINGUISHABLE,
            "tolerance_affected": False,
            "winner": winner.as_dict(),
            "candidates": [winner.as_dict()],
            "diff_fp": 0,
        }

    # Compare the incumbent against the strongest challenger (highest version).
    challengers = [c for c in ordered if c is not incumbent] if incumbent else ordered[1:]
    challenger = sorted(challengers, key=lambda c: (-c.version, c.node_id))[0]
    diff = abs(incumbent.fixed_total - challenger.fixed_total)
    if diff <= tolerance:
        # Explicit stable rule: shard id order, then lease version descending.
        winner = _stable_pick([incumbent, challenger])
        assert winner.version == max(incumbent.version, challenger.version)
        return {
            "shard_id": shard_id,
            "cause": TOLERANCE_INDISTINGUISHABLE,
            "tolerance_affected": True,
            "winner": winner.as_dict(),
            "candidates": [c.as_dict() for c in ordered],
            "diff_fp": diff,
        }

    pair = [incumbent, challenger]
    winner = max(pair, key=lambda c: (c.fixed_total, -c.version))
    return {
        "shard_id": shard_id,
        "cause": DISTINGUISHABLE,
        "tolerance_affected": False,
        "winner": winner.as_dict(),
        "candidates": [c.as_dict() for c in ordered],
        "diff_fp": diff,
    }


def adjudicate_batch(
    candidates_by_shard: Dict[str, List[Candidate]],
    expected_versions: Dict[str, Optional[int]],
    frozen_shards: List[str],
    tolerance: int,
) -> List[dict]:
    frozen = set(frozen_shards)
    results = []
    # Stable output order independent of caller dict ordering.
    for shard_id in sorted(candidates_by_shard):
        results.append(
            adjudicate_shard(
                shard_id,
                candidates_by_shard[shard_id],
                expected_versions.get(shard_id),
                shard_id in frozen,
                tolerance,
            )
        )
    return results


def revocation_statistics(decisions: List[dict]) -> dict:
    stats = {cause: 0 for cause in REVOKE_STAT_CAUSES}
    stats["frozen_excluded"] = 0
    stats["tolerance_affected"] = 0
    for decision in decisions:
        if decision["cause"] == UPSTREAM_UNDETERMINED:
            stats["frozen_excluded"] += 1
            continue
        stats[decision["cause"]] = stats.get(decision["cause"], 0) + 1
        if decision["tolerance_affected"]:
            stats["tolerance_affected"] += 1
    return stats
