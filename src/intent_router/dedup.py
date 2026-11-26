"""Fixed-point near-duplicate clustering based on subword sets."""

from __future__ import annotations

from dataclasses import dataclass, field

from .fixedpoint import QuantGear
from .scoring import ScoreResult

PERMILLE = 1000


@dataclass
class DedupMember:
    text_id: str
    representative_id: str
    similarity_permille: int
    boundary: bool

    def to_dict(self) -> dict:
        return {
            "text_id": self.text_id,
            "representative_id": self.representative_id,
            "similarity_permille": self.similarity_permille,
            "boundary": self.boundary,
        }


@dataclass
class DedupCluster:
    cluster_id: str
    representative_id: str
    members: list[str] = field(default_factory=list)
    removed: list[DedupMember] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "representative_id": self.representative_id,
            "members": list(self.members),
            "removed": [m.to_dict() for m in self.removed],
        }


def subword_set(tokens: tuple[str, ...]) -> frozenset[str]:
    return frozenset(tokens)


def jaccard_permille(
    left: frozenset[str], right: frozenset[str]
) -> tuple[int, bool]:
    """Fixed-point integer Jaccard in parts-per-thousand.

    Returns ``(similarity_permille, boundary)`` where boundary marks an exact
    threshold-equality candidate.  The division uses integer floor; the
    boundary flag is computed from the exact rational equality separately so a
    precise threshold value is never lost to truncation.
    """
    if not left and not right:
        return PERMILLE, False
    union = left | right
    intersection = left & right
    if not union:
        return 0, False
    num = len(intersection) * PERMILLE
    sim = num // len(union)
    # Exact rational equality: |inter|/|union| == sim/1000
    exact_eq = (
        len(intersection) * PERMILLE == sim * len(union)
    )
    return sim, exact_eq


def _composite_score(score: ScoreResult) -> int:
    """Fixed-point composite used to pick representatives: max path raw."""
    if not score.paths:
        return 0
    return max(path.quantized_raw for path in score.paths.values())


def cluster_texts(
    ordered_text_ids: list[str],
    token_sets: dict[str, frozenset[str]],
    scores: dict[str, ScoreResult],
    threshold_permille: int,
    gear: QuantGear,
) -> tuple[list[DedupCluster], dict[str, DedupMember]]:
    """Greedy deterministic single-link clustering.

    Texts are visited in the provided stable order.  A text joins the first
    cluster (in cluster creation order) whose representative is similar enough.
    Similarity ``>= threshold`` clusters them; an exact equality marks a
    boundary member.  Representatives are then re-chosen by highest composite
    fixed-point score, ties broken by text id lexicographic order.
    """
    clusters: list[DedupCluster] = []
    rep_sets: list[frozenset[str]] = []
    membership: dict[str, DedupMember] = {}

    for text_id in ordered_text_ids:
        current = token_sets[text_id]
        placed = False
        for idx, cluster in enumerate(clusters):
            sim, exact = jaccard_permille(current, rep_sets[idx])
            if sim >= threshold_permille:
                boundary = exact and sim == threshold_permille
                membership[text_id] = DedupMember(
                    text_id=text_id,
                    representative_id=cluster.representative_id,
                    similarity_permille=sim,
                    boundary=boundary,
                )
                cluster.members.append(text_id)
                placed = True
                break
        if not placed:
            cluster = DedupCluster(
                cluster_id=f"CL{len(clusters):04d}",
                representative_id=text_id,
                members=[text_id],
            )
            clusters.append(cluster)
            rep_sets.append(current)

    # Re-select representatives deterministically inside each cluster.
    removed: dict[str, DedupMember] = {}
    for cluster in clusters:
        if len(cluster.members) == 1:
            continue
        representative = min(
            cluster.members,
            key=lambda tid: (-_composite_score(scores[tid]), tid),
        )
        cluster.representative_id = representative
        rep_set = token_sets[representative]
        new_removed: list[DedupMember] = []
        for tid in cluster.members:
            if tid == representative:
                membership.pop(tid, None)
                continue
            sim, exact = jaccard_permille(token_sets[tid], rep_set)
            boundary = exact and sim == threshold_permille
            member = DedupMember(
                text_id=tid,
                representative_id=representative,
                similarity_permille=sim,
                boundary=boundary,
            )
            membership[tid] = member
            new_removed.append(member)
            removed[tid] = member
        new_removed.sort(key=lambda m: m.text_id)
        cluster.removed = new_removed

    return clusters, removed
