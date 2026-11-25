from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCORE_SCALE = 1_000_000
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_dedupe_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def subtoken_set(text: str, subword_size: int) -> frozenset[str]:
    normalized = normalize_dedupe_text(text)
    tokens: set[str] = set()
    words = WORD_RE.findall(normalized)
    for word in words:
        tokens.add("w:" + word)
        if len(word) <= subword_size:
            tokens.add("s:" + word)
        else:
            for index in range(0, len(word) - subword_size + 1):
                tokens.add("s:" + word[index : index + subword_size])
    compact = normalized.replace(" ", "\u241f")
    if compact:
        if len(compact) <= subword_size:
            tokens.add("c:" + compact)
        else:
            for index in range(0, len(compact) - subword_size + 1):
                tokens.add("c:" + compact[index : index + subword_size])
    return frozenset(tokens)


def similarity_permille(left: frozenset[str], right: frozenset[str]) -> int:
    if not left and not right:
        return 1000
    if not left or not right:
        return 0
    intersection = len(left & right)
    union = len(left | right)
    return (intersection * 1000) // union


@dataclass(frozen=True)
class DedupeInput:
    sample_id: str
    quality_score: int
    tokens: frozenset[str]

    def as_ref(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "quality_score": self.quality_score,
            "subtoken_fingerprint": _token_fingerprint(self.tokens),
        }


def _token_fingerprint(tokens: frozenset[str]) -> str:
    from .canonical import digest

    return digest({"kind": "subtoken-set", "tokens": sorted(tokens)})


@dataclass(frozen=True)
class RemovedSample:
    sample_id: str
    keeper_id: str
    similarity: int
    boundary: bool
    quality_score: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "keeper_id": self.keeper_id,
            "similarity_permille": self.similarity,
            "boundary": self.boundary,
            "quality_score": self.quality_score,
        }


@dataclass(frozen=True)
class DedupeCluster:
    cluster_id: str
    keeper_id: str
    member_ids: tuple[str, ...]
    removed: tuple[RemovedSample, ...]
    quality_scores: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "keeper_id": self.keeper_id,
            "member_ids": list(self.member_ids),
            "removed": [item.as_dict() for item in self.removed],
            "quality_scores": dict(self.quality_scores),
        }


@dataclass(frozen=True)
class DedupeResult:
    clusters: tuple[DedupeCluster, ...]
    sample_clusters: dict[str, str]
    sample_keepers: dict[str, str | None]
    sample_status: dict[str, str]
    similarity_to_keeper: dict[str, int | None]
    pairwise_similarities: tuple[dict[str, Any], ...]
    boundary_pairs: tuple[dict[str, Any], ...]
    removed_count: int
    threshold_permille: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "clusters": [cluster.as_dict() for cluster in self.clusters],
            "sample_clusters": dict(sorted(self.sample_clusters.items())),
            "sample_keepers": dict(sorted(self.sample_keepers.items())),
            "sample_status": dict(sorted(self.sample_status.items())),
            "similarity_to_keeper": dict(sorted(self.similarity_to_keeper.items())),
            "pairwise_similarities": list(self.pairwise_similarities),
            "boundary_pairs": list(self.boundary_pairs),
            "removed_count": self.removed_count,
            "threshold_permille": self.threshold_permille,
        }


def build_dedupe(
    items: Iterable[DedupeInput],
    threshold_permille: int,
) -> DedupeResult:
    ordered = sorted(items, key=lambda item: item.sample_id)
    similarities: dict[tuple[str, str], int] = {}
    pairwise: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            value = similarity_permille(left.tokens, right.tokens)
            pair = (left.sample_id, right.sample_id)
            similarities[pair] = value
            pairwise.append(
                {
                    "left": left.sample_id,
                    "right": right.sample_id,
                    "similarity_permille": value,
                    "boundary": value == threshold_permille,
                }
            )
            if value == threshold_permille:
                boundaries.append(
                    {
                        "left": left.sample_id,
                        "right": right.sample_id,
                        "similarity_permille": value,
                        "decision": "duplicate_on_boundary",
                    }
                )

    # Greedy star clustering: the highest-quality item anchors a cluster and all
    # directly-at-or-above-threshold items join it. This guarantees every removed
    # record has a direct, auditable similarity to its keeper.
    ranked = sorted(ordered, key=lambda item: (-item.quality_score, item.sample_id))
    assigned: set[str] = set()
    clusters: list[DedupeCluster] = []
    sample_clusters: dict[str, str] = {}
    sample_keepers: dict[str, str | None] = {}
    sample_status: dict[str, str] = {}
    similarity_to_keeper: dict[str, int | None] = {}
    for keeper in ranked:
        if keeper.sample_id in assigned:
            continue
        cluster_id = f"cluster:{keeper.sample_id}"
        members = [keeper.sample_id]
        removed: list[RemovedSample] = []
        assigned.add(keeper.sample_id)
        sample_clusters[keeper.sample_id] = cluster_id
        sample_keepers[keeper.sample_id] = keeper.sample_id
        sample_status[keeper.sample_id] = "kept"
        similarity_to_keeper[keeper.sample_id] = 1000
        for candidate in ranked:
            if candidate.sample_id in assigned:
                continue
            a, b = sorted((keeper.sample_id, candidate.sample_id))
            value = similarities.get((a, b), 1000 if a == b else 0)
            if value < threshold_permille:
                continue
            assigned.add(candidate.sample_id)
            members.append(candidate.sample_id)
            sample_clusters[candidate.sample_id] = cluster_id
            sample_keepers[candidate.sample_id] = keeper.sample_id
            sample_status[candidate.sample_id] = "removed"
            similarity_to_keeper[candidate.sample_id] = value
            removed.append(
                RemovedSample(
                    sample_id=candidate.sample_id,
                    keeper_id=keeper.sample_id,
                    similarity=value,
                    boundary=value == threshold_permille,
                    quality_score=candidate.quality_score,
                )
            )
        removed.sort(key=lambda item: item.sample_id)
        members.sort()
        clusters.append(
            DedupeCluster(
                cluster_id=cluster_id,
                keeper_id=keeper.sample_id,
                member_ids=tuple(members),
                removed=tuple(removed),
                quality_scores=tuple(
                    (sample_id, next(item.quality_score for item in ordered if item.sample_id == sample_id))
                    for sample_id in members
                ),
            )
        )
    clusters.sort(key=lambda item: item.keeper_id)
    pairwise.sort(key=lambda item: (item["left"], item["right"]))
    boundaries.sort(key=lambda item: (item["left"], item["right"]))
    return DedupeResult(
        clusters=tuple(clusters),
        sample_clusters=dict(sorted(sample_clusters.items())),
        sample_keepers=dict(sorted(sample_keepers.items())),
        sample_status=dict(sorted(sample_status.items())),
        similarity_to_keeper=dict(sorted(similarity_to_keeper.items())),
        pairwise_similarities=tuple(pairwise),
        boundary_pairs=tuple(boundaries),
        removed_count=sum(len(cluster.removed) for cluster in clusters),
        threshold_permille=threshold_permille,
    )

