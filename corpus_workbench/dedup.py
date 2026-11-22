"""MinHash signature records, union-find grouping and representative decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .cache import GroupDecisionCache
from .config import DedupConfig
from .hashing import group_decision_cache_key, minhash_similarity
from .models import SignatureRecord


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


@dataclass
class DedupResult:
    groups: list[dict[str, Any]]
    retained_documents: list[dict[str, Any]]
    removed_documents: list[dict[str, Any]]
    input_document_count: int
    output_document_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_document_count": self.input_document_count,
            "output_document_count": self.output_document_count,
            "before_dedup_document_count": self.input_document_count,
            "after_dedup_document_count": self.output_document_count,
            "removed_document_count": len(self.removed_documents),
            "duplicate_group_count": len(self.groups),
            "groups": self.groups,
            "retained_documents": self.retained_documents,
            "removed_documents": self.removed_documents,
        }


def representative_sort_key(record: SignatureRecord, weights: dict[str, float]) -> tuple[Any, ...]:
    return (-weights.get(record.source, 0.0), record.doc_id, record.source)


def _public_document(record: SignatureRecord) -> dict[str, Any]:
    return {
        "source": record.source,
        "doc_id": record.doc_id,
        "fingerprint": record.fingerprint,
    }


def deduplicate(
    signatures: Iterable[SignatureRecord],
    weights: dict[str, float],
    config: DedupConfig,
    decision_cache: GroupDecisionCache | None = None,
) -> DedupResult:
    records = sorted(signatures, key=lambda record: (record.source, record.doc_id))
    count = len(records)
    if count == 0:
        return DedupResult([], [], [], 0, 0)

    similarity_cache: dict[tuple[int, int], float] = {}

    def similarity(left: int, right: int) -> float:
        key = (left, right) if left < right else (right, left)
        if key not in similarity_cache:
            similarity_cache[key] = minhash_similarity(records[key[0]].signature, records[key[1]].signature)
        return similarity_cache[key]

    union_find = UnionFind(count)
    for left in range(count):
        for right in range(left + 1, count):
            if similarity(left, right) + 1e-15 >= config.jaccard_threshold:
                union_find.union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(union_find.find(index), []).append(index)

    groups: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for indices in components.values():
        indices.sort(key=lambda index: (records[index].source, records[index].doc_id))
        if len(indices) == 1:
            retained.append(_public_document(records[indices[0]]))
            continue

        member_records = [records[index] for index in indices]
        fingerprints = sorted(record.fingerprint for record in member_records)
        member_weights = {record.source: weights.get(record.source, 0.0) for record in member_records}
        decision_key = group_decision_cache_key(fingerprints, member_weights, config)
        cached = decision_cache.get(decision_key) if decision_cache is not None else None
        decision_cache_hit = cached is not None

        if cached is not None:
            cached_key = cached.get("representative_stable_key")
            candidate = next((record for record in member_records if record.stable_key == cached_key), None)
        else:
            candidate = None
        if candidate is None:
            candidate = min(member_records, key=lambda record: representative_sort_key(record, weights))
            if decision_cache is not None:
                decision_cache.put(
                    decision_key,
                    {
                        "representative_stable_key": candidate.stable_key,
                        "representative_source": candidate.source,
                        "representative_doc_id": candidate.doc_id,
                        "member_fingerprints": fingerprints,
                    },
                )

        representative_index = records.index(candidate)
        representative_public = _public_document(candidate)
        group_members = [_public_document(record) for record in member_records]
        removals: list[dict[str, Any]] = []
        for record in member_records:
            if record.stable_key == candidate.stable_key:
                continue
            index = records.index(record)
            score = similarity(representative_index, index)
            removal = {
                "source": record.source,
                "doc_id": record.doc_id,
                "representative_source": candidate.source,
                "representative_id": candidate.doc_id,
                "similarity": round(score, 8),
                "similarity_metric": "minhash_position_match",
            }
            removals.append(removal)
            removed.append(removal)
        group = {
            "group_id": "grp-" + decision_key[:16],
            "decision_cache_key": decision_key,
            "decision_cache_hit": decision_cache_hit,
            "representative": representative_public,
            "members": group_members,
            "removed": removals,
            "size": len(group_members),
        }
        groups.append(group)
        retained.append(representative_public)

    groups.sort(
        key=lambda group: (
            group["representative"]["source"],
            group["representative"]["doc_id"],
            group["group_id"],
        )
    )
    retained.sort(key=lambda item: (item["source"], item["doc_id"]))
    removed.sort(key=lambda item: (item["source"], item["doc_id"], item["representative_id"]))
    return DedupResult(groups, retained, removed, count, len(retained))
