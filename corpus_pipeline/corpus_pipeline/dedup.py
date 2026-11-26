from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import Code, failure_record
from .fixed import similarity_fixed, threshold_fixed, to_permille
from .models import Document, PipelineConfig, content_hash


SCORE_TERM_OCCURRENCE_WEIGHT = 1_000_000
SCORE_UNIQUE_TERM_WEIGHT = 1_000
SCORE_SUBWORD_WEIGHT = 1


def composite_score(doc: Document) -> int:
    """Fixed-weight integer score used for representative selection."""

    return (
        doc.length * SCORE_TERM_OCCURRENCE_WEIGHT
        + doc.unique_terms * SCORE_UNIQUE_TERM_WEIGHT
        + len(doc.subwords) * SCORE_SUBWORD_WEIGHT
    )


class _UnionFind:
    def __init__(self, identifiers: Sequence[str]) -> None:
        self.parent = {identifier: identifier for identifier in identifiers}
        self.rank = {identifier: 0 for identifier in identifiers}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _pair_key(left: Document, right: Document) -> tuple[str, str]:
    return (left.doc_id, right.doc_id) if left.doc_id < right.doc_id else (right.doc_id, left.doc_id)


def _doc_input_fingerprint(documents: Sequence[Document]) -> str:
    return content_hash(
        [
            [
                doc.doc_id,
                doc.content_hash,
                sorted(doc.term_counts.items()),
                sorted(doc.subwords),
            ]
            for doc in sorted(documents, key=lambda item: item.doc_id)
        ]
    )


def expected_dedup_fingerprint(
    config: PipelineConfig,
    shard: Mapping[str, Any],
    documents: Sequence[Document],
) -> str:
    return content_hash(
        {
            "shard_id": shard["shard_id"],
            "domain": shard["domain"],
            "sampling_fingerprint": shard["sampling_fingerprint"],
            "input_fingerprint": _doc_input_fingerprint(documents),
            "threshold_permille": config.dedup.similarity_threshold_permille,
            "boundary_rule": config.dedup.boundary_rule,
            "precision": config.precision.to_dict(),
            "comparison_limit": config.sampling.max_dedup_comparisons,
        }
    )


def deduplicate_shard(
    config: PipelineConfig,
    shard: Mapping[str, Any],
    documents: Sequence[Document],
) -> dict[str, Any]:
    """Quantized Jaccard clustering for one shard.

    Pairs are visited in ``(doc_a, doc_b)`` lexicographic order.  A comparison
    cap therefore consumes a deterministic prefix, independent of worker
    scheduling.  The equality decision is fixed by ``boundary_rule`` and every
    equality pair is emitted as a boundary pair.
    """

    shard_id = str(shard["shard_id"])
    domain = str(shard["domain"])
    ordered_docs = sorted(documents, key=lambda doc: doc.doc_id)
    scale = config.precision.scale
    threshold = threshold_fixed(config.dedup.similarity_threshold_permille, scale)
    comparison_limit = config.sampling.max_dedup_comparisons
    pair_records: list[dict[str, Any]] = []
    boundary_pairs: list[dict[str, Any]] = []
    union_find = _UnionFind([doc.doc_id for doc in ordered_docs])
    total_pairs = len(ordered_docs) * (len(ordered_docs) - 1) // 2
    processed_pairs = 0
    limit_exhausted = False

    for left_index, left in enumerate(ordered_docs):
        for right in ordered_docs[left_index + 1 :]:
            if processed_pairs >= comparison_limit:
                limit_exhausted = True
                break
            processed_pairs += 1
            intersection = len(left.subwords.intersection(right.subwords))
            union = len(left.subwords.union(right.subwords))
            similarity = similarity_fixed(intersection, union, scale)
            is_boundary = similarity.quotient == threshold.quotient
            if similarity.quotient > threshold.quotient:
                duplicate = True
            elif similarity.quotient < threshold.quotient:
                duplicate = False
            else:
                duplicate = config.dedup.boundary_rule == "include"
            record = {
                "doc_a": left.doc_id,
                "doc_b": right.doc_id,
                "intersection": intersection,
                "union": union,
                "similarity_fixed": similarity.quotient,
                "similarity_residual_numerator": similarity.residual_numerator,
                "similarity_permille": to_permille(similarity.quotient, scale),
                "threshold_fixed": threshold.quotient,
                "threshold_permille": config.dedup.similarity_threshold_permille,
                "boundary": is_boundary,
                "duplicate": duplicate,
            }
            pair_records.append(record)
            if is_boundary:
                boundary_pairs.append(
                    {
                        "doc_a": left.doc_id,
                        "doc_b": right.doc_id,
                        "similarity_fixed": similarity.quotient,
                        "similarity_permille": record["similarity_permille"],
                        "decision": "duplicate" if duplicate else "distinct",
                        "rule": config.dedup.boundary_rule,
                    }
                )
            if duplicate:
                union_find.union(left.doc_id, right.doc_id)
        if limit_exhausted:
            break

    components: dict[str, list[str]] = {}
    for doc in ordered_docs:
        components.setdefault(union_find.find(doc.doc_id), []).append(doc.doc_id)
    by_id = {doc.doc_id: doc for doc in ordered_docs}
    cluster_for_doc: dict[str, str] = {}
    representatives: dict[str, str] = {}
    clusters: list[dict[str, Any]] = []
    for root, member_ids in sorted(components.items()):
        sorted_members = sorted(member_ids)
        representative = min(
            sorted_members,
            key=lambda doc_id: (-composite_score(by_id[doc_id]), doc_id),
        )
        cluster_id = f"{shard_id}:cluster:{representative}"
        representatives[cluster_id] = representative
        for doc_id in sorted_members:
            cluster_for_doc[doc_id] = cluster_id
        clusters.append(
            {
                "cluster_id": cluster_id,
                "members": sorted_members,
                "representative_doc_id": representative,
                "representative_score_fixed": composite_score(by_id[representative]),
                "member_scores_fixed": {
                    doc_id: composite_score(by_id[doc_id]) for doc_id in sorted_members
                },
            }
        )

    pair_similarity: dict[tuple[str, str], int] = {
        (record["doc_a"], record["doc_b"]): record["similarity_fixed"] for record in pair_records
    }
    neighbor_similarity: dict[str, int] = {doc.doc_id: 0 for doc in ordered_docs}
    for record in pair_records:
        if record["duplicate"]:
            neighbor_similarity[record["doc_a"]] = max(neighbor_similarity[record["doc_a"]], record["similarity_fixed"])
            neighbor_similarity[record["doc_b"]] = max(neighbor_similarity[record["doc_b"]], record["similarity_fixed"])

    decisions: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for doc in ordered_docs:
        cluster_id = cluster_for_doc[doc.doc_id]
        representative = representatives[cluster_id]
        kept = doc.doc_id == representative
        score = composite_score(doc)
        decision = {
            "doc_id": doc.doc_id,
            "cluster_id": cluster_id,
            "representative_doc_id": representative,
            "kept": kept,
            "composite_score_fixed": score,
        }
        if not kept:
            similarity_to_representative = pair_similarity.get(_pair_key(doc, by_id[representative]), 0)
            decision.update(
                {
                    "similarity_to_representative_fixed": similarity_to_representative,
                    "similarity_to_representative_permille": to_permille(similarity_to_representative, scale),
                    "nearest_member_similarity_fixed": neighbor_similarity[doc.doc_id],
                    "nearest_member_similarity_permille": to_permille(neighbor_similarity[doc.doc_id], scale),
                }
            )
            removed.append(dict(decision))
        decisions.append(decision)

    failures: list[dict[str, Any]] = []
    if limit_exhausted:
        failures.append(
            failure_record(
                Code.DEDUP_COMPARE_LIMIT,
                "per-shard dedup comparison limit reached; remaining pairs were not evaluated",
                domain=domain,
                shard_id=shard_id,
                processed_pairs=processed_pairs,
                total_pairs=total_pairs,
                comparison_limit=comparison_limit,
            )
        )

    input_fingerprint = _doc_input_fingerprint(ordered_docs)
    dedup_fingerprint = expected_dedup_fingerprint(config, shard, ordered_docs)
    retained_count = len(clusters)
    return {
        "shard_id": shard_id,
        "domain": domain,
        "sampling_fingerprint": shard["sampling_fingerprint"],
        "input_fingerprint": input_fingerprint,
        "dedup_fingerprint": dedup_fingerprint,
        "parameter_fingerprint": content_hash(
            {
                "threshold_permille": config.dedup.similarity_threshold_permille,
                "boundary_rule": config.dedup.boundary_rule,
                "precision": config.precision.to_dict(),
                "comparison_limit": comparison_limit,
            }
        ),
        "documents": decisions,
        "clusters": clusters,
        "removed_documents": removed,
        "boundary_pairs": boundary_pairs,
        "pairs": pair_records,
        "threshold_fixed": threshold.quotient,
        "threshold_residual_numerator": threshold.residual_numerator,
        "sampled_count": len(ordered_docs),
        "retained_count": retained_count,
        "comparisons_attempted": processed_pairs,
        "total_candidate_pairs": total_pairs,
        "comparison_limit": comparison_limit,
        "limit_exhausted": limit_exhausted,
        "failures": failures,
    }
