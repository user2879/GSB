"""Content-fingerprint based shard/bucket reuse and invalidation ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .canonical import fingerprint
from .indexing import summarize_document_terms, term_bucket
from .models import CompressionProfile, Document, Project


def document_fingerprint(document: Document) -> str:
    return fingerprint(document.to_dict())


def current_document_fingerprints(project: Project) -> dict[str, str]:
    return {doc.doc_id: document_fingerprint(doc) for doc in project.documents}


def _previous_document_terms(summary: Mapping[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for bucket in summary.get("buckets", []):
        for term_item in bucket.get("terms", []):
            for posting in term_item.get("postings", []):
                result.setdefault(posting["doc_id"], set()).add(term_item["term"])
    return result


@dataclass
class InvalidationPlan:
    reused_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    recompute_shards: list[str] = field(default_factory=list)
    affected_buckets: dict[str, set[int]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    changed_documents: list[dict[str, Any]] = field(default_factory=list)
    previous_profile: CompressionProfile | None = None
    profile_changed: bool = False
    source_snapshot: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_snapshot": self.source_snapshot,
            "profile_changed": self.profile_changed,
            "changed_documents": list(self.changed_documents),
            "reused_shards": sorted(self.reused_summaries),
            "recompute_shards": sorted(self.recompute_shards),
            "affected_buckets": {
                shard_id: sorted(indices)
                for shard_id, indices in sorted(self.affected_buckets.items())
            },
            "records": list(self.records),
        }


def _shard_config_changed(previous_shard: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    fields = ("term_bucket_capacity", "revision")
    return any(previous_shard.get(field) != current.get(field) for field in fields)


def plan_invalidation(
    project: Project,
    *,
    previous_manifest: Mapping[str, Any] | None,
    previous_summaries: Mapping[str, Mapping[str, Any]],
    previous_shards: Mapping[str, Mapping[str, Any]],
    previous_recall: Mapping[str, Any] | None,
    mode: str,
    recall_config_changed: bool = False,
) -> InvalidationPlan:
    plan = InvalidationPlan()
    current_docs = project.document_by_id()
    current_fps = {doc_id: document_fingerprint(doc) for doc_id, doc in current_docs.items()}
    current_profile = project.profile_for_revision(project.current_revision)
    previous_profile: CompressionProfile | None = None
    if previous_manifest is not None and isinstance(previous_manifest.get("compression"), dict):
        raw = previous_manifest["compression"]
        try:
            previous_profile = CompressionProfile(
                int(raw["bucket_count"]),
                tuple(float(weight) for weight in raw["bucket_weights"]),
                str(raw["encoding"]),
            )
        except (KeyError, TypeError, ValueError):
            previous_profile = None
    plan.previous_profile = previous_profile
    plan.source_snapshot = str(previous_manifest.get("snapshot_id")) if previous_manifest else None
    profile_changed = bool(
        previous_profile is not None
        and fingerprint(previous_profile.to_dict()) != fingerprint(current_profile.to_dict())
    )
    plan.profile_changed = profile_changed

    previous_doc_fps: dict[str, str] = {}
    previous_doc_shards: dict[str, str] = {}
    previous_doc_terms: dict[str, set[str]] = {}
    for shard_id, summary in previous_summaries.items():
        for item in summary.get("document_fingerprints", []):
            previous_doc_fps[item["doc_id"]] = item["fingerprint"]
            previous_doc_shards[item["doc_id"]] = shard_id
        for doc_id, terms in _previous_document_terms(summary).items():
            previous_doc_terms.setdefault(doc_id, set()).update(terms)

    changed: list[dict[str, Any]] = []
    for doc_id in sorted(set(previous_doc_fps) | set(current_fps)):
        previous = previous_doc_fps.get(doc_id)
        current = current_fps.get(doc_id)
        if previous == current:
            continue
        if previous is None:
            change_type = "added"
        elif current is None:
            change_type = "deleted"
        else:
            change_type = "replaced"
        changed.append(
            {
                "doc_id": doc_id,
                "change_type": change_type,
                "previous_shard_id": previous_doc_shards.get(doc_id),
                "current_shard_id": current_docs[doc_id].shard_id if doc_id in current_docs else None,
                "previous_fingerprint": previous,
                "current_fingerprint": current,
            }
        )
    plan.changed_documents = changed

    changed_by_shard: dict[str, set[str]] = {}
    for change in changed:
        if change["previous_shard_id"]:
            changed_by_shard.setdefault(change["previous_shard_id"], set()).add(change["doc_id"])
        if change["current_shard_id"]:
            changed_by_shard.setdefault(change["current_shard_id"], set()).add(change["doc_id"])

    sequence = 0
    shard_ids = sorted(shard.shard_id for shard in project.shards)
    for shard_id in shard_ids:
        sequence += 1
        shard = project.shard_map()[shard_id]
        previous_summary = previous_summaries.get(shard_id)
        reason_basis: list[str] = []
        recompute = False
        affected: set[int] = set()
        if mode == "full" or previous_summary is None:
            recompute = True
            affected = set(range(current_profile.bucket_count))
            reason_basis = ["full_rebuild_from_blank"] if mode == "full" else ["no_previous_shard_snapshot"]
        else:
            previous_shard_raw = previous_shards.get(shard_id, {})
            if profile_changed:
                recompute = True
                affected = set(range(current_profile.bucket_count))
                reason_basis = ["compression_profile_changed"]
            elif _shard_config_changed(previous_shard_raw, shard.to_dict()):
                recompute = True
                affected = set(range(current_profile.bucket_count))
                reason_basis = ["shard_bucket_configuration_changed"]

        documents_changed_here = sorted(changed_by_shard.get(shard_id, set()))
        if documents_changed_here and not profile_changed:
            recompute = True
            if not reason_basis:
                reason_basis = ["referenced_documents_changed"]
            elif "referenced_documents_changed" not in reason_basis:
                reason_basis.append("referenced_documents_changed")
            for doc_id in documents_changed_here:
                for term in previous_doc_terms.get(doc_id, set()):
                    affected.add(term_bucket(term, current_profile.bucket_count))
                if doc_id in current_docs:
                    current_terms = summarize_document_terms(
                        [current_docs[doc_id]], project.vocabulary, current_profile.bucket_count
                    )
                    affected.update(current_terms.get(doc_id, set()))

        if not recompute:
            plan.reused_summaries[shard_id] = dict(previous_summary)
            plan.records.append(
                {
                    "sequence": sequence,
                    "scope": "shard",
                    "shard_id": shard_id,
                    "decision": "reuse",
                    "basis": ["unchanged_documents", "unchanged_shard_configuration"],
                    "source_snapshot": plan.source_snapshot,
                    "affected_buckets": [],
                }
            )
            continue

        plan.recompute_shards.append(shard_id)
        plan.affected_buckets[shard_id] = set(affected)
        plan.records.append(
            {
                "sequence": sequence,
                "scope": "shard",
                "shard_id": shard_id,
                "decision": "recompute",
                "basis": sorted(set(reason_basis)),
                "source_snapshot": plan.source_snapshot,
                "affected_buckets": sorted(affected),
                "changed_document_ids": documents_changed_here,
            }
        )
        for bucket_index in range(current_profile.bucket_count):
            sequence += 1
            if bucket_index in affected:
                bucket_reason = ["bucket_domain_changed"]
                if profile_changed:
                    bucket_reason.append("compression_profile_changed")
                if documents_changed_here:
                    bucket_reason.append("referenced_document_changed")
                plan.records.append(
                    {
                        "sequence": sequence,
                        "scope": "bucket",
                        "shard_id": shard_id,
                        "bucket_index": bucket_index,
                        "decision": "recompute",
                        "basis": bucket_reason,
                        "source_snapshot": plan.source_snapshot,
                    }
                )
            else:
                plan.records.append(
                    {
                        "sequence": sequence,
                        "scope": "bucket",
                        "shard_id": shard_id,
                        "bucket_index": bucket_index,
                        "decision": "reuse",
                        "basis": ["bucket_content_fingerprint_unaffected"],
                        "source_snapshot": plan.source_snapshot,
                    }
                )

    if previous_recall is not None:
        recompute_recall = bool(
            plan.recompute_shards
            or profile_changed
            or plan.changed_documents
            or recall_config_changed
        )
        plan.records.append(
            {
                "sequence": sequence + 1,
                "scope": "recall",
                "object_id": "all_queries",
                "decision": "recompute" if recompute_recall else "reuse",
                "basis": (
                    ["downstream_of_invalidated_shard_or_bucket_or_query_profile"]
                    if recompute_recall
                    else ["all_contributing_buckets_reused"]
                ),
                "source_snapshot": plan.source_snapshot,
            }
        )
    return plan


def merge_reused_buckets(
    new_summary: dict[str, Any],
    previous_summary: Mapping[str, Any],
    affected_buckets: Iterable[int],
    profile: CompressionProfile,
    documents: Iterable[Document],
    revision: int,
) -> tuple[dict[str, Any], list[int]]:
    """Restore unaffected bucket objects byte-for-byte from the snapshot."""

    affected = set(int(index) for index in affected_buckets)
    reused_indices: list[int] = []
    previous_by_index = {
        int(bucket["bucket_index"]): bucket for bucket in previous_summary.get("buckets", [])
    }
    merged_buckets: list[dict[str, Any]] = []
    for bucket in sorted(new_summary.get("buckets", []), key=lambda item: item["bucket_index"]):
        index = int(bucket["bucket_index"])
        if index not in affected and index in previous_by_index:
            merged_buckets.append(previous_by_index[index])
            reused_indices.append(index)
        else:
            merged_buckets.append(bucket)
    new_summary = dict(new_summary)
    new_summary["buckets"] = merged_buckets
    # Recompute the shard fingerprint from the mixed object.
    from .indexing import shard_fingerprint

    new_summary["shard_fingerprint"] = shard_fingerprint(
        new_summary["shard_id"],
        revision,
        profile,
        merged_buckets,
        documents,
    )
    return new_summary, sorted(reused_indices)
