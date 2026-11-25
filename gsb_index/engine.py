"""End-to-end orchestration of validation, shards, budget, arbitration and snapshots."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import fingerprint
from .cluster import ClusterStartError, ShardCluster
from .codes import (
    E_RECEIPT_PROFILE_MISMATCH,
    E_SHARD_START_FAILED,
    E_SNAPSHOT_COMMIT,
    Issue,
    issue_sort_key,
)
from .config import load_project, project_config_hash
from .invalidation import current_document_fingerprints, merge_reused_buckets, plan_invalidation
from .ledger import assemble_ledger, business_projection
from .models import CompressionProfile, Project
from .revisions import RevisionChain
from .scheduler import dispatch_batches
from .scoring import recall_all
from .snapshot import LoadedSnapshot, SnapshotStore


@dataclass
class BuildResult:
    mode: str
    global_version: int
    config_hash: str
    project: Project
    ledger: dict[str, Any]
    recall: list[dict[str, Any]]
    report: dict[str, Any]
    reuse_ledger: list[dict[str, Any]]
    invalidation: dict[str, Any]
    recovery: dict[str, Any]
    failure_list: list[dict[str, Any]]
    snapshot_id: str
    committed: bool
    staging_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "ledger": self.ledger,
            "recall": self.recall,
            "report": self.report,
            "reuse_ledger": self.reuse_ledger,
            "invalidation": self.invalidation,
            "recovery": self.recovery,
            "failure_list": self.failure_list,
            "snapshot_id": self.snapshot_id,
            "committed": self.committed,
            "staging_path": self.staging_path,
        }


def _parse_profile(raw: Mapping[str, Any] | None) -> CompressionProfile | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        return CompressionProfile(
            bucket_count=int(raw["bucket_count"]),
            bucket_weights=tuple(float(weight) for weight in raw["bucket_weights"]),
            encoding=str(raw["encoding"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _receipt_arbitration_entry(
    response: Mapping[str, Any],
    arbitration,
) -> dict[str, Any]:
    return {
        "request_id": response.get("request_id"),
        "shard_id": response.get("shard_id"),
        "served_revision": arbitration.served_revision,
        "current_revision": arbitration.current_revision,
        "decision": arbitration.decision,
        "code": arbitration.code,
        "basis": list(arbitration.basis),
        "covered_by": list(arbitration.covered_by),
        "compatible": arbitration.compatible,
        "counted": arbitration.compatible,
        "summary_fingerprint": (
            response.get("summary", {}).get("shard_fingerprint")
            if isinstance(response.get("summary"), Mapping)
            else None
        ),
        "raw_receipt": dict(response),
    }


def _deterministic_failure_sort(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=issue_sort_key)


def run_build(
    config_path: str | Path,
    workdir: str | Path,
    *,
    mode: str = "incremental",
    workers: int = 4,
    global_version: int | None = None,
    prior_pending: list[str] | None = None,
    start_services: bool = True,
    commit: bool = True,
    batch_number: int = 1,
) -> BuildResult:
    if mode not in {"incremental", "full"}:
        raise ValueError("mode must be 'incremental' or 'full'")
    started_ns = time.monotonic_ns()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    project, validation_issues = load_project(config_path)
    config_hash = project_config_hash(project)
    store = SnapshotStore(workdir)
    previous = store.load_current()
    if global_version is None:
        global_version = int(previous.manifest.get("global_version", 0)) + 1 if previous is not None else 1
    global_version = int(global_version)
    document_fingerprints = current_document_fingerprints(project)
    document_fingerprint = fingerprint(document_fingerprints)
    recovery = store.recovery_decision(
        requested_global_version=global_version,
        expected_config_hash=config_hash,
        expected_document_fingerprint=document_fingerprint,
        mode=mode,
    )
    failures: list[dict[str, Any]] = [dict(issue) for issue in validation_issues]

    chain = RevisionChain(project.revisions, project.current_revision)
    current_profile = project.profile_for_revision(project.current_revision)
    previous_for_plan: LoadedSnapshot | None = previous if mode == "incremental" else None
    previous_summaries = previous_for_plan.summaries if previous_for_plan else {}
    previous_manifest = previous_for_plan.manifest if previous_for_plan else None
    previous_recall = previous_for_plan.recall if previous_for_plan else None
    previous_shards = {
        str(shard["shard_id"]): dict(shard)
        for shard in (previous_manifest or {}).get("shard_specs", [])
        if isinstance(shard, Mapping) and "shard_id" in shard
    }
    recall_config_changed = bool(
        previous_manifest is not None
        and (
            previous_manifest.get("scoring") != project.scoring.to_dict()
            or previous_manifest.get("queries") != [query.to_dict() for query in project.queries]
        )
    )
    invalidation_plan = plan_invalidation(
        project,
        previous_manifest=previous_manifest,
        previous_summaries=previous_summaries,
        previous_shards=previous_shards,
        previous_recall=previous_recall,
        mode=mode,
        recall_config_changed=recall_config_changed,
    )

    prior_pending_ids = list(prior_pending or [])
    if not prior_pending_ids and previous_manifest is not None:
        manifest_pending = previous_manifest.get("pending_shards", [])
        if isinstance(manifest_pending, list):
            prior_pending_ids = [str(shard_id) for shard_id in manifest_pending]

    summaries: dict[str, dict[str, Any]] = {
        shard_id: dict(summary)
        for shard_id, summary in invalidation_plan.reused_summaries.items()
    }
    reuse_entry_count = sum(int(summary.get("entries_count", 0)) for summary in summaries.values())

    recompute_shards = sorted(invalidation_plan.recompute_shards)
    batches_by_shard = project.documents_by_shard()
    batches = {shard_id: batches_by_shard.get(shard_id, []) for shard_id in recompute_shards}
    scheduler_failures: list[dict[str, Any]] = []
    pending_items: list[dict[str, Any]] = []
    dispatch_order: list[str] = []
    priority_order: list[str] = []
    allocations: list[dict[str, Any]] = []
    receipts: list[Mapping[str, Any]] = []

    cluster: ShardCluster | None = None
    cluster_start_error: Exception | None = None
    try:
        if start_services and batches:
            cluster = ShardCluster(project, workdir, recompute_shards)
            try:
                cluster.start(continue_on_error=True)
            except ClusterStartError as exc:
                cluster_start_error = exc
                for shard_id in recompute_shards:
                    failures.append(
                        Issue(
                            code=E_SHARD_START_FAILED,
                            message=f"unable to start shard subprocess: {exc}",
                            object_type="shard",
                            object_id=shard_id,
                            details={"log_path": str(exc.log_path) if exc.log_path else None},
                        ).to_dict()
                    )
            if cluster.start_errors:
                for shard_id, exc in sorted(cluster.start_errors.items()):
                    failures.append(
                        Issue(
                            code=E_SHARD_START_FAILED,
                            message=f"unable to start shard subprocess: {exc}",
                            object_type="shard",
                            object_id=shard_id,
                            details={"log_path": str(exc.log_path) if exc.log_path else None},
                        ).to_dict()
                    )
                batches = {
                    shard_id: docs
                    for shard_id, docs in batches.items()
                    if shard_id not in cluster.start_errors
                }
                if not batches:
                    cluster_start_error = RuntimeError("all requested shard processes failed to start")
        elif batches and not start_services:
            cluster_start_error = RuntimeError("service startup disabled")
            for shard_id in recompute_shards:
                failures.append(
                    Issue(
                        code=E_SHARD_START_FAILED,
                        message="shard service startup is disabled",
                        object_type="shard",
                        object_id=shard_id,
                    ).to_dict()
                )

        if batches and cluster_start_error is None:
            outcome = dispatch_batches(
                project,
                batches,
                batch_id=f"v{global_version}-{config_hash[:12]}",
                batch_number=batch_number,
                already_consumed_entries=reuse_entry_count,
                prior_pending=prior_pending_ids,
                workers=workers,
            )
            receipts = list(outcome.receipts)
            scheduler_failures.extend(outcome.failures)
            pending_items = outcome.pending_dicts()
            dispatch_order = outcome.dispatch_order
            priority_order = outcome.priority_order
            allocations = outcome.allocations
    finally:
        if cluster is not None:
            cluster.stop()

    failures.extend(scheduler_failures)
    arbitration_records: list[dict[str, Any]] = []
    accepted_receipt_entries: list[int] = []
    for response in receipts:
        summary_raw = response.get("summary")
        if not isinstance(summary_raw, Mapping):
            failures.append(
                Issue(
                    code=E_RECEIPT_PROFILE_MISMATCH,
                    message="successful receipt did not contain a shard summary",
                    object_type="receipt",
                    object_id=str(response.get("request_id")),
                ).to_dict()
            )
            continue
        served = response.get("served_revision")
        served_int = int(served) if isinstance(served, int) else 0
        receipt_profile = _parse_profile(response.get("compression"))
        bucket_indices = [
            int(bucket["bucket_index"])
            for bucket in summary_raw.get("buckets", [])
            if isinstance(bucket, Mapping) and "bucket_index" in bucket
        ]
        arbitration = chain.arbitrate(
            served_int,
            bucket_indices,
            receipt_profile=receipt_profile,
        )
        arbitration_record = _receipt_arbitration_entry(response, arbitration)
        arbitration_records.append(arbitration_record)
        if not arbitration.compatible:
            failures.append(
                Issue(
                    code=arbitration.code,
                    message="receipt was retained but cannot be counted in the current-revision ledger",
                    object_type="receipt",
                    object_id=str(response.get("request_id")),
                    details={
                        "shard_id": response.get("shard_id"),
                        "served_revision": arbitration.served_revision,
                        "current_revision": arbitration.current_revision,
                        "basis": list(arbitration.basis),
                        "covered_by": list(arbitration.covered_by),
                    },
                ).to_dict()
            )
            continue
        shard_id = str(summary_raw.get("shard_id"))
        summary = dict(summary_raw)
        previous_summary = previous_summaries.get(shard_id)
        if (
            previous_summary is not None
            and not invalidation_plan.profile_changed
            and shard_id in invalidation_plan.affected_buckets
        ):
            documents_for_shard = batches.get(shard_id, [])
            summary, reused_indices = merge_reused_buckets(
                summary,
                previous_summary,
                invalidation_plan.affected_buckets[shard_id],
                current_profile,
                documents_for_shard,
                project.current_revision,
            )
            arbitration_record["reused_bucket_indices"] = reused_indices
        summaries[shard_id] = summary
        accepted_receipt_entries.append(int(summary.get("entries_count", 0)))

    failures = _deterministic_failure_sort(failures)
    all_summary_entry_count = sum(int(summary.get("entries_count", 0)) for summary in summaries.values())
    consumed_budget = all_summary_entry_count
    completed_shards = sorted(summaries)
    pending_shards = sorted(str(item["shard_id"]) for item in pending_items)
    completed_doc_cursor = 0
    for document in sorted(project.documents, key=lambda doc: (doc.doc_id, doc.version)):
        if document.shard_id in summaries:
            completed_doc_cursor += 1

    ledger = assemble_ledger(
        global_version=global_version,
        config_hash=config_hash,
        index_revision=project.current_revision,
        summaries=summaries,
        receipt_arbitrations=arbitration_records,
        failures=failures,
        consumed_budget=consumed_budget,
    )

    recall_reuse = False
    if (
        previous_for_plan is not None
        and previous_recall is not None
        and not invalidation_plan.recompute_shards
        and not recall_config_changed
        and invalidation_plan.changed_documents == []
    ):
        recall = previous_for_plan.recall
        recall_reuse = True
    else:
        recall = recall_all(summaries.values(), project.queries, project.scoring)

    reuse_records = list(invalidation_plan.records)
    if recall_reuse:
        for record in reuse_records:
            if record.get("scope") == "recall":
                record["decision"] = "reuse"
                record["basis"] = ["persisted_recall_fingerprint_reused"]
    reuse_counts = {
        "shard_reuse": sum(1 for record in reuse_records if record.get("scope") == "shard" and record.get("decision") == "reuse"),
        "shard_recompute": sum(1 for record in reuse_records if record.get("scope") == "shard" and record.get("decision") == "recompute"),
        "bucket_reuse": sum(1 for record in reuse_records if record.get("scope") == "bucket" and record.get("decision") == "reuse"),
        "bucket_recompute": sum(1 for record in reuse_records if record.get("scope") == "bucket" and record.get("decision") == "recompute"),
        "recall_reuse": int(recall_reuse),
    }

    status = "complete"
    if pending_shards or any(str(item.get("object_type")) == "shard" for item in failures):
        status = "partial"
    arbitration_stats = ledger["arbitration_stats"]
    code_counts = dict(ledger["business_code_counts"])
    for record in arbitration_records:
        if record.get("decision") == "accepted_compatible":
            code_counts[record["code"]] = code_counts.get(record["code"], 0) + 1
    code_counts = dict(sorted(code_counts.items()))
    end_to_end_ms = (time.monotonic_ns() - started_ns) // 1_000_000
    report = {
        "schema_version": 1,
        "build_mode": mode,
        "status": status,
        "global_version": global_version,
        "config_hash": config_hash,
        "index_revision": project.current_revision,
        "shard_entry_counts": ledger["shard_entry_counts"],
        "entries_count": ledger["entries_count"],
        "consumed_budget": consumed_budget,
        "global_term_entry_budget": project.budget.global_term_entry_budget,
        "per_shard_term_entry_limit": project.budget.per_shard_term_entry_limit,
        "arbitration": arbitration_stats,
        "business_code_counts": code_counts,
        "failure_count": len(failures),
        "failure_list": failures,
        "recovery": recovery,
        "reuse": {
            "counts": reuse_counts,
            "records": reuse_records,
            "source_snapshot": invalidation_plan.source_snapshot,
        },
        "scheduler": {
            "dispatch_order": dispatch_order,
            "priority_order": priority_order,
            "allocations": allocations,
            "pending": pending_items,
            "next_batch_priority": [item["shard_id"] for item in pending_items],
            "prior_pending": sorted(prior_pending_ids),
        },
        "completed_doc_cursor": completed_doc_cursor,
        "document_count": len(project.documents),
        "completed_shards": completed_shards,
        "pending_shards": pending_shards,
        "end_to_end_ms_monotonic": end_to_end_ms,
        "business_digest": ledger["business_digest"],
    }

    shard_specs = [shard.to_dict() for shard in project.shards]
    manifest_fields = {
        "global_version": global_version,
        "config_hash": config_hash,
        "index_revision": project.current_revision,
        "current_revision": project.current_revision,
        "compression": current_profile.to_dict(),
        "completed_doc_cursor": completed_doc_cursor,
        "document_count": len(project.documents),
        "document_fingerprint": document_fingerprint,
        "document_fingerprints": document_fingerprints,
        "completed_shards": completed_shards,
        "pending_shards": pending_shards,
        "consumed_budget": consumed_budget,
        "global_term_entry_budget": project.budget.global_term_entry_budget,
        "per_shard_term_entry_limit": project.budget.per_shard_term_entry_limit,
        "shard_specs": shard_specs,
        "scoring": project.scoring.to_dict(),
        "queries": [query.to_dict() for query in project.queries],
        "failure_count": len(failures),
        "failure_digest": fingerprint(failures),
        "reuse_digest": fingerprint(reuse_records),
        "previous_snapshot": invalidation_plan.source_snapshot,
        "build_mode": mode,
    }
    staged = store.stage(
        global_version=global_version,
        config_hash=config_hash,
        manifest_fields=manifest_fields,
        summaries=summaries,
        ledger=ledger,
        recall=recall,
        report=report,
    )
    committed = False
    snapshot_id = staged.snapshot_id
    commit_failure: dict[str, Any] | None = None
    if commit:
        try:
            loaded = store.commit(staged)
            snapshot_id = loaded.snapshot_id
            committed = True
            report["snapshot_id"] = snapshot_id
        except RuntimeError as exc:
            commit_failure = Issue(
                code=E_SNAPSHOT_COMMIT,
                message=str(exc),
                object_type="snapshot",
                object_id=staged.snapshot_id,
            ).to_dict()
            failures = _deterministic_failure_sort(failures + [commit_failure])
            report["failure_list"] = failures
            report["failure_count"] = len(failures)
            report["business_code_counts"] = dict(
                sorted({code: sum(1 for item in failures if item.get("code") == code) for code in {item.get("code") for item in failures}}.items())
            )
            report["status"] = "partial"

    return BuildResult(
        mode=mode,
        global_version=global_version,
        config_hash=config_hash,
        project=project,
        ledger=ledger,
        recall=recall,
        report=report,
        reuse_ledger=reuse_records,
        invalidation=invalidation_plan.to_dict(),
        recovery=recovery,
        failure_list=failures,
        snapshot_id=snapshot_id,
        committed=committed,
        staging_path=str(staged.staging_path) if not committed else None,
    )


def equivalence_projection(result: BuildResult) -> dict[str, Any]:
    """Business fields that must match incremental and blank full rebuilds."""

    ledger = business_projection(result.ledger)
    for key in ("config_hash", "global_version", "business_digest"):
        ledger.pop(key, None)
    return {
        "entries_count": ledger.get("entries_count"),
        "consumed_budget": ledger.get("consumed_budget"),
        "shard_entry_counts": ledger.get("shard_entry_counts"),
        "buckets": ledger.get("buckets"),
        # Accepted network receipts depend on whether a shard was recomputed or
        # served from a persisted summary.  Rejected adjudications are business
        # decisions and must match across execution paths.
        "rejected_receipt_arbitrations": [
            {
                key: value
                for key, value in item.items()
                if key
                in {
                    "shard_id",
                    "served_revision",
                    "current_revision",
                    "decision",
                    "code",
                    "basis",
                    "covered_by",
                    "compatible",
                    "counted",
                    "summary_fingerprint",
                }
            }
            for item in ledger.get("receipt_arbitrations", [])
            if not item.get("counted")
        ],
        "failure_list": ledger.get("failure_list"),
        "recall": business_projection({"recall": result.recall})["recall"],
    }
