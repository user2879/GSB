"""Deterministic budget allocation and parallel shard dispatch."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Mapping

from .codes import (
    E_BUDGET_PENDING,
    E_ENTRY_LIMIT_EXCEEDED,
    E_SHARD_ENTRY_LIMIT,
    E_SHARD_REQUEST_FAILED,
    Issue,
)
from .models import BuildBudget, Document, Project
from .protocol import ShardClient, request_batch
from .indexing import estimate_entry_count


@dataclass
class PendingItem:
    shard_id: str
    document_ids: list[str]
    reason: str
    batch_number: int
    required_entries: int | None = None
    allocated_quota: int | None = None
    next_priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "document_ids": list(self.document_ids),
            "reason": self.reason,
            "batch_number": self.batch_number,
            "required_entries": self.required_entries,
            "allocated_quota": self.allocated_quota,
            "next_priority": self.next_priority,
        }


@dataclass
class SchedulerOutcome:
    receipts: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    pending: list[PendingItem] = field(default_factory=list)
    consumed_entries: int = 0
    dispatch_order: list[str] = field(default_factory=list)
    priority_order: list[str] = field(default_factory=list)
    allocations: list[dict[str, Any]] = field(default_factory=list)
    responses_by_shard: dict[str, dict[str, Any]] = field(default_factory=dict)

    def pending_dicts(self) -> list[dict[str, Any]]:
        ordered = sorted(self.pending, key=lambda item: (item.next_priority, item.shard_id))
        return [item.to_dict() for item in ordered]


def dispatch_batches(
    project: Project,
    batches_by_shard: Mapping[str, list[Document]],
    *,
    batch_id: str,
    batch_number: int = 1,
    already_consumed_entries: int = 0,
    prior_pending: list[str] | None = None,
    workers: int = 4,
) -> SchedulerOutcome:
    """Dispatch one batch per affected shard.

    Quotas are assigned in stable shard-id order.  Requests from
    ``prior_pending`` are removed from the front of the order, which makes the
    next batch process carry-over work without rescanning untouched shards.
    """

    prior_pending_set = set(prior_pending or [])
    outcome = SchedulerOutcome()
    budget: BuildBudget = project.budget
    remaining = max(0, budget.global_term_entry_budget - already_consumed_entries)
    shard_map = project.shard_map()

    ordered_ids = sorted(
        batches_by_shard,
        key=lambda shard_id: (0 if shard_id in prior_pending_set else 1, shard_id),
    )
    tasks: list[dict[str, Any]] = []
    for shard_id in ordered_ids:
        if shard_id not in shard_map:
            outcome.failures.append(
                Issue(
                    code="E_SHARD_UNAVAILABLE",
                    message="scheduled shard is not present in usable layout",
                    object_type="shard",
                    object_id=shard_id,
                ).to_dict()
            )
            continue
        shard = shard_map[shard_id]
        documents = sorted(batches_by_shard[shard_id], key=lambda doc: (doc.doc_id, doc.version))
        document_ids = [doc.doc_id for doc in documents]
        hard_limit = budget.per_shard_term_entry_limit
        profile = project.profile_for_revision(project.current_revision)
        required_entries = estimate_entry_count(
            documents, project.vocabulary, profile.bucket_count
        )
        quota = min(hard_limit, remaining)
        if required_entries > hard_limit:
            outcome.failures.append(
                Issue(
                    code=E_SHARD_ENTRY_LIMIT,
                    message="shard was aborted before dispatch because its entry count exceeds the per-shard limit",
                    object_type="shard",
                    object_id=shard_id,
                    field="per_shard_term_entry_limit",
                    details={
                        "required_entries": required_entries,
                        "per_shard_limit": hard_limit,
                        "allocated_quota": quota,
                    },
                ).to_dict()
            )
            continue
        if not documents:
            # An empty summary is still required after a delete so the ledger has
            # explicit empty buckets; it consumes no term entries.
            quota_effective = 0
        elif remaining < required_entries:
            pending = PendingItem(
                shard_id=shard_id,
                document_ids=document_ids,
                reason="global_budget_exhausted_before_dispatch",
                batch_number=batch_number + 1,
                required_entries=required_entries,
                allocated_quota=remaining,
                next_priority=0,
            )
            outcome.pending.append(pending)
            outcome.failures.append(
                Issue(
                    code=E_BUDGET_PENDING,
                    message="global term-entry budget is exhausted; work remains pending",
                    object_type="shard",
                    object_id=shard_id,
                    field="global_term_entry_budget",
                    details={
                        "pending_batch": batch_number + 1,
                        "document_ids": document_ids,
                        "required_entries": required_entries,
                        "remaining_budget": remaining,
                    },
                ).to_dict()
            )
            continue
        else:
            quota_effective = required_entries
        remaining -= quota_effective
        task = {
            "shard": shard,
            "documents": documents,
            "document_ids": document_ids,
            "quota": quota_effective,
            "hard_limit": hard_limit,
            "request_id": f"{batch_id}:{batch_number}:{shard_id}",
        }
        tasks.append(task)
        outcome.dispatch_order.append(shard_id)
        outcome.priority_order.append(shard_id)
        outcome.allocations.append(
            {
                "shard_id": shard_id,
                "allocated_quota": quota_effective,
                "per_shard_limit": hard_limit,
                "remaining_after_reservation": remaining,
                "prior_pending": shard_id in prior_pending_set,
                "request_id": task["request_id"],
            }
        )

    if not tasks:
        outcome.consumed_entries = 0
        return outcome

    profile = project.profile_for_revision(project.current_revision)

    def run_task(task: dict[str, Any]) -> tuple[str, dict[str, Any] | Exception]:
        shard = task["shard"]
        client = ShardClient("127.0.0.1", shard.port)
        message = request_batch(
            request_id=task["request_id"],
            shard_id=shard.shard_id,
            revision=project.current_revision,
            documents=[doc.to_dict() for doc in task["documents"]],
            compression=profile.to_dict(),
            entry_limit=int(task["quota"]),
        )
        try:
            return shard.shard_id, client.request(message)
        except Exception as exc:  # network errors become business failures
            return shard.shard_id, exc

    worker_count = max(1, min(int(workers), len(tasks)))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="gsb-shard") as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            shard_id = task["shard"].shard_id
            try:
                _, response = future.result()
            except Exception as exc:  # pragma: no cover - run_task captures most
                response = exc
            if isinstance(response, Exception):
                outcome.failures.append(
                    Issue(
                        code=E_SHARD_REQUEST_FAILED,
                        message=f"shard request failed: {response}",
                        object_type="shard",
                        object_id=shard_id,
                        details={"request_id": task["request_id"]},
                    ).to_dict()
                )
                continue
            outcome.responses_by_shard[shard_id] = response
            if response.get("ok"):
                summary = response.get("summary", {})
                required = int(summary.get("entries_count", 0))
                task["required_entries"] = required
                outcome.consumed_entries += required
                remaining -= int(task["quota"]) - required
                outcome.receipts.append(response)
                continue
            code = response.get("code", E_SHARD_REQUEST_FAILED)
            if code == E_ENTRY_LIMIT_EXCEEDED:
                required = response.get("required_entries")
                required_int = int(required) if isinstance(required, int) else None
                if required_int is not None and required_int > task["hard_limit"]:
                    outcome.failures.append(
                        Issue(
                            code=E_SHARD_ENTRY_LIMIT,
                            message="shard aborted because its entry count exceeds the per-shard limit",
                            object_type="shard",
                            object_id=shard_id,
                            field="per_shard_term_entry_limit",
                            details={
                                "required_entries": required_int,
                                "per_shard_limit": task["hard_limit"],
                                "request_id": task["request_id"],
                            },
                        ).to_dict()
                    )
                else:
                    pending = PendingItem(
                        shard_id=shard_id,
                        document_ids=task["document_ids"],
                        reason="global_budget_quota_insufficient",
                        batch_number=batch_number + 1,
                        required_entries=required_int,
                        allocated_quota=task["quota"],
                        next_priority=0,
                    )
                    outcome.pending.append(pending)
                    outcome.failures.append(
                        Issue(
                            code=E_BUDGET_PENDING,
                            message="shard needs more global budget than its allocated quota",
                            object_type="shard",
                            object_id=shard_id,
                            field="global_term_entry_budget",
                            details={
                                "required_entries": required_int,
                                "allocated_quota": task["quota"],
                                "pending_batch": batch_number + 1,
                                "request_id": task["request_id"],
                            },
                        ).to_dict()
                    )
                continue
            outcome.failures.append(
                Issue(
                    code=str(code),
                    message=str(response.get("message", "shard rejected batch")),
                    object_type="shard",
                    object_id=shard_id,
                    details={
                        "request_id": task["request_id"],
                        "served_revision": response.get("served_revision"),
                    },
                ).to_dict()
            )

    outcome.pending.sort(key=lambda item: (item.next_priority, item.shard_id))
    outcome.failures.sort(
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("object_type", "")),
            str(item.get("object_id", "")),
            str(item.get("field", "")),
            str(item.get("message", "")),
        )
    )
    outcome.receipts.sort(key=lambda receipt: str(receipt.get("request_id", "")))
    outcome.dispatch_order.sort()
    outcome.allocations.sort(key=lambda item: item["shard_id"])
    return outcome
