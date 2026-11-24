"""Batch aggregation and duplicate-delivery arbitration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import (
    E_BATCH_VERSION_CONFLICT,
    E_BATCH_WORKER_CONFLICT,
    E_COVERAGE_CYCLE,
    E_COVERAGE_REF_INVALID,
    E_DUPLICATE_RANK,
    E_RECEIPT_MALFORMED,
    Issue,
)
from .model import AppConfig, Receipt, RequestSpec, is_missing


UNKNOWN_WORKER = "<unknown>"


@dataclass(frozen=True)
class Delivery:
    receipt: Receipt
    selected: bool
    status: str = "candidate"
    covers: tuple[str, ...] = ()
    covered_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class Batch:
    batch_id: str
    worker_id: str
    param_version: int | None
    receipts: tuple[Receipt, ...]
    request_ids: tuple[str, ...]
    selected_by_merge_rule: bool
    invalidated: bool
    invalidation_reason: str
    source_files: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (self.batch_id, self.worker_id)

    @property
    def request_count(self) -> int:
        return len(self.request_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "param_version": self.param_version,
            "request_count": self.request_count,
            "request_ids": list(self.request_ids),
            "selected_by_merge_rule": self.selected_by_merge_rule,
            "invalidated": self.invalidated,
            "invalidation_reason": self.invalidation_reason,
            "receipt_count": len(self.receipts),
            "source_files": list(self.source_files),
        }


@dataclass(frozen=True)
class AuthorityDecision:
    request_id: str
    authority: Receipt
    rule: str
    reason: dict[str, Any]
    deliveries: tuple[Delivery, ...]
    coverage_edges: tuple[dict[str, str], ...]
    candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "authority_key": self.authority.key,
            "rule": self.rule,
            "reason": self.reason,
            "candidate_count": self.candidate_count,
            "deliveries": [
                {
                    "key": delivery.receipt.key,
                    "batch_id": delivery.receipt.batch_id,
                    "worker_id": delivery.receipt.resolved_worker_id,
                    "param_version": delivery.receipt.resolved_param_version,
                    "duration_ms": (
                        None
                        if is_missing(delivery.receipt.duration_ms)
                        else delivery.receipt.duration_ms
                    ),
                    "rank": (
                        None if is_missing(delivery.receipt.rank) else delivery.receipt.rank
                    ),
                    "status": delivery.status,
                    "selection": "selected" if delivery.selected else "excluded",
                    "covers": list(delivery.covers),
                    "covered_by": list(delivery.covered_by),
                    "covered_refs": (
                        []
                        if is_missing(delivery.receipt.covered_refs)
                        else list(delivery.receipt.covered_refs)
                    ),
                    "source": {
                        "path": delivery.receipt.source_path,
                        "line": delivery.receipt.source_line,
                        "version": delivery.receipt.version,
                    },
                }
                for delivery in self.deliveries
            ],
            "coverage_edges": list(self.coverage_edges),
        }


@dataclass
class AttributionResult:
    batches: tuple[Batch, ...]
    decisions: tuple[AuthorityDecision, ...]
    issues: tuple[Issue, ...]
    invalid_receipts: tuple[Receipt, ...]
    uncovered_request_ids: tuple[str, ...]
    batch_by_key: dict[tuple[str, str], Batch] = field(default_factory=dict)

    @property
    def selected_batches(self) -> tuple[Batch, ...]:
        return tuple(batch for batch in self.batches if batch.selected_by_merge_rule)


def _version_sort(receipt: Receipt) -> tuple[Any, ...]:
    version = receipt.resolved_param_version
    return (
        1 if version is None else 0,
        -(version if version is not None else 0),
        receipt.batch_id,
        receipt.resolved_worker_id or "",
        receipt.source_path,
        receipt.source_line,
    )


def _duration_sort(receipt: Receipt) -> tuple[Any, ...]:
    duration = receipt.duration_ms
    missing_duration = is_missing(duration)
    return (
        1 if missing_duration else 0,
        int(duration) if not missing_duration else 0,
    )


def _resolve_worker_conflicts(
    receipts: tuple[Receipt, ...],
    config: AppConfig,
) -> tuple[list[Receipt], list[Issue]]:
    valid: list[Receipt] = []
    issues: list[Issue] = []
    for receipt in receipts:
        worker_id = receipt.resolved_worker_id
        field_worker = None if is_missing(receipt.worker_id) else str(receipt.worker_id)
        source_worker = receipt.default_worker_id
        if field_worker and source_worker and field_worker != source_worker:
            issues.append(
                Issue(
                    code=E_BATCH_WORKER_CONFLICT,
                    message=(
                        f"receipt worker {field_worker!r} conflicts with source worker "
                        f"{source_worker!r}"
                    ),
                    path=receipt.source_path,
                    line=receipt.source_line,
                    item=receipt.key,
                    injection_id=",".join(receipt.injection_ids),
                )
            )
            continue
        if worker_id is not None and worker_id not in config.worker_by_id:
            issues.append(
                Issue(
                    code=E_BATCH_WORKER_CONFLICT,
                    message=f"receipt references unknown worker {worker_id!r}",
                    path=receipt.source_path,
                    line=receipt.source_line,
                    item=receipt.key,
                    injection_id=",".join(receipt.injection_ids),
                )
            )
            continue
        valid.append(receipt)
    return valid, issues


def _rank_conflicts(
    receipts: list[Receipt],
) -> tuple[list[Receipt], list[Issue], list[Receipt]]:
    issues: list[Issue] = []
    invalid: list[Receipt] = []
    seen: dict[tuple[str, str, int], Receipt] = {}
    ordered = sorted(receipts, key=lambda item: (item.source_path, item.source_line))
    valid: list[Receipt] = []
    for receipt in ordered:
        if is_missing(receipt.rank):
            valid.append(receipt)
            continue
        rank_key = (receipt.batch_id, receipt.resolved_worker_id or UNKNOWN_WORKER, int(receipt.rank))
        previous = seen.get(rank_key)
        if previous is not None:
            issues.append(
                Issue(
                    code=E_DUPLICATE_RANK,
                    message=(
                        f"duplicate rank {receipt.rank} in batch {receipt.batch_id}; "
                        f"first seen at {previous.source_path}:{previous.source_line}"
                    ),
                    path=receipt.source_path,
                    line=receipt.source_line,
                    item=receipt.key,
                    injection_id=",".join(receipt.injection_ids),
                )
            )
            invalid.append(receipt)
            continue
        seen[rank_key] = receipt
        valid.append(receipt)
    return valid, issues, invalid


def _batch_version(
    batch_id: str,
    worker_id: str,
    receipts: tuple[Receipt, ...],
    config: AppConfig,
) -> tuple[int | None, list[Issue], list[Receipt]]:
    issues: list[Issue] = []
    valid: list[Receipt] = []
    version: int | None = None
    for receipt in receipts:
        candidate = receipt.resolved_param_version
        if version is None:
            version = candidate
            valid.append(receipt)
            continue
        if candidate != version:
            issues.append(
                Issue(
                    code=E_BATCH_VERSION_CONFLICT,
                    message=(
                        f"batch {batch_id!r} for worker {worker_id!r} mixes parameter versions "
                        f"{version} and {candidate}"
                    ),
                    path=receipt.source_path,
                    line=receipt.source_line,
                    item=receipt.key,
                    injection_id=",".join(receipt.injection_ids),
                )
            )
            continue
        valid.append(receipt)
    return version, issues, valid


def attribute_receipts(
    receipts: tuple[Receipt, ...],
    requests: tuple[RequestSpec, ...],
    config: AppConfig,
) -> AttributionResult:
    issues: list[Issue] = []
    valid_receipts, worker_issues = _resolve_worker_conflicts(receipts, config)
    issues.extend(worker_issues)
    rank_valid, rank_issues, rank_invalid = _rank_conflicts(valid_receipts)
    issues.extend(rank_issues)

    grouped: dict[tuple[str, str], list[Receipt]] = {}
    for receipt in rank_valid:
        worker_id = receipt.resolved_worker_id or UNKNOWN_WORKER
        grouped.setdefault((receipt.batch_id, worker_id), []).append(receipt)

    merge_pairs = {(entry.worker_id, entry.param_version) for entry in config.merge_entries}
    active_versions = {worker.worker_id: worker.param_version for worker in config.workers}
    batches: list[Batch] = []
    invalid_receipts: list[Receipt] = list(rank_invalid)
    selected_receipts: dict[str, list[Receipt]] = {}
    for (batch_id, worker_id), members in sorted(grouped.items()):
        version, version_issues, batch_members = _batch_version(
            batch_id, worker_id, tuple(sorted(members, key=lambda item: item.source_line)), config
        )
        issues.extend(version_issues)
        invalid_receipts.extend(item for item in members if item not in batch_members)
        if not batch_members:
            continue
        # One request key is a physical delivery slot within a batch. Keep the first
        # source row and diagnose later collisions instead of making the result
        # dependent on list ordering.
        unique_members: list[Receipt] = []
        seen_request_ids: dict[str, Receipt] = {}
        for receipt in sorted(batch_members, key=lambda item: (item.source_path, item.source_line)):
            previous = seen_request_ids.get(receipt.request_id)
            if previous is not None:
                issues.append(
                    Issue(
                        code=E_RECEIPT_MALFORMED,
                        message=(
                            f"duplicate delivery key {receipt.key}; first seen at "
                            f"{previous.source_path}:{previous.source_line}"
                        ),
                        path=receipt.source_path,
                        line=receipt.source_line,
                        item=receipt.key,
                        detail_code="DUPLICATE_DELIVERY_KEY",
                        injection_id=",".join(receipt.injection_ids),
                    )
                )
                invalid_receipts.append(receipt)
                continue
            seen_request_ids[receipt.request_id] = receipt
            unique_members.append(receipt)
        if not unique_members:
            continue
        effective_version = unique_members[0].resolved_param_version
        selected = (
            worker_id != UNKNOWN_WORKER
            and effective_version is not None
            and (worker_id, effective_version) in merge_pairs
        )
        active = active_versions.get(worker_id)
        if effective_version is None:
            invalidated = True
            reason = "param_version_missing"
        elif worker_id == UNKNOWN_WORKER or active is None:
            invalidated = True
            reason = "worker_unresolved"
        elif effective_version != active:
            invalidated = True
            reason = f"param_version_stale:{effective_version}!={active}"
        else:
            invalidated = False
            reason = "param_version_active"
        batch = Batch(
            batch_id=batch_id,
            worker_id=worker_id,
            param_version=effective_version,
            receipts=tuple(unique_members),
            request_ids=tuple(sorted({item.request_id for item in unique_members})),
            selected_by_merge_rule=selected,
            invalidated=invalidated,
            invalidation_reason=reason,
            source_files=tuple(sorted({item.source_path for item in unique_members})),
        )
        batches.append(batch)
        if selected:
            for receipt in unique_members:
                selected_receipts.setdefault(receipt.request_id, []).append(receipt)

    decisions: list[AuthorityDecision] = []
    for request_id in sorted(selected_receipts):
        candidates = sorted(
            selected_receipts[request_id],
            key=lambda item: (
                item.batch_id,
                item.resolved_worker_id or "",
                item.source_path,
                item.source_line,
            ),
        )
        decision = _decide_request(request_id, candidates)
        decisions.append(decision)
        issues.extend(decision_issues(decision))

    request_ids = {request.request_id for request in requests}
    uncovered = tuple(sorted(request_ids - set(selected_receipts)))
    return AttributionResult(
        batches=tuple(sorted(batches, key=lambda item: (item.batch_id, item.worker_id))),
        decisions=tuple(decisions),
        issues=tuple(sorted(issues, key=Issue.sort_key)),
        invalid_receipts=tuple(
            sorted(invalid_receipts, key=lambda item: (item.source_path, item.source_line))
        ),
        uncovered_request_ids=uncovered,
        batch_by_key={(batch.batch_id, batch.worker_id): batch for batch in batches},
    )


def decision_issues(decision: AuthorityDecision) -> list[Issue]:
    issues: list[Issue] = []
    for edge in decision.coverage_edges:
        if edge.get("status") == "invalid_reference":
            issues.append(
                Issue(
                    code=E_COVERAGE_REF_INVALID,
                    message=f"coverage reference {edge['target']!r} is not a candidate delivery",
                    path=edge.get("source_path", ""),
                    line=int(edge.get("source_line", 0)),
                    item=decision.request_id,
                    injection_id=edge.get("injection_id", ""),
                )
            )
    if decision.rule == "coverage_cycle_fallback":
        issues.append(
            Issue(
                code=E_COVERAGE_CYCLE,
                message="coverage references form a cycle; deterministic fallback applied",
                item=decision.request_id,
            )
        )
    return issues


def _decide_request(request_id: str, candidates: list[Receipt]) -> AuthorityDecision:
    by_key = {receipt.key: receipt for receipt in candidates}
    covers: dict[str, list[str]] = {receipt.key: [] for receipt in candidates}
    covered_by: dict[str, list[str]] = {receipt.key: [] for receipt in candidates}
    edges: list[dict[str, str]] = []
    for receipt in candidates:
        if is_missing(receipt.covered_refs):
            continue
        for target in receipt.covered_refs:
            if target not in by_key:
                edges.append(
                    {
                        "source": receipt.key,
                        "target": target,
                        "status": "invalid_reference",
                        "source_path": receipt.source_path,
                        "source_line": str(receipt.source_line),
                        "injection_id": ",".join(receipt.injection_ids),
                    }
                )
                continue
            if target == receipt.key:
                edges.append(
                    {
                        "source": receipt.key,
                        "target": target,
                        "status": "self_reference",
                        "source_path": receipt.source_path,
                        "source_line": str(receipt.source_line),
                        "injection_id": ",".join(receipt.injection_ids),
                    }
                )
                continue
            covers[receipt.key].append(target)
            covered_by[target].append(receipt.key)
            edges.append(
                {
                    "source": receipt.key,
                    "target": target,
                    "status": "resolved",
                    "source_path": receipt.source_path,
                    "source_line": str(receipt.source_line),
                    "injection_id": ",".join(receipt.injection_ids),
                }
            )

    roots = [receipt for receipt in candidates if not covered_by[receipt.key]]
    if len(candidates) == 1:
        authority = candidates[0]
        rule = "single_delivery"
        reason = {"detail": "only one selected delivery exists"}
    elif not any(covers.values()):
        ordered = sorted(candidates, key=_version_sort)
        authority = ordered[0]
        rule = "param_version_desc_batch_id_asc"
        reason = {
            "detail": "no coverage reference; parameter version descending, batch id ascending",
            "comparison_key": [
                {
                    "key": receipt.key,
                    "param_version": receipt.resolved_param_version,
                    "batch_id": receipt.batch_id,
                }
                for receipt in ordered
            ],
        }
    elif len(roots) == 1:
        authority = roots[0]
        rule = "coverage_ref"
        reason = {
            "detail": "unique rooted delivery after applying r3 coverage references",
            "coverage_roots": [roots[0].key],
        }
    elif roots:
        ordered = sorted(roots, key=lambda item: (_duration_sort(item), _version_sort(item)))
        authority = ordered[0]
        rule = "coverage_ref_then_duration_ms_asc"
        reason = {
            "detail": (
                "multiple coverage roots; finite integer duration ascending, then parameter "
                "version descending and batch id ascending"
            ),
            "comparison_key": [
                {
                    "key": receipt.key,
                    "duration_ms": (
                        None if is_missing(receipt.duration_ms) else receipt.duration_ms
                    ),
                    "param_version": receipt.resolved_param_version,
                    "batch_id": receipt.batch_id,
                }
                for receipt in ordered
            ],
        }
    else:
        ordered = sorted(candidates, key=_version_sort)
        authority = ordered[0]
        rule = "coverage_cycle_fallback"
        reason = {
            "detail": "coverage graph has no root; deterministic parameter-version fallback",
            "comparison_key": [item.key for item in ordered],
        }

    deliveries: list[Delivery] = []
    for receipt in candidates:
        if receipt.key == authority.key:
            status = "authority"
        elif covered_by[receipt.key]:
            status = "covered"
        else:
            status = "losing_root"
        deliveries.append(
            Delivery(
                receipt=receipt,
                selected=True,
                status=status,
                covers=tuple(covers[receipt.key]),
                covered_by=tuple(covered_by[receipt.key]),
            )
        )
    return AuthorityDecision(
        request_id=request_id,
        authority=authority,
        rule=rule,
        reason=reason,
        deliveries=tuple(
            sorted(
                deliveries,
                key=lambda item: (
                    item.receipt.batch_id,
                    item.receipt.resolved_worker_id or "",
                    item.receipt.source_path,
                    item.receipt.source_line,
                ),
            )
        ),
        coverage_edges=tuple(sorted(edges, key=lambda item: (item["source"], item["target"]))),
        candidate_count=len(candidates),
    )
