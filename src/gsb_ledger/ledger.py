"""Canonical ledger assembly from validated stage products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .attribution import AttributionResult, AuthorityDecision, Batch
from .config import config_hash, requests_hash
from .errors import E_RECEIPT_MISSING, Issue
from .model import AppConfig, RequestSpec, jsonable
from .receipts import ReceiptLoadResult
from .resources import ResourceMetrics
from .util import digest_value


LEDGER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LedgerBuild:
    ledger: dict[str, Any]
    ledger_version: str
    entries: tuple[dict[str, Any], ...]


def _resource_lookup(resources: tuple[ResourceMetrics, ...]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(item.batch_id, item.worker_id): item.to_dict() for item in resources}


def _authority_payload(decision: AuthorityDecision) -> dict[str, Any]:
    receipt = decision.authority
    return {
        "batch_id": receipt.batch_id,
        "worker_id": receipt.resolved_worker_id,
        "param_version": receipt.resolved_param_version,
        "output_digest": receipt.output_digest,
        "duration_ms": jsonable(receipt.duration_ms),
        "stable_rank": jsonable(receipt.rank),
        "covered_refs": jsonable(receipt.covered_refs),
        "param_digest": jsonable(receipt.param_digest),
        "source_version": receipt.version,
        "missing_fields": list(receipt.missing_fields),
        "completeness": receipt.completeness,
        "source": {
            "path": receipt.source_path,
            "line": receipt.source_line,
        },
    }


def _batch_payload(batch: Batch, resource: dict[str, Any] | None) -> dict[str, Any]:
    payload = batch.to_dict()
    payload["resource"] = resource
    return payload


def _request_lookup(requests: tuple[RequestSpec, ...]) -> dict[str, RequestSpec]:
    return {request.request_id: request for request in requests}


def assemble_ledger(
    *,
    config: AppConfig,
    requests: tuple[RequestSpec, ...],
    receipt_load: ReceiptLoadResult,
    attribution: AttributionResult,
    resources: tuple[ResourceMetrics, ...],
    input_fingerprint: str,
    extra_issues: tuple[Issue, ...] = (),
    diagnostics: tuple[dict[str, Any], ...] = (),
    degradations: tuple[dict[str, Any], ...] = (),
) -> LedgerBuild:
    resource_by_batch = _resource_lookup(resources)
    request_by_id = _request_lookup(requests)
    batch_by_key = {(batch.batch_id, batch.worker_id): batch for batch in attribution.batches}
    active_versions = {worker.worker_id: worker.param_version for worker in config.workers}

    issues: list[Issue] = []
    issues.extend(receipt_load.issues)
    issues.extend(attribution.issues)
    issues.extend(extra_issues)
    for request_id in attribution.uncovered_request_ids:
        request = request_by_id.get(request_id)
        issues.append(
            Issue(
                code=E_RECEIPT_MISSING,
                message="no selected receipt can supply an authoritative batch",
                path="" if request is None else request.source_path,
                line=0 if request is None else request.source_line,
                item=request_id,
            )
        )

    decisions = {decision.request_id: decision for decision in attribution.decisions}
    entries: list[dict[str, Any]] = []
    for request in sorted(requests, key=lambda item: item.request_id):
        decision = decisions.get(request.request_id)
        if decision is None:
            entries.append(
                {
                    "entry_id": request.request_id,
                    "request_id": request.request_id,
                    "status": "uncovered",
                    "request": request.to_dict(),
                    "authority": None,
                    "authority_rule": "uncovered",
                    "authority_reason": {
                        "detail": "no selected delivery after merge-rule filtering and validation"
                    },
                    "deliveries": [],
                    "dependency": {
                        "worker_id": None,
                        "batch_id": None,
                        "param_version": None,
                        "active_param_version": None,
                        "batch_invalidated": True,
                        "recompute_required": True,
                        "reuse_eligible": False,
                    },
                    "resource": None,
                }
            )
            continue
        authority = decision.authority
        batch = batch_by_key.get((authority.batch_id, authority.resolved_worker_id or ""))
        if batch is None:
            batch = next(
                (
                    candidate
                    for candidate in attribution.batches
                    if candidate.batch_id == authority.batch_id
                    and candidate.worker_id == (authority.resolved_worker_id or "")
                ),
                None,
            )
        active_version = active_versions.get(authority.resolved_worker_id or "")
        resource = resource_by_batch.get((authority.batch_id, authority.resolved_worker_id or ""))
        entry = {
            "entry_id": request.request_id,
            "request_id": request.request_id,
            "status": "authoritative",
            "request": request.to_dict(),
            "authority": _authority_payload(decision),
            "authority_rule": decision.rule,
            "authority_reason": decision.reason,
            "deliveries": [
                {
                    "key": delivery.receipt.key,
                    "batch_id": delivery.receipt.batch_id,
                    "worker_id": delivery.receipt.resolved_worker_id,
                    "param_version": delivery.receipt.resolved_param_version,
                    "duration_ms": jsonable(delivery.receipt.duration_ms),
                    "rank": jsonable(delivery.receipt.rank),
                    "status": delivery.status,
                    "covers": list(delivery.covers),
                    "covered_by": list(delivery.covered_by),
                    "covered_refs": jsonable(delivery.receipt.covered_refs),
                    "source_version": delivery.receipt.version,
                    "missing_fields": list(delivery.receipt.missing_fields),
                    "source": {
                        "path": delivery.receipt.source_path,
                        "line": delivery.receipt.source_line,
                    },
                }
                for delivery in decision.deliveries
            ],
            "coverage_edges": list(decision.coverage_edges),
            "dependency": {
                "worker_id": authority.resolved_worker_id,
                "batch_id": authority.batch_id,
                "param_version": authority.resolved_param_version,
                "active_param_version": active_version,
                "batch_invalidated": bool(batch.invalidated) if batch else True,
                "batch_invalidation_reason": batch.invalidation_reason if batch else "batch_missing",
                "recompute_required": bool(batch.invalidated) if batch else True,
                "reuse_eligible": bool(batch and not batch.invalidated),
            },
            "resource": resource,
        }
        entries.append(entry)

    batch_payloads: list[dict[str, Any]] = []
    invalidated_batches: list[dict[str, Any]] = []
    reused_batches: list[dict[str, Any]] = []
    excluded_batches: list[dict[str, Any]] = []
    invalidated_request_ids: set[str] = set()
    reused_request_ids: set[str] = set()
    batch_decisions: dict[tuple[str, str], list[str]] = {}
    for decision in attribution.decisions:
        batch_decisions.setdefault(
            (decision.authority.batch_id, decision.authority.resolved_worker_id or ""), []
        ).append(decision.request_id)
    for batch in attribution.batches:
        resource = resource_by_batch.get((batch.batch_id, batch.worker_id))
        payload = _batch_payload(batch, resource)
        payload["affected_request_ids"] = list(batch.request_ids)
        payload["authoritative_request_ids"] = sorted(
            batch_decisions.get((batch.batch_id, batch.worker_id), [])
        )
        batch_payloads.append(payload)
        if not batch.selected_by_merge_rule:
            excluded_batches.append(payload)
            continue
        if batch.invalidated:
            invalidated_batches.append(payload)
            invalidated_request_ids.update(payload["authoritative_request_ids"])
        else:
            reused_batches.append(payload)
            reused_request_ids.update(payload["authoritative_request_ids"])

    invalidated_batches.sort(key=lambda item: (item["batch_id"], item["worker_id"]))
    reused_batches.sort(key=lambda item: (item["batch_id"], item["worker_id"]))
    excluded_batches.sort(key=lambda item: (item["batch_id"], item["worker_id"]))
    failures = [issue.to_dict() for issue in sorted(issues, key=Issue.sort_key)]
    degradation_list = sorted(
        degradations,
        key=lambda item: (
            item.get("receipt_key", ""),
            item.get("source_version", ""),
            item.get("target_version", ""),
        ),
    )
    diagnostic_list = sorted(diagnostics, key=lambda item: item.get("injection_id", ""))
    authority_rules: dict[str, int] = {}
    for entry in entries:
        rule = entry["authority_rule"]
        authority_rules[rule] = authority_rules.get(rule, 0) + 1
    summary = {
        "workers": len(config.workers),
        "receipts_loaded": len(receipt_load.receipts),
        "batches_total": len(batch_payloads),
        "batches_selected": len(attribution.selected_batches),
        "batches_invalidated": len(invalidated_batches),
        "batches_reused": len(reused_batches),
        "batches_excluded": len(excluded_batches),
        "requests_total": len(requests),
        "entries_authoritative": sum(1 for entry in entries if entry["status"] == "authoritative"),
        "entries_uncovered": sum(1 for entry in entries if entry["status"] == "uncovered"),
        "invalidated_requests": len(invalidated_request_ids),
        "reused_requests": len(reused_request_ids),
        "failures": len(failures),
        "diagnostic_hits": sum(1 for item in diagnostic_list if item.get("hit")),
        "degradations": len(degradation_list),
        "authority_rule_counts": dict(sorted(authority_rules.items())),
    }
    worker_payload = []
    for worker in sorted(config.workers, key=lambda item: item.worker_id):
        worker_payload.append(
            {
                **worker.to_dict(),
                "known_param_versions": list(
                    worker.known_param_versions or (worker.param_version,)
                ),
            }
        )
    ledger_material = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "input_fingerprint": input_fingerprint,
        "config_hash": config_hash(config),
        "requests_hash": requests_hash(requests),
        "receipt_fingerprints": dict(sorted(receipt_load.file_fingerprints.items())),
        "workers": worker_payload,
        "merge_rule": {
            "entries": [entry.to_dict() for entry in config.merge_entries],
            "tie_break": list(config.tie_break),
        },
        "resource_policy": {
            "fixed_point_bits": config.fixed_point_bits,
            "throughput_ops_per_ms": config.throughput_ops_per_ms,
            "integer_only": True,
        },
        "batches": batch_payloads,
        "entries": entries,
        "invalidation": {
            "worker_param_versions": dict(sorted(active_versions.items())),
            "invalidated_batch_ids": [
                f"{item['batch_id']}/{item['worker_id']}" for item in invalidated_batches
            ],
            "reused_batch_ids": [
                f"{item['batch_id']}/{item['worker_id']}" for item in reused_batches
            ],
            "excluded_batch_ids": [
                f"{item['batch_id']}/{item['worker_id']}" for item in excluded_batches
            ],
            "invalidated_request_ids": sorted(invalidated_request_ids),
            "reused_request_ids": sorted(reused_request_ids),
            "invalidated_scope_request_ids": sorted(
                {
                    request_id
                    for item in invalidated_batches
                    for request_id in item["request_ids"]
                }
            ),
            "dependency_chain": [
                {
                    "worker_id": worker.worker_id,
                    "param_version": worker.param_version,
                    "batch_ids": [
                        f"{batch.batch_id}/{batch.worker_id}"
                        for batch in attribution.batches
                        if batch.worker_id == worker.worker_id
                    ],
                }
                for worker in sorted(config.workers, key=lambda item: item.worker_id)
            ],
        },
        "failures": failures,
        "diagnostics": diagnostic_list,
        "degradations": degradation_list,
        "summary": summary,
    }
    ledger_version = digest_value(ledger_material)
    ledger = {
        **ledger_material,
        "ledger_version": ledger_version,
    }
    return LedgerBuild(ledger=ledger, ledger_version=ledger_version, entries=tuple(entries))
