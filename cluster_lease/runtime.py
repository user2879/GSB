"""End-to-end run, recovery and parameter-evolution orchestration."""

from __future__ import annotations

import os
import uuid
from typing import Any

from .canonical import LEDGER_FORMAT_VERSION, canonical_bytes, sha256_bytes
from .errors import LeaseError
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .models import LeaseEvent, SpecBundle
from .orchestrator import ServiceOrchestrator
from .state_machine import event_to_request


def _versions_and_priorities(
    bundle: SpecBundle, ledger: Ledger
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    versions = ledger.versions()
    missing = [cluster.cluster_id for cluster in bundle.clusters if cluster.cluster_id not in versions]
    if missing:
        raise LeaseError("E_STATE", f"clusters are not registered: {missing}", field="clusters")
    priorities = {
        cluster.cluster_id: bundle.priorities_for(cluster.cluster_id)
        for cluster in bundle.clusters
    }
    return versions, priorities


def _start_services(
    bundle: SpecBundle,
    ledger: Ledger,
    runtime_dir: str,
    logger: StructuredLogger,
    *,
    clock_mode: str,
    startup_timeout_ms: int,
    handshake_timeout_ms: int,
) -> ServiceOrchestrator:
    versions, priorities = _versions_and_priorities(bundle, ledger)
    orchestrator = ServiceOrchestrator(
        runtime_dir,
        clock_mode=clock_mode,
        startup_timeout_ms=startup_timeout_ms,
        handshake_timeout_ms=handshake_timeout_ms,
        log_callback=logger.service_line,
    )
    orchestrator.start(list(bundle.clusters), priorities, versions)
    return orchestrator


def _restore_services(
    bundle: SpecBundle,
    ledger: Ledger,
    orchestrator: ServiceOrchestrator,
    logger: StructuredLogger,
) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    for spec in sorted(bundle.clusters, key=lambda item: item.cluster_id):
        progress = ledger.progress_for(spec.cluster_id)
        active = ledger.current_active(spec.cluster_id)
        leases = [
            {
                "expires_ms": row["expires_ms"],
                "holder_id": row["holder_id"],
                "last_event_seq": row["last_event_seq"],
                "last_renewed_ms": row["last_renewed_ms"],
                "priority": row["priority"],
                "renewal_count": row["renewal_count"],
                "started_ms": row["started_ms"],
                "state": row["state"],
            }
            for row in active
        ]
        reply = orchestrator.get(spec.cluster_id).restore(
            leases,
            int(progress["last_event_seq"]),
            int(progress["last_timestamp_ms"]),
        )
        for continuation in reply.get("continuations", []):
            logger.log(
                "RECOVERY_CONTINUATION",
                "active lease reconstructed from committed ledger",
                cluster_id=spec.cluster_id,
                continuation=continuation["continuation"],
                holder_id=continuation["holder_id"],
                recorded_expires_ms=continuation["recorded_expires_ms"],
                recomputed_expires_ms=continuation["recomputed_expires_ms"],
            )
        clusters.append(
            {
                "cluster_id": spec.cluster_id,
                "committed_event_seq": int(progress["last_event_seq"]),
                "last_timestamp_ms": int(progress["last_timestamp_ms"]),
                "restored_active_leases": reply.get("active_holders", []),
                "continuations": reply.get("continuations", []),
            }
        )
    return {"clusters": clusters, "restored": True}


def _decision_trace(
    events: list[LeaseEvent], responses: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    trace = []
    for event in sorted(events, key=lambda item: item.event_seq):
        response = responses[event.event_seq]
        trace.append(
            {
                "active_holder_ids": [item["holder_id"] for item in response["active_holders"]],
                "cluster_id": event.cluster_id,
                "decision": response["decision"],
                "event_seq": event.event_seq,
                "reason": response["reason"],
                "used_capacity": response["used_capacity"],
                "watermark_fixed": response["watermark_fixed"],
            }
        )
    return trace


def _reply_digest(responses: dict[int, dict[str, Any]]) -> str:
    payload = b"".join(
        canonical_bytes(responses[event_seq]) + b"\n" for event_seq in sorted(responses)
    )
    return sha256_bytes(payload)


def _report_snapshot(ledger: Ledger, bundle: SpecBundle) -> dict[str, Any]:
    clusters = []
    for spec in sorted(bundle.clusters, key=lambda item: item.cluster_id):
        progress = ledger.progress_for(spec.cluster_id)
        clusters.append(
            {
                "active_leases": ledger.current_active(spec.cluster_id),
                "capacity": spec.capacity,
                "cluster_id": spec.cluster_id,
                "parameter_version": int(progress["parameter_version"]),
                "ownership_intervals": ledger.current_ownership_intervals(
                    spec.cluster_id
                ),
                "progress": {
                    "last_event_seq": int(progress["last_event_seq"]),
                    "last_timestamp_ms": int(progress["last_timestamp_ms"]),
                },
                "watermark_curve": ledger.current_watermarks(spec.cluster_id),
            }
        )
    return {
        "clusters": clusters,
        "format_version": LEDGER_FORMAT_VERSION,
        "ledger_content_digest": ledger.content_digest(),
        "ledger_format_version": int(ledger.get_meta("ledger_format_version", "1") or 1),
        "stream_cursor": ledger.stream_cursor(),
        "event_stream_digest": bundle.event_stream_digest(),
    }


def run_session(
    bundle: SpecBundle,
    db_path: str,
    *,
    runtime_dir: str,
    logger: StructuredLogger,
    batch_size: int = 4,
    clock_mode: str = "hybrid",
    startup_timeout_ms: int = 5000,
    handshake_timeout_ms: int = 2000,
    crash_after_inflight_batch: int | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise LeaseError("E_SPEC", "batch_size must be positive", field="batch_size")
    ledger = Ledger(db_path)
    orchestrator: ServiceOrchestrator | None = None
    try:
        ledger.register_bundle(bundle)
        ledger.mark_inflight_unknown()
        unknown_batches = ledger.batch_runs("unknown")
        for batch in unknown_batches:
            logger.log(
                "RECOVERY_UNKNOWN_BATCH",
                "uncommitted events marked unknown and required to rerun",
                batch_id=batch["batch_id"],
                from_seq=batch["from_seq"],
                to_seq=batch["to_seq"],
                rerun_required=True,
            )
        orchestrator = _start_services(
            bundle,
            ledger,
            runtime_dir,
            logger,
            clock_mode=clock_mode,
            startup_timeout_ms=startup_timeout_ms,
            handshake_timeout_ms=handshake_timeout_ms,
        )
        recovery = _restore_services(bundle, ledger, orchestrator, logger)
        cursor = ledger.stream_cursor()
        pending = [event for event in bundle.events if event.event_seq > cursor]
        processed_events: list[LeaseEvent] = []
        processed_responses: dict[int, dict[str, Any]] = {}
        batch_number = 0
        for offset in range(0, len(pending), batch_size):
            batch_number += 1
            batch = pending[offset : offset + batch_size]
            batch_id = uuid.uuid4().hex
            commit_id = uuid.uuid4().hex
            ledger.record_inflight(
                batch_id,
                batch,
                sorted({event.cluster_id for event in batch}),
            )
            if crash_after_inflight_batch is not None and batch_number == crash_after_inflight_batch:
                logger.log(
                    "FAULT_INJECTED_FORCED_TERMINATION",
                    "process exits after uncommitted in-flight batch marker",
                    batch_id=batch_id,
                    from_seq=batch[0].event_seq,
                    to_seq=batch[-1].event_seq,
                )
                os._exit(70)
            responses: dict[int, dict[str, Any]] = {}
            for event in batch:
                reply = orchestrator.get(event.cluster_id).request(event_to_request(event))
                if int(reply["event_seq"]) != event.event_seq:
                    raise LeaseError(
                        "E_REPLY",
                        "reply event_seq does not match request",
                        field="reply.event_seq",
                    )
                if reply["cluster_id"] != event.cluster_id or reply["holder_id"] != event.holder_id:
                    raise LeaseError(
                        "E_REPLY",
                        "reply cluster_id or holder_id does not match the request",
                        field="reply",
                        details={
                            "cluster_id": event.cluster_id,
                            "event_seq": event.event_seq,
                            "holder_id": event.holder_id,
                        },
                    )
                responses[event.event_seq] = reply
                logger.log(
                    "LEASE_DECISION",
                    "lease decision received",
                    active_holder_ids=[item["holder_id"] for item in reply["active_holders"]],
                    cluster_id=event.cluster_id,
                    decision=reply["decision"],
                    event_seq=event.event_seq,
                    holder_id=event.holder_id,
                    reason=reply["reason"],
                    used_capacity=reply["used_capacity"],
                    watermark_fixed=reply["watermark_fixed"],
                )
            ledger.commit_batch(batch_id, batch, responses, commit_id)
            processed_events.extend(batch)
            processed_responses.update(responses)
        unknown_after = [
            row for row in ledger.batch_runs() if row["status"] in ("unknown", "inflight")
        ]
        state = _report_snapshot(ledger, bundle)
        return {
            "code": "OK",
            "decision_trace": _decision_trace(processed_events, processed_responses),
            "ledger": state,
            "ok": True,
            "processed_event_count": len(processed_events),
            "processed_event_sequences": [event.event_seq for event in processed_events],
            "recovery": recovery,
            "reply_sequence_digest": _reply_digest(processed_responses),
            "unknown_batches": unknown_after,
        }
    finally:
        if orchestrator is not None:
            orchestrator.stop()
        ledger.close()


def recover_session(
    bundle: SpecBundle,
    db_path: str,
    *,
    runtime_dir: str,
    logger: StructuredLogger,
    clock_mode: str = "logical",
    startup_timeout_ms: int = 5000,
    handshake_timeout_ms: int = 2000,
) -> dict[str, Any]:
    ledger = Ledger(db_path)
    orchestrator: ServiceOrchestrator | None = None
    try:
        ledger.register_bundle(bundle)
        newly_unknown = ledger.mark_inflight_unknown()
        unknown_batches = ledger.batch_runs("unknown")
        for batch in unknown_batches:
            logger.log(
                "RECOVERY_UNKNOWN_BATCH",
                "uncommitted events marked unknown and required to rerun",
                batch_id=batch["batch_id"],
                from_seq=batch["from_seq"],
                rerun_required=True,
                to_seq=batch["to_seq"],
            )
        orchestrator = _start_services(
            bundle,
            ledger,
            runtime_dir,
            logger,
            clock_mode=clock_mode,
            startup_timeout_ms=startup_timeout_ms,
            handshake_timeout_ms=handshake_timeout_ms,
        )
        recovery = _restore_services(bundle, ledger, orchestrator, logger)
        return {
            "code": "OK",
            "exit_report_available": False,
            "message": (
                "commit-only recovery completed; truth for an abrupt termination can only come "
                "from committed records, and unknown ranges must be rerun"
            ),
            "newly_marked_unknown_batches": newly_unknown,
            "ok": True,
            "recovery": recovery,
            "state": _report_snapshot(ledger, bundle),
            "unknown_batches": unknown_batches,
        }
    finally:
        if orchestrator is not None:
            orchestrator.stop()
        ledger.close()


def evolve_session(
    bundle: SpecBundle,
    db_path: str,
    *,
    runtime_dir: str,
    logger: StructuredLogger,
    startup_timeout_ms: int = 5000,
    handshake_timeout_ms: int = 2000,
) -> dict[str, Any]:
    ledger = Ledger(db_path)
    orchestrator: ServiceOrchestrator | None = None
    try:
        existing_rows = {row["cluster_id"]: row for row in ledger.cluster_versions()}
        new_ids = {spec.cluster_id for spec in bundle.clusters}
        removed = sorted(set(existing_rows) - new_ids)
        if removed:
            raise LeaseError(
                "E_STATE",
                f"removing clusters requires a separate migration: {removed}",
                field="clusters",
            )
        affected: list[dict[str, Any]] = []
        for spec in sorted(bundle.clusters, key=lambda item: item.cluster_id):
            priority_digest = bundle.priority_digest(spec.cluster_id)
            old = existing_rows.get(spec.cluster_id)
            if old is None:
                affected.append(
                    {
                        "basis": [{"field": "cluster", "new": "added", "old": None}],
                        "cluster_id": spec.cluster_id,
                        "from_version": 0,
                        "old": None,
                        "priority_digest": priority_digest,
                        "spec": spec,
                        "to_version": 1,
                    }
                )
                continue
            differences = Ledger._spec_differences(old, spec, priority_digest)
            if differences:
                affected.append(
                    {
                        "basis": differences,
                        "cluster_id": spec.cluster_id,
                        "from_version": int(old["parameter_version"]),
                        "old": dict(old),
                        "priority_digest": priority_digest,
                        "spec": spec,
                        "to_version": int(old["parameter_version"]) + 1,
                    }
                )
        if not affected:
            return {
                "affected_clusters": [],
                "code": "OK",
                "message": "no parameter version changes detected",
                "ok": True,
                "replacement": "not_required",
            }

        affected_ids = [item["cluster_id"] for item in affected]
        old_active = {
            cluster_id: ledger.current_active(cluster_id) for cluster_id in affected_ids
        }
        old_stale_before = ledger.stale_records(affected_ids)
        versions = ledger.versions()
        for item in affected:
            versions[item["cluster_id"]] = item["to_version"]
        affected_specs = [item["spec"] for item in affected]
        priorities = {
            spec.cluster_id: bundle.priorities_for(spec.cluster_id)
            for spec in affected_specs
        }
        orchestrator = ServiceOrchestrator(
            runtime_dir,
            clock_mode="logical",
            startup_timeout_ms=startup_timeout_ms,
            handshake_timeout_ms=handshake_timeout_ms,
            log_callback=logger.service_line,
        )
        orchestrator.start(affected_specs, priorities, versions)

        plans: dict[str, dict[str, Any]] = {}
        stream_cursor = ledger.stream_cursor()
        for item in affected:
            cluster_id = item["cluster_id"]
            progress = ledger.progress_for(cluster_id)
            cursor = (
                int(progress["last_event_seq"])
                if item["old"] is not None
                else stream_cursor
            )
            events = bundle.events_for(cluster_id, max_seq=cursor)
            responses: dict[int, dict[str, Any]] = {}
            for event in events:
                responses[event.event_seq] = orchestrator.get(cluster_id).request(
                    event_to_request(event)
                )
            plans[cluster_id] = {
                "events": events,
                "last_event_seq": cursor,
                "last_timestamp_ms": (
                    events[-1].timestamp_ms if events else int(progress["last_timestamp_ms"])
                ),
                "new_version": item["to_version"],
                "old_version": item["from_version"] if item["old"] is not None else None,
                "responses": responses,
            }
            logger.log(
                "PARAMETER_VERSION_REPLAYED",
                "cluster events replayed under new parameters",
                cluster_id=cluster_id,
                event_count=len(events),
                from_version=item["from_version"],
                to_version=item["to_version"],
            )
        orchestrator.stop()
        orchestrator = None

        commit_id = uuid.uuid4().hex
        all_specs = {spec.cluster_id: spec for spec in bundle.clusters}
        ledger.commit_evolution(plans, all_specs, bundle, commit_id)
        stale_after = ledger.stale_records(affected_ids)
        retained = {
            "decisions": stale_after["decisions"],
            "ownership_intervals": stale_after["ownership_intervals"],
            "watermarks": stale_after["watermarks"],
        }
        discarded = {
            "active_leases": old_active,
            "ownership_intervals_excluded_from_current_view": [
                row for row in stale_after["ownership_intervals"]
                if row not in old_stale_before["ownership_intervals"]
            ],
            "watermarks_excluded_from_current_view": [
                row for row in stale_after["watermarks"]
                if row not in old_stale_before["watermarks"]
            ],
        }
        affected_report = []
        for item in affected:
            cluster_id = item["cluster_id"]
            affected_report.append(
                {
                    "basis": item["basis"],
                    "cluster_id": cluster_id,
                    "from_version": item["from_version"],
                    "new_active_leases": ledger.current_active(cluster_id),
                    "to_version": item["to_version"],
                }
            )
        return {
            "affected_clusters": affected_report,
            "code": "OK",
            "discarded_stale_records": discarded,
            "new_ledger_digest": ledger.content_digest(),
            "ok": True,
            "replacement": "overall_replace",
            "retained_read_only_records": retained,
            "revoked_active_leases": [
                lease
                for cluster_id in sorted(old_active)
                for lease in old_active[cluster_id]
            ],
        }
    finally:
        if orchestrator is not None:
            orchestrator.stop()
        ledger.close()
