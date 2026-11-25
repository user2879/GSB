"""End-to-end local and socket-backed batch pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .engine import BatchPayload, RawBatchResult, infer_all_local, make_batch_payloads
from .errors import DiagnosticError
from .invalidation import REPLAY_IGNORED_FIELDS
from .leases import Lease, LeaseScheduler
from .ledger import Ledger
from .orchestration import WorkerProcess, probe_workload_ports, start_workers, stop_workers
from .receipts import NormalizedReceipt, context_for, normalize_receipt
from .spec import WorkloadSpec
from .synthetic import SyntheticBundle


@dataclass
class SocketRun:
    raw_receipts: dict[str, dict[str, Any]]
    normalized: dict[str, NormalizedReceipt]
    lease_snapshot: dict[str, Any]
    lease_replay: dict[str, Any]
    port_probe: dict[str, Any]
    commits: list[dict[str, Any]]


def normalize_local_results(
    spec: WorkloadSpec,
    bundle: SyntheticBundle,
    *,
    batch_ids: Iterable[str] | None = None,
) -> tuple[dict[str, RawBatchResult], dict[str, NormalizedReceipt]]:
    raw = infer_all_local(spec, bundle, batch_ids=batch_ids)
    normalized = {
        batch_id: normalize_receipt(result.receipt, context=context_for(spec, batch_id))
        for batch_id, result in raw.items()
    }
    return raw, normalized


def commit_receipts(
    ledger: Ledger,
    spec: WorkloadSpec,
    bundle: SyntheticBundle,
    receipts: Mapping[str, NormalizedReceipt],
) -> list[dict[str, Any]]:
    truth_by_window = {window.window_id: window for window in bundle.windows}
    ordinal_by_batch = {request.batch_id: request.request_index for request in spec.requests}
    missing = sorted(set(receipts) - set(ordinal_by_batch))
    if missing:
        raise DiagnosticError("E_REPLAY", "receipt batch is not declared by the specification", field="batch_id", details={"batch_id": missing[0]})
    return ledger.commit_many(receipts, truth_by_window, ordinal_by_batch)


def run_local_ledger(
    spec: WorkloadSpec,
    bundle: SyntheticBundle,
    ledger: Ledger,
) -> tuple[dict[str, RawBatchResult], dict[str, NormalizedReceipt], list[dict[str, Any]]]:
    raw, normalized = normalize_local_results(spec, bundle)
    commits = commit_receipts(ledger, spec, bundle, normalized)
    return raw, normalized, commits


def execute_over_loopback(
    spec: WorkloadSpec,
    bundle: SyntheticBundle,
    spec_path: str | Path,
    *,
    ledger: Ledger | None = None,
    only_worker_ids: list[str] | None = None,
) -> SocketRun:
    port_probe = probe_workload_ports(spec)
    payloads = list(make_batch_payloads(spec, bundle))
    if only_worker_ids is not None:
        selected_workers = set(only_worker_ids)
        payloads = [payload for payload in payloads if payload.worker_id in selected_workers]
    scheduler = LeaseScheduler.from_spec(spec)
    workers: dict[str, WorkerProcess] = {}
    try:
        workers = start_workers(spec, spec_path, worker_ids=only_worker_ids)

        def action(batch_id: str, worker_id: str, lease: Lease, extra: BatchPayload) -> dict[str, Any]:
            client = workers[worker_id]
            receipt = client.request_batch(extra)
            normalized = normalize_receipt(receipt, context=context_for(spec, batch_id))
            return {"receipt": receipt, "normalized": normalized, "lease": lease.to_dict(scheduler.origin_ns)}

        tasks = [(payload.batch_id, payload.worker_id, payload) for payload in payloads]
        executed = scheduler.apply_parallel(
            tasks,
            action,
            max_workers=max(1, min(len(tasks), spec.limits.global_batch_leases)),
            timeout_ms_each=spec.limits.io_timeout_ms + spec.limits.handshake_timeout_ms,
        )
    finally:
        stop_workers(workers)

    raw_receipts: dict[str, dict[str, Any]] = {}
    normalized: dict[str, NormalizedReceipt] = {}
    for payload in payloads:
        value = executed[payload.batch_id]["value"]
        raw_receipts[payload.batch_id] = value["receipt"]
        normalized[payload.batch_id] = value["normalized"]
    commits: list[dict[str, Any]] = []
    if ledger is not None:
        commits = commit_receipts(ledger, spec, bundle, normalized)
    return SocketRun(
        raw_receipts=raw_receipts,
        normalized=normalized,
        lease_snapshot=scheduler.snapshot(),
        lease_replay=scheduler.replay(),
        port_probe=port_probe,
        commits=commits,
    )


def replay_local_against_ledger(
    spec: WorkloadSpec,
    bundle: SyntheticBundle,
    ledger: Ledger,
) -> dict[str, Any]:
    _, regenerated = normalize_local_results(spec, bundle)
    expected_order = [request.batch_id for request in spec.requests]
    return ledger.verify_replay(
        regenerated,
        expected_order=expected_order,
        ignore_fields=REPLAY_IGNORED_FIELDS,
    )
