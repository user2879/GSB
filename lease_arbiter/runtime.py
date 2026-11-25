from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_json, digest
from .engine import LeaseArbiterEngine, MutationBatch, choose_path
from .errors import LeaseError, LedgerError
from .ledger import Ledger
from .scheduler import QuotaScheduler
from .spec import Event, NodeSpec, Spec
from .transport import JsonLineNodeClient, LiveNodeCluster


@dataclass
class NodeAction:
    op: str
    lease_id: str
    node_id: str
    lease_type: str
    expected_label: int
    expected_response_label: int
    trigger_seq: int
    reason: str

    def request(self, request_id: str) -> dict[str, Any]:
        return {
            "op": self.op,
            "lease_id": self.lease_id,
            "lease_type": self.lease_type,
            "expected_label": self.expected_label,
            "request_id": request_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "lease_id": self.lease_id,
            "node_id": self.node_id,
            "lease_type": self.lease_type,
            "expected_label": self.expected_label,
            "expected_response_label": self.expected_response_label,
            "trigger_seq": self.trigger_seq,
            "reason": self.reason,
        }


def _wait_until_monotonic(start_ns: int, at_ms: int) -> int:
    target_ns = start_ns + at_ms * 1_000_000
    while True:
        now_ns = time.monotonic_ns()
        remaining_ns = target_ns - now_ns
        if remaining_ns <= 0:
            return (now_ns - start_ns) // 1_000_000
        time.sleep(min(remaining_ns / 1_000_000_000, 0.05))


def _partition_result(engine: LeaseArbiterEngine, event_seq: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transitions = [item for item in engine.transitions if item["event_seq"] == event_seq]
    ownership = [item for item in engine.ownership_changes if item["event_seq"] == event_seq]
    return transitions, ownership


def _actions_for_batches(
    expiry: MutationBatch,
    applied: MutationBatch,
    event: Event,
) -> list[NodeAction]:
    actions: list[NodeAction] = []
    for reclaim in expiry.reclaims + applied.reclaims:
        actions.append(
            NodeAction(
                op="release",
                lease_id=reclaim["lease_id"],
                node_id=reclaim["node_id"],
                lease_type=reclaim["lease_type"],
                expected_label=reclaim["label_before"],
                expected_response_label=reclaim["label_after"],
                trigger_seq=reclaim["event_seq"],
                reason=reclaim["reason"],
            )
        )
    outcome = applied.outcome or {}
    if outcome.get("status") in {"granted", "renewed", "released"}:
        op = {"granted": "grant", "renewed": "renew", "released": "release"}[outcome["status"]]
        actions.append(
            NodeAction(
                op=op,
                lease_id=outcome["lease_id"],
                node_id=outcome["node_id"],
                lease_type=outcome["lease_type"],
                expected_label=int(event.expected_label),
                expected_response_label=int(outcome["label"]),
                trigger_seq=event.seq,
                reason=outcome.get("reason", op),
            )
        )
    return actions




def _dispatch_actions(
    actions: list[NodeAction],
    *,
    event_seq: int,
    pending_id: str | None,
    cluster_by_id: dict[str, NodeSpec],
    client: JsonLineNodeClient,
    scheduler: QuotaScheduler,
) -> None:
    """Dispatch independent actions in parallel while serializing each lease chain."""

    chains: dict[str, list[int]] = {}
    for index, action in enumerate(actions):
        chains.setdefault(action.lease_id, []).append(index)
    max_rounds = max((len(chain) for chain in chains.values()), default=0)
    for round_index in range(max_rounds):
        ready = [
            index
            for index, action in enumerate(actions)
            if chains[action.lease_id][round_index] == index
        ]
        futures = []
        for index in sorted(ready):
            action = actions[index]
            node = cluster_by_id[action.node_id]
            request_id = f"e{event_seq}-a{index}-{pending_id or 'none'}"
            future, _task = scheduler.submit(
                action.node_id,
                request_id,
                lambda node=node, action=action, request_id=request_id: client.request(
                    node, action.request(request_id)
                ),
            )
            futures.append((index, future))
        for index, future in futures:
            action = actions[index]
            response = future.result()
            if response.get("ok") is not True:
                raise LeaseError(
                    "live node rejected a centrally accepted transition",
                    event_seq=event_seq,
                    lease_id=action.lease_id,
                    node_id=action.node_id,
                    node_code=response.get("code"),
                    node_state=response.get("state"),
                    node_label=response.get("label"),
                )
            if int(response.get("label", -1)) != action.expected_response_label:
                raise LeaseError(
                    "live node label diverged from central arbitration",
                    event_seq=event_seq,
                    lease_id=action.lease_id,
                    node_id=action.node_id,
                    expected_label=action.expected_response_label,
                    node_label=response.get("label"),
                )


def _iter_engine_events(
    spec: Spec,
    path: str,
    engine: LeaseArbiterEngine,
    *,
    start_ns: int,
    event_timing: list[dict[str, Any]],
) -> Iterator[tuple[Event, MutationBatch, MutationBatch, int]]:
    if path == "sequential":
        for event in spec.events:
            observed_ms = _wait_until_monotonic(start_ns, event.at_ms)
            event_timing.append({"seq": event.seq, "at_ms": event.at_ms, "observed_ms": observed_ms})
            expiry = engine.expire_due(event.at_ms, event.seq)
            applied = engine.apply_event(event)
            yield event, expiry, applied, 1
        return

    frames: list[list[Event]] = []
    for event in spec.events:
        if not frames or frames[-1][0].at_ms != event.at_ms:
            frames.append([event])
        else:
            frames[-1].append(event)
    for frame in frames:
        observed_ms = _wait_until_monotonic(start_ns, frame[0].at_ms)
        groups: dict[tuple[str, str], list[Event]] = {}
        for event in frame:
            if event.op == "quota_invalidate":
                key = (event.node_id or "", event.lease_type or "")
            else:
                lease = spec.lease_by_id[event.lease_id or ""]
                key = (lease.node_id, lease.lease_type)
            groups.setdefault(key, []).append(event)
        merged = sorted(
            (event for group_events in groups.values() for event in group_events),
            key=lambda item: item.seq,
        )
        expiry = engine.expire_due(frame[0].at_ms, frame[0].seq)
        for index, event in enumerate(merged):
            event_timing.append({"seq": event.seq, "at_ms": event.at_ms, "observed_ms": observed_ms})
            applied = engine.apply_event(event)
            yield event, expiry if index == 0 else MutationBatch(), applied, len(groups) if index == 0 else 0


def run_live(
    spec: Spec,
    *,
    ledger_path: str | Path,
    requested_path: str = "auto",
    fresh: bool = False,
    crash_after_event: int | None = None,
) -> dict[str, Any]:
    selection = choose_path(
        requested_path=requested_path,
        node_count=len(spec.nodes),
        event_count=len(spec.events),
        remaining_capacity=spec.global_concurrency,
    )
    path = selection["selected"]
    initial_engine = LeaseArbiterEngine(spec)
    ledger = Ledger(ledger_path, spec, initial_engine.snapshot(), fresh=fresh)
    session = ledger.session_info()
    if not fresh:
        ledger.recover_pending()
        committed = ledger.read_commits()
        if committed:
            ledger.close()
            raise LedgerError(
                "ledger already contains committed events; use --fresh for a new live session or the replay command",
                committed_events=len(committed),
            )
    cluster = LiveNodeCluster(spec)
    scheduler = QuotaScheduler(
        spec.global_concurrency,
        {node.node_id: node.max_leases for node in spec.nodes},
        spec.worker_threads,
    )
    client = JsonLineNodeClient(spec.connect_timeout_ms, spec.read_timeout_ms)
    engine = LeaseArbiterEngine(spec)
    event_timing: list[dict[str, Any]] = []
    batch_group_total = 0
    scheduler_metrics: dict[str, Any] | None = None
    try:
        cluster.start()
        cluster_by_id = {node.node_id: node for node in spec.nodes}
        start_ns = time.monotonic_ns()
        for event, expiry, applied, group_count in _iter_engine_events(
            spec,
            path,
            engine,
            start_ns=start_ns,
            event_timing=event_timing,
        ):
            if path == "batch":
                batch_group_total += group_count
            outcome = applied.outcome
            if outcome is None:
                raise LeaseError("event did not produce an outcome", event_seq=event.seq)
            actions = _actions_for_batches(expiry, applied, event)
            reservations: list[dict[str, Any]] = []
            if outcome.get("status") == "granted":
                reservations.append(
                    {
                        "lease_id": outcome["lease_id"],
                        "node_id": outcome["node_id"],
                        "global_slot": outcome["global_slot"],
                        "node_slot": outcome["node_slot"],
                    }
                )
            pending_id: str | None = None
            if actions:
                pending_id = ledger.record_pending(
                    event.seq,
                    {
                        "event_seq": event.seq,
                        "op": event.op,
                        "actions": [action.to_dict() for action in actions],
                        "reservations": reservations,
                    },
                )
            if crash_after_event is not None and event.seq == crash_after_event:
                # Deliberate abrupt termination: no finally blocks, no exit report.
                os._exit(137)
            _dispatch_actions(
                actions,
                event_seq=event.seq,
                pending_id=pending_id,
                cluster_by_id=cluster_by_id,
                client=client,
                scheduler=scheduler,
            )
            transitions, ownership = _partition_result(engine, event.seq)
            commit = ledger.commit_event(
                event_seq=event.seq,
                outcome=outcome,
                reclaims=expiry.reclaims + applied.reclaims,
                transitions=transitions,
                ownership_changes=ownership,
                snapshot=engine.snapshot(),
                pending_request_id=pending_id,
            )
            event_timing[-1]["snapshot_version"] = commit["snapshot_version"]
            event_timing[-1]["commit_digest"] = commit["commit_digest"]
            event_timing[-1]["action_count"] = len(actions)
        scheduler_metrics = scheduler.snapshot()
        result = engine.result(
            path,
            selection,
            (
                {
                    "frame_count": sum(
                        1 for index, item in enumerate(spec.events)
                        if index == 0 or item.at_ms != spec.events[index - 1].at_ms
                    ),
                    "group_count": batch_group_total,
                    "aggregation_keys": "node_id+lease_type",
                    "backfill_order": "event seq",
                }
                if path == "batch"
                else None
            ),
        )
        result["runtime"] = {
            "live": True,
            "format_version": 1,
            "scheduler": scheduler_metrics,
            "event_timing": event_timing,
            "ledger": {
                "path": str(Path(ledger_path)),
                "run_id": session["run_id"],
                "snapshot_replaced": ledger.snapshot_replaced,
                "replacement_reason": ledger.replacement_reason,
                "snapshot_version": ledger.session_info()["snapshot_version"],
                "snapshot_sha256": ledger.session_info()["content_digest"],
            },
        }
        result["live_projection_sha256"] = digest(result["comparison_projection"])
        return result
    finally:
        scheduler.shutdown(wait=True)
        cluster.stop()
        ledger.close()
