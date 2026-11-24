from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json
from .client import NodeClient, NodePool
from .content import generate_content
from .errors import CacheLeaseError, ProtocolError, ReplyError, ReplayError, SpecError
from .ledger import Ledger
from .node import mono_ms
from .scheduler import (
    DecisionSequencer,
    NoEligibleRoute,
    QueuePermit,
    RequestScheduler,
    SchedulerAborted,
    rank_candidates,
)
from .spec import Config, Event, EventBatch, group_batches


RETRYABLE_NODE_ERRORS = {"E_CAPACITY", "E_LEASE_LIMIT", "E_QUOTA_VERSION", "E_NOT_FOUND"}
FATAL_NODE_ERRORS = {"E_INTERNAL", "E_PROTOCOL", "E_REPLY", "E_HANDSHAKE"}


class Engine:
    def __init__(
        self,
        config: Config,
        events: tuple[Event, ...],
        ledger: Ledger,
        work_dir: str | Path,
    ) -> None:
        self.config = config
        self.events = events
        self.ledger = ledger
        self.work_dir = Path(work_dir)
        self.pool: NodePool | None = None
        self._lock = threading.RLock()
        self.last_recovery: dict[str, Any] = {}
        self.processed_events = 0
        self.batch_summaries: list[dict[str, Any]] = []
        self.scheduler_metrics: list[dict[str, Any]] = []

    def run(
        self,
        *,
        until_seq: int | None = None,
        resume: bool = False,
        crash_at_batch: int | None = None,
    ) -> dict[str, Any]:
        progress = self.ledger.progress()
        if not resume and int(progress["last_ordinal"]) > 0:
            raise SpecError(
                "ledger already has committed progress; use resume=true/--resume or a fresh ledger path",
                self.ledger.path,
            )
        self._validate_committed_prefix()
        recovery_info: dict[str, Any] = {}
        if resume:
            recovery_info = self._prepare_ledger_recovery()
        batches = self._batches_to_process(until_seq)
        self.pool = NodePool(self.config, self.work_dir)
        try:
            self.pool.start()
            if resume:
                recovery_info.update(self._restore_nodes_and_reconcile())
                self.last_recovery = recovery_info
            for batch in batches:
                self._execute_batch(batch, crash_at_batch=crash_at_batch)
            statuses = self.pool.refresh_all()
            report = self._build_report(statuses, recovery_info)
            return report
        finally:
            if self.pool is not None:
                self.pool.stop()
                self.pool = None

    def recover(self) -> dict[str, Any]:
        self._validate_committed_prefix()
        recovery_info = self._prepare_ledger_recovery()
        self.pool = NodePool(self.config, self.work_dir)
        try:
            self.pool.start()
            recovery_info.update(self._restore_nodes_and_reconcile())
            self.last_recovery = recovery_info
            statuses = self.pool.refresh_all()
            return self._build_report(statuses, recovery_info)
        finally:
            if self.pool is not None:
                self.pool.stop()
                self.pool = None

    def _validate_committed_prefix(self) -> None:
        committed = self.ledger.committed_events()
        for row in committed:
            ordinal = int(row["ordinal"])
            if ordinal > len(self.events):
                raise SpecError(
                    f"ledger has committed event ordinal {ordinal}, but the event stream has only {len(self.events)} events",
                    self.ledger.path,
                )
            current = self.events[ordinal - 1]
            if current.ordinal != ordinal or current.digest != row["event_digest"]:
                raise SpecError(
                    "committed ledger prefix no longer matches the event stream; rebuild from a checkpoint before this edit",
                    f"event ordinal {ordinal}",
                )

    def _prepare_ledger_recovery(self) -> dict[str, Any]:
        open_batches = self.ledger.mark_open_batches_unknown(
            "process terminated before the final batch transaction committed"
        )
        replay = self.ledger.replay_and_validate(set(self.config.node_by_id))
        return {
            "open_batches_marked_unknown": [
                {
                    "batch_id": int(row["batch_id"]),
                    "first_ordinal": int(row["first_ordinal"]),
                    "last_ordinal": int(row["last_ordinal"]),
                    "action": "rerun_from_committed_progress",
                }
                for row in open_batches
            ],
            "replay": replay,
        }

    def _restore_nodes_and_reconcile(self) -> dict[str, Any]:
        assert self.pool is not None
        progress = self.ledger.progress()
        last_ordinal = int(progress["last_ordinal"])
        last_seq = int(progress["last_seq"])
        snapshot = self.ledger.latest_snapshot()
        if last_ordinal > 0 and snapshot is None:
            raise ReplayError(last_seq, "committed progress has no state snapshot")
        if snapshot is not None:
            for spec in self.config.nodes:
                node_snapshot = snapshot.get("nodes", {}).get(spec.node_id)
                if not isinstance(node_snapshot, dict):
                    raise ReplayError(last_seq, f"state snapshot has no node state for {spec.node_id}")
                self.pool.get(spec.node_id).restore(node_snapshot)
        unknown: list[dict[str, Any]] = []
        for spec in self.config.nodes:
            response = self.pool.get(spec.node_id).reconcile(last_seq, last_ordinal)
            for item in response.get("unknown_reservations", []):
                unknown.append({"node_id": spec.node_id, **item})
            for item in response.get("unknown_leases", []):
                unknown.append({"node_id": spec.node_id, **item})
        if unknown:
            self.ledger.record_recovery_action(
                None,
                "UNKNOWN_NODE_RESERVATIONS_OR_LEASES",
                {"last_committed_seq": last_seq, "items": unknown},
            )
        statuses = self.pool.refresh_all()
        if snapshot is not None:
            for spec in self.config.nodes:
                node_snapshot = snapshot["nodes"][spec.node_id]
                status = statuses[spec.node_id]
                for field_name in (
                    "used_slots",
                    "free_slots",
                    "active_leases",
                    "lease_remaining",
                    "watermark_units",
                    "quantization_error_upper",
                ):
                    if int(node_snapshot["watermark"][field_name]) != int(status["watermark"][field_name]):
                        raise ReplayError(
                            last_seq,
                            f"restored node {spec.node_id} watermark field {field_name} differs from ledger snapshot",
                        )
        return {
            "restored_from_snapshot": snapshot is not None,
            "unknown_node_state": unknown,
            "node_statuses": statuses,
        }

    def _batches_to_process(self, until_seq: int | None) -> list[EventBatch]:
        progress = self.ledger.progress()
        last_ordinal = int(progress["last_ordinal"])
        result: list[EventBatch] = []
        for batch in group_batches(self.events):
            if batch.events[-1].ordinal <= last_ordinal:
                continue
            if batch.events[0].ordinal <= last_ordinal < batch.events[-1].ordinal:
                raise SpecError(
                    f"ledger progress {last_ordinal} cuts through batch {batch.batch_id}", self.ledger.path
                )
            if until_seq is not None:
                if batch.events[0].seq > until_seq:
                    break
                if batch.events[-1].seq > until_seq:
                    raise SpecError(
                        f"until_seq {until_seq} must fall on a batch boundary", f"batch {batch.batch_id}"
                    )
            result.append(batch)
        return result

    def _execute_batch(self, batch: EventBatch, *, crash_at_batch: int | None) -> None:
        assert self.pool is not None
        self.ledger.begin_batch(batch.batch_id, batch.events)
        if batch.all_requests:
            records = self._execute_parallel(batch)
        else:
            records = self._execute_single(batch.events[0])
        statuses = self.pool.refresh_all()
        records["statuses"] = statuses
        self._reconcile_expired_ledger_rows(records, statuses)
        if crash_at_batch is not None and batch.batch_id == crash_at_batch:
            # Deliberately skip the final transaction. The open batch marker remains durable.
            os._exit(137)
        snapshot = self._build_snapshot(
            statuses,
            int(batch.events[-1].ordinal),
            int(batch.events[-1].seq),
        )
        digest = self.ledger.commit_batch(
            batch.batch_id,
            events=records["events"],
            lease_upserts=records.get("lease_upserts", []),
            lease_updates=records.get("lease_updates", []),
            reservation_upserts=records.get("reservation_upserts", []),
            reservation_updates=records.get("reservation_updates", []),
            capacity_changes=records.get("capacity_changes", []),
            watermarks=records.get("watermarks", []),
            stale_watermarks=records.get("stale_watermarks", []),
            arbitrations=records.get("arbitrations", []),
            queue_observations=records.get("queue_observations", []),
            snapshot=snapshot,
        )
        self.processed_events += len(batch.events)
        self.batch_summaries.append(
            {
                "batch_id": batch.batch_id,
                "first_seq": int(batch.events[0].seq),
                "last_seq": int(batch.events[-1].seq),
                "events": len(batch.events),
                "ledger_digest": digest,
                "watermarks": {
                    node_id: status["watermark"] for node_id, status in sorted(statuses.items())
                },
            }
        )

    def _reconcile_expired_ledger_rows(
        self, records: dict[str, Any], statuses: dict[str, dict[str, Any]]
    ) -> None:
        touched_leases = {
            str(row["lease_id"])
            for row in records.get("lease_updates", []) + records.get("lease_upserts", [])
        }
        node_lease_ids = {
            node_id: {str(item["lease_id"]) for item in status.get("leases", [])}
            for node_id, status in statuses.items()
        }
        for lease in self.ledger.active_leases():
            lease_id = str(lease["lease_id"])
            if lease_id in touched_leases:
                continue
            if lease_id not in node_lease_ids.get(str(lease["node_id"]), set()):
                records.setdefault("lease_updates", []).append(
                    {
                        "lease_id": lease_id,
                        "status": "expired",
                        "stale": 0,
                        "disposition": "expired_by_monotonic_clock",
                        "closed_ms": mono_ms(),
                    }
                )
        touched_reservations = {
            str(row["reservation_id"])
            for row in records.get("reservation_updates", []) + records.get("reservation_upserts", [])
        }
        node_reservation_ids = {
            node_id: {str(item["reservation_id"]) for item in status.get("reservations", [])}
            for node_id, status in statuses.items()
        }
        for reservation in self.ledger.active_reservations():
            reservation_id = str(reservation["reservation_id"])
            if reservation_id in touched_reservations:
                continue
            if reservation_id not in node_reservation_ids.get(str(reservation["node_id"]), set()):
                records.setdefault("reservation_updates", []).append(
                    {
                        "reservation_id": reservation_id,
                        "status": "unknown",
                        "stale": 1,
                        "disposition": "missing_after_grace_or_recovery",
                        "closed_ms": mono_ms(),
                    }
                )

    def _execute_parallel(self, batch: EventBatch) -> dict[str, Any]:
        assert self.pool is not None
        scheduler = RequestScheduler(self.config)
        sequencer = DecisionSequencer()
        artifact_locks = {
            artifact.key: threading.Lock() for artifact in self.config.artifacts
        }
        results: list[dict[str, Any]] = []
        fatal: BaseException | None = None
        with ThreadPoolExecutor(max_workers=len(batch.events), thread_name_prefix="cachelease-request") as executor:
            futures: list[Future[dict[str, Any]]] = []
            for order, event in enumerate(batch.events):
                futures.append(
                    executor.submit(
                        self._request_task,
                        event,
                        order,
                        sequencer,
                        scheduler,
                        artifact_locks,
                    )
                )
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except (CacheLeaseError, SchedulerAborted) as exc:
                    if fatal is None:
                        fatal = exc
                    scheduler.abort(exc)
        self.scheduler_metrics.append(
            {
                "batch_id": batch.batch_id,
                "max_global_in_flight": scheduler.max_global_observed,
                "max_node_in_flight": dict(scheduler.max_node_observed),
                "max_stage_in_flight": dict(scheduler.max_stage_observed),
            }
        )
        if fatal is not None:
            if isinstance(fatal, SchedulerAborted):
                raise fatal.cause
            raise fatal
        results.sort(key=lambda item: item["event"].ordinal)
        statuses = self.pool.refresh_all()
        return self._records_from_parallel(results, statuses)

    def _request_task(
        self,
        event: Event,
        order: int,
        sequencer: DecisionSequencer,
        scheduler: RequestScheduler,
        artifact_locks: dict[str, threading.Lock],
    ) -> dict[str, Any]:
        assert self.pool is not None and event.artifact is not None
        artifact = self.config.artifact_by_key[event.artifact]
        assert event.lease_ms is not None
        lock = artifact_locks[event.artifact]
        permit: QueuePermit | None = None
        lock_held = False
        reservation: dict[str, Any] | None = None
        selected_node: str | None = None
        candidates: list[dict[str, Any]] = []
        node_elapsed_ms = 0
        watermark: dict[str, Any] | None = None
        quota_version: int | None = None
        outcome: dict[str, Any] = {
            "event": event,
            "status": "error",
            "operation": event.event_type,
            "node_id": None,
            "candidates": candidates,
            "queue": None,
            "reservation": None,
            "lease": None,
            "watermark": None,
            "quota_version": None,
            "elapsed_ms": 0,
        }
        try:
            with sequencer.turn(order):
                lock.acquire()
                lock_held = True
                tried: set[str] = set()
                while True:
                    selection_holder: dict[str, Any] = {}

                    def select(node_in_flight: dict[str, int]) -> tuple[str | None, list[dict[str, Any]]]:
                        snapshots = {
                            node.node_id: self.pool.snapshot(node.node_id) for node in self.config.nodes
                        }
                        chosen, ranked = rank_candidates(
                            event,
                            self.config,
                            snapshots,
                            node_in_flight,
                            excluded=tried,
                        )
                        selection_holder["candidates"] = ranked
                        selection_holder["chosen"] = chosen
                        return chosen, ranked

                    try:
                        permit = scheduler.acquire(artifact.stage, select)
                    except NoEligibleRoute as exc:
                        candidates.extend(
                            item for item in exc.candidates if item not in candidates
                        )
                        outcome.update(
                            {
                                "status": "error",
                                "error_code": _route_error_code(exc.candidates),
                                "error_message": "no eligible node can satisfy this request",
                            }
                        )
                        permit = None
                        break
                    candidates = selection_holder.get("candidates", candidates)
                    selected_node = permit.node_id
                    client = self.pool.get(selected_node)
                    try:
                        if event.event_type == "write":
                            response = client.reserve(
                                event.artifact,
                                event.lease_ms,
                                event.seq,
                                self.config.runtime.grace_ms,
                                int(client.status_snapshot.get("quota_version", -1)),
                                event.ordinal,
                            )
                            now = mono_ms()
                            reservation = {
                                "reservation_id": response["reservation_id"],
                                "node_id": selected_node,
                                "artifact_key": event.artifact,
                                "apply_seq": event.seq,
                                "ordinal": event.ordinal,
                                "quota_version": int(response["quota_version"]),
                                "status": "reserved",
                                "stale": 0,
                                "disposition": None,
                                "grace_deadline_ms": int(response["grace_deadline_ms"]),
                                "created_ms": now,
                                "closed_ms": None,
                            }
                        else:
                            response = client.read(
                                event.artifact, event.lease_ms, event.seq, event.ordinal
                            )
                        node_elapsed_ms = int(response.get("elapsed_ms", 0))
                        quota_version = int(response["quota_version"])
                        watermark = dict(response.get("watermark", {}))
                    except ProtocolError as exc:
                        scheduler.release(permit)
                        permit = None
                        if exc.code in RETRYABLE_NODE_ERRORS:
                            tried.add(selected_node)
                            try:
                                self.pool.refresh(selected_node)
                            except CacheLeaseError:
                                pass
                            continue
                        raise
                    try:
                        self.pool.refresh(selected_node)
                    except CacheLeaseError:
                        pass
                    break

            outcome.update(
                {
                    "node_id": selected_node,
                    "candidates": candidates,
                    "queue": _permit_dict(permit) if permit is not None else None,
                    "reservation": reservation,
                    "quota_version": quota_version,
                    "watermark": watermark,
                    "elapsed_ms": node_elapsed_ms,
                }
            )
            if permit is None:
                return outcome

            assert self.pool is not None and selected_node is not None
            client = self.pool.get(selected_node)
            if event.event_type == "write":
                assert reservation is not None
                if event.hold_ms > 0:
                    threading.Event().wait(event.hold_ms / 1000.0)
                try:
                    commit_response = client.commit(reservation["reservation_id"])
                except ProtocolError as exc:
                    reservation["status"] = "rolled_back"
                    reservation["disposition"] = (
                        "grace_expired_auto_rollback" if exc.code == "E_EXPIRED" else f"commit_failed:{exc.code}"
                    )
                    reservation["closed_ms"] = mono_ms()
                    if exc.code != "E_EXPIRED":
                        try:
                            client.rollback(reservation["reservation_id"], f"commit_failed:{exc.code}")
                        except CacheLeaseError:
                            reservation["status"] = "unknown"
                    outcome.update(
                        {
                            "status": "error",
                            "error_code": exc.code,
                            "error_message": exc.message,
                            "reservation": reservation,
                        }
                    )
                    try:
                        self.pool.refresh(selected_node)
                    except CacheLeaseError:
                        pass
                    return outcome
                expected = generate_content(
                    selected_node,
                    event.artifact,
                    self.config.node_by_id[selected_node].seed,
                    artifact.shards,
                    artifact.shard_bytes,
                ).digest
                if commit_response.get("digest") != expected:
                    raise ReplyError(
                        "committed content digest does not match the deterministic digest",
                        f"node[{selected_node}].digest",
                    )
                if int(commit_response["quota_version"]) != int(reservation["quota_version"]):
                    raise ReplyError(
                        "quota version changed between reserve and commit",
                        f"node[{selected_node}].quota_version",
                    )
                reservation.update(
                    {
                        "status": "committed",
                        "disposition": "committed",
                        "closed_ms": mono_ms(),
                    }
                )
                lease = {
                    "lease_id": commit_response["lease_id"],
                    "node_id": selected_node,
                    "artifact_key": event.artifact,
                    "apply_seq": event.seq,
                    "ordinal": event.ordinal,
                    "quota_version": int(commit_response["quota_version"]),
                    "digest": commit_response["digest"],
                    "status": "active",
                    "stale": 0,
                    "disposition": "write_grant",
                    "parent_lease_id": None,
                    "opened_ms": mono_ms(),
                    "closed_ms": None,
                }
                outcome.update(
                    {
                        "status": "ok",
                        "reservation": reservation,
                        "lease": lease,
                        "quota_version": int(commit_response["quota_version"]),
                        "watermark": dict(commit_response.get("watermark", {})),
                        "elapsed_ms": node_elapsed_ms + int(commit_response.get("elapsed_ms", 0)),
                    }
                )
                try:
                    self.pool.refresh(selected_node)
                except CacheLeaseError:
                    pass
                return outcome

            expected = generate_content(
                selected_node,
                event.artifact,
                self.config.node_by_id[selected_node].seed,
                artifact.shards,
                artifact.shard_bytes,
            ).digest
            if response.get("digest") != expected:
                raise ReplyError(
                    "read content digest does not match the deterministic digest",
                    f"node[{selected_node}].digest",
                )
            lease = {
                "lease_id": response["lease_id"],
                "node_id": selected_node,
                "artifact_key": event.artifact,
                "apply_seq": event.seq,
                "ordinal": event.ordinal,
                "quota_version": int(response["quota_version"]),
                "digest": response["digest"],
                "status": "active",
                "stale": 0,
                "disposition": "read_grant",
                "parent_lease_id": None,
                "opened_ms": mono_ms(),
                "closed_ms": None,
            }
            outcome.update({"status": "ok", "lease": lease})
            return outcome
        except SchedulerAborted:
            raise
        except CacheLeaseError as exc:
            if exc.code in FATAL_NODE_ERRORS or exc.code in {"E_HANDSHAKE", "E_REPLY", "E_DIVERGENCE"}:
                scheduler.abort(exc)
                raise
            outcome.update(
                {
                    "status": "error",
                    "error_code": exc.code,
                    "error_message": exc.message,
                }
            )
            return outcome
        finally:
            if permit is not None:
                scheduler.release(permit)
            if lock_held:
                lock.release()

    def _execute_single(self, event: Event) -> dict[str, Any]:
        assert self.pool is not None
        records: dict[str, Any] = {
            "events": [],
            "lease_upserts": [],
            "lease_updates": [],
            "reservation_upserts": [],
            "reservation_updates": [],
            "capacity_changes": [],
            "watermarks": [],
            "stale_watermarks": [],
            "arbitrations": [],
            "queue_observations": [],
        }
        if event.event_type == "release":
            self._execute_release(event, records)
        elif event.event_type == "quota":
            self._execute_quota(event, records)
        else:
            raise SpecError("non-request batch must contain a release or quota event", f"event {event.ordinal}")
        statuses = self.pool.refresh_all()
        result = records.pop("_result")
        records["watermarks"] = self._watermark_records([result], statuses)
        records["events"] = [self._event_record(result)]
        records["statuses"] = statuses
        return records

    def _execute_release(self, event: Event, records: dict[str, Any]) -> None:
        assert self.pool is not None
        outcome: dict[str, Any] = {
            "event": event,
            "status": "ok",
            "operation": "release",
            "node_id": None,
            "candidates": [],
            "queue": None,
            "reservation": None,
            "lease": None,
            "watermark": None,
            "quota_version": None,
            "elapsed_ms": 0,
            "released_lease_ids": [],
            "released_slots": 0,
        }
        targets: list[dict[str, Any]] = []
        if event.lease_id is not None:
            lease = self.ledger.lease(event.lease_id)
            if lease is None or lease["status"] != "active" or int(lease["stale"]) != 0:
                outcome.update(
                    {
                        "status": "error",
                        "error_code": "E_NOT_FOUND",
                        "error_message": "release lease_id is not active in the ledger",
                    }
                )
                records["_result"] = outcome
                return
            targets = [lease]
        else:
            assert event.artifact is not None
            targets = self.ledger.active_leases(artifact=event.artifact)
            if not targets:
                outcome.update(
                    {
                        "status": "error",
                        "error_code": "E_NOT_FOUND",
                        "error_message": "no active lease exists for the artifact",
                    }
                )
                records["_result"] = outcome
                return

        by_node: dict[str, list[dict[str, Any]]] = {}
        for lease in targets:
            by_node.setdefault(str(lease["node_id"]), []).append(lease)
        released_ids: set[str] = set()
        freed_slots = 0
        node_elapsed = 0
        errors: list[str] = []
        for node_id, leases in sorted(by_node.items()):
            client = self.pool.get(node_id)
            try:
                if event.lease_id is not None:
                    response = client.release(event.lease_id)
                    released_ids.update(response.get("released_lease_ids", [event.lease_id]))
                    if response.get("released"):
                        released_ids.add(event.lease_id)
                    freed_slots += 1 if response.get("freed_slot") else 0
                else:
                    response = client.release_artifact(str(event.artifact))
                    released_ids.update(str(item) for item in response.get("released_lease_ids", []))
                    freed_slots += 1 if response.get("freed_slot") else 0
                node_elapsed += int(response.get("elapsed_ms", 0))
                outcome["node_id"] = node_id
                outcome["quota_version"] = int(client.status_snapshot.get("quota_version", -1))
                outcome["watermark"] = dict(response.get("watermark", {}))
            except ProtocolError as exc:
                if exc.code in {"E_NOT_FOUND", "E_EXPIRED"}:
                    errors.append(f"{node_id}:{exc.code}")
                    continue
                raise

        for lease in targets:
            lease_id = str(lease["lease_id"])
            if lease_id in released_ids:
                records["lease_updates"].append(
                    {
                        "lease_id": lease_id,
                        "status": "released",
                        "stale": 0,
                        "disposition": "released_by_event",
                        "closed_ms": mono_ms(),
                    }
                )
                outcome["released_lease_ids"].append(lease_id)
            elif errors:
                records["lease_updates"].append(
                    {
                        "lease_id": lease_id,
                        "status": "expired",
                        "stale": 0,
                        "disposition": "not_active_on_node",
                        "closed_ms": mono_ms(),
                    }
                )
        if errors and not released_ids:
            outcome.update(
                {
                    "status": "error",
                    "error_code": "E_NOT_FOUND",
                    "error_message": "; ".join(errors),
                }
            )
        outcome["released_slots"] = freed_slots
        outcome["elapsed_ms"] = node_elapsed
        records["_result"] = outcome

    def _execute_quota(self, event: Event, records: dict[str, Any]) -> None:
        assert self.pool is not None and event.node_id is not None and event.quota_version is not None
        client = self.pool.get(event.node_id)
        pre_status = self.pool.refresh(event.node_id)
        old_version = int(pre_status["quota_version"])
        old_leases = self.ledger.active_leases(node_id=event.node_id)
        node_active_ids = {str(item["lease_id"]) for item in pre_status.get("leases", [])}
        expired_rows: list[dict[str, Any]] = []
        for lease in old_leases:
            if str(lease["lease_id"]) not in node_active_ids:
                expired_rows.append(
                    {
                        "lease_id": str(lease["lease_id"]),
                        "status": "expired",
                        "stale": 0,
                        "disposition": "expired_before_quota_change",
                        "closed_ms": mono_ms(),
                    }
                )
        records["lease_updates"].extend(expired_rows)
        old_active = [lease for lease in old_leases if str(lease["lease_id"]) in node_active_ids]
        old_reservations = [
            row for row in self.ledger.active_reservations() if str(row["node_id"]) == event.node_id
        ]
        response = client.set_quota(
            event.quota_version,
            event.capacity_slots,
            event.lease_limit,
            str(event.policy),
        )
        post_status = self.pool.refresh(event.node_id)
        post_lease_ids = {str(item["lease_id"]) for item in post_status.get("leases", [])}
        dispositions = response.get("lease_dispositions", [])
        by_old: dict[str, list[dict[str, Any]]] = {}
        by_new: dict[str, dict[str, Any]] = {}
        for disposition in dispositions:
            old_id = str(disposition["lease_id"])
            by_old.setdefault(old_id, []).append(disposition)
            if disposition.get("new_lease_id"):
                by_new[str(disposition["new_lease_id"])] = disposition

        for lease in old_active:
            old_id = str(lease["lease_id"])
            disposition = by_old.get(old_id, [])
            if disposition:
                first = disposition[0]
                status = str(first["status"])
                reason = str(first["reason"])
            else:
                status = "stale"
                reason = "quota_version_replaced"
            records["lease_updates"].append(
                {
                    "lease_id": old_id,
                    "status": status,
                    "stale": 1,
                    "disposition": reason,
                    "closed_ms": mono_ms(),
                }
            )

        for lease_id, item in sorted(by_new.items()):
            parent_id = str(item["lease_id"])
            old_lease = next((lease for lease in old_active if str(lease["lease_id"]) == parent_id), None)
            if old_lease is None:
                continue
            records["lease_upserts"].append(
                {
                    "lease_id": lease_id,
                    "node_id": event.node_id,
                    "artifact_key": str(old_lease["artifact_key"]),
                    "apply_seq": int(old_lease["apply_seq"]),
                    "ordinal": event.ordinal,
                    "quota_version": event.quota_version,
                    "digest": str(old_lease["digest"]),
                    "status": "active" if lease_id in post_lease_ids else "revoked",
                    "stale": 0 if lease_id in post_lease_ids else 1,
                    "disposition": "renewed_by_quota_version" if lease_id in post_lease_ids else item.get("reason"),
                    "parent_lease_id": parent_id,
                    "opened_ms": mono_ms(),
                    "closed_ms": None if lease_id in post_lease_ids else mono_ms(),
                }
            )

        for lease in post_status.get("leases", []):
            lease_id = str(lease["lease_id"])
            if lease_id in {str(item["lease_id"]) for item in old_active}:
                continue
            if lease_id in by_new:
                continue
            records["lease_upserts"].append(
                {
                    "lease_id": lease_id,
                    "node_id": event.node_id,
                    "artifact_key": str(lease["artifact_key"]),
                    "apply_seq": int(lease["apply_seq"]),
                    "ordinal": event.ordinal,
                    "quota_version": int(lease["quota_version"]),
                    "digest": str(lease["digest"]),
                    "status": "active",
                    "stale": 0,
                    "disposition": "present_after_quota_change",
                    "parent_lease_id": None,
                    "opened_ms": mono_ms(),
                    "closed_ms": None,
                }
            )

        reservation_dispositions = {
            str(item["reservation_id"]): item for item in response.get("reservation_dispositions", [])
        }
        for reservation in old_reservations:
            reservation_id = str(reservation["reservation_id"])
            item = reservation_dispositions.get(reservation_id)
            records["reservation_updates"].append(
                {
                    "reservation_id": reservation_id,
                    "status": "rolled_back",
                    "stale": 1,
                    "disposition": str(item.get("reason")) if item else "quota_version_replaced",
                    "closed_ms": mono_ms(),
                }
            )

        records["capacity_changes"].append(
            {
                "ordinal": event.ordinal,
                "node_id": event.node_id,
                "old_quota_version": int(response["old_quota_version"]),
                "new_quota_version": int(response["quota_version"]),
                "old_capacity_slots": int(response["old_capacity_slots"]),
                "new_capacity_slots": int(response["capacity_slots"]),
                "old_lease_limit": int(response["old_lease_limit"]),
                "new_lease_limit": int(response["lease_limit"]),
                "policy": str(response["policy"]),
                "detail_json": canonical_json(response),
            }
        )
        records["stale_watermarks"].append(
            {"node_id": event.node_id, "quota_version": old_version}
        )
        records["_result"] = {
            "event": event,
            "status": "ok",
            "operation": "quota",
            "node_id": event.node_id,
            "candidates": [],
            "queue": None,
            "reservation": None,
            "lease": None,
            "watermark": dict(response.get("watermark", {})),
            "quota_version": int(response["quota_version"]),
            "elapsed_ms": int(response.get("elapsed_ms", 0)),
            "old_quota_version": int(response["old_quota_version"]),
            "capacity_slots": int(response["capacity_slots"]),
            "lease_limit": int(response["lease_limit"]),
            "policy": str(response["policy"]),
            "lease_dispositions": dispositions,
            "reservation_dispositions": response.get("reservation_dispositions", []),
        }

    def _records_from_parallel(
        self, results: list[dict[str, Any]], statuses: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        records: dict[str, Any] = {
            "events": [],
            "lease_upserts": [],
            "lease_updates": [],
            "reservation_upserts": [],
            "reservation_updates": [],
            "capacity_changes": [],
            "watermarks": [],
            "stale_watermarks": [],
            "arbitrations": [],
            "queue_observations": [],
        }
        for result in results:
            event = result["event"]
            records["events"].append(self._event_record(result))
            if result.get("lease") is not None:
                records["lease_upserts"].append(result["lease"])
            if result.get("reservation") is not None:
                records["reservation_upserts"].append(result["reservation"])
            if result.get("queue") is not None:
                queue = result["queue"]
                records["queue_observations"].append(
                    {
                        "ordinal": event.ordinal,
                        "seq": event.seq,
                        "node_id": queue["node_id"],
                        "artifact_key": str(event.artifact),
                        "stage": queue["stage"],
                        "wait_start_ms": int(queue["wait_start_ms"]),
                        "wait_end_ms": int(queue["wait_end_ms"]),
                        "wait_ms": int(queue["wait_ms"]),
                        "global_in_flight": int(queue["global_in_flight"]),
                        "node_in_flight": int(queue["node_in_flight"]),
                        "stage_in_flight": int(queue["stage_in_flight"]),
                    }
                )
            if result.get("candidates"):
                records["arbitrations"].append(
                    {
                        "ordinal": event.ordinal,
                        "artifact_key": str(event.artifact),
                        "chosen_node": result.get("node_id"),
                        "rule": "digest_match_then_free_slots_desc_then_node_id",
                        "candidates": result["candidates"],
                    }
                )
        records["watermarks"] = self._watermark_records(results, statuses)
        return records

    def _event_record(self, result: dict[str, Any]) -> dict[str, Any]:
        event: Event = result["event"]
        outcome = {
            key: value
            for key, value in result.items()
            if key != "event"
        }
        return {
            "ordinal": event.ordinal,
            "seq": event.seq,
            "event_type": event.event_type,
            "event_digest": event.digest,
            "event": event.canonical_payload(),
            "outcome": outcome,
            "status": result["status"],
            "error_code": result.get("error_code"),
        }

    def _watermark_records(
        self, results: list[dict[str, Any]], statuses: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in results:
            event: Event = result["event"]
            for spec in self.config.nodes:
                node_id = spec.node_id
                status = statuses[node_id]
                # A request batch is a ledger/watermark barrier. Every event row in the
                # batch therefore observes the committed post-batch watermark. The
                # operation itself still returns its immediate node-local watermark.
                watermark = status["watermark"]
                quota_version = int(status["quota_version"])
                elapsed_ms = int(result.get("elapsed_ms", 0)) if result.get("node_id") == node_id else 0
                rows.append(
                    {
                        "ordinal": event.ordinal,
                        "seq": event.seq,
                        "node_id": node_id,
                        "quota_version": quota_version,
                        "used_slots": int(watermark["used_slots"]),
                        "free_slots": int(watermark["free_slots"]),
                        "active_leases": int(watermark["active_leases"]),
                        "lease_remaining": int(watermark["lease_remaining"]),
                        "watermark_units": int(watermark["watermark_units"]),
                        "quantization_error_upper": int(watermark["quantization_error_upper"]),
                        "stale": 0,
                        "elapsed_ms": max(0, elapsed_ms),
                    }
                )
        return rows

    def _build_snapshot(
        self,
        statuses: dict[str, dict[str, Any]],
        last_ordinal: int,
        last_seq: int,
    ) -> dict[str, Any]:
        nodes: dict[str, Any] = {}
        artifact_state: dict[str, Any] = {
            artifact.key: {"active_leases": [], "nodes": []} for artifact in self.config.artifacts
        }
        for spec in self.config.nodes:
            status = statuses[spec.node_id]
            entries = sorted(
                (
                    {
                        "artifact_key": str(item["artifact_key"]),
                        "digest": str(item["digest"]),
                        "quota_version": int(item["quota_version"]),
                        "lease_count": int(item["lease_count"]),
                        "shards": int(item["shards"]),
                        "shard_bytes": int(item["shard_bytes"]),
                    }
                    for item in status.get("entries", [])
                ),
                key=lambda item: item["artifact_key"],
            )
            leases = sorted(
                (
                    {
                        "lease_id": str(item["lease_id"]),
                        "artifact_key": str(item["artifact_key"]),
                        "digest": str(item["digest"]),
                        "quota_version": int(item["quota_version"]),
                        "apply_seq": int(item["apply_seq"]),
                        "apply_ordinal": int(item.get("apply_ordinal", item["apply_seq"])),
                        "remaining_ms": int(item["remaining_ms"]),
                    }
                    for item in status.get("leases", [])
                ),
                key=lambda item: item["lease_id"],
            )
            reservations = sorted(
                (
                    {
                        "reservation_id": str(item["reservation_id"]),
                        "artifact_key": str(item["artifact_key"]),
                        "apply_seq": int(item["apply_seq"]),
                        "quota_version": int(item["quota_version"]),
                        "remaining_grace_ms": int(item["remaining_grace_ms"]),
                    }
                    for item in status.get("reservations", [])
                ),
                key=lambda item: item["reservation_id"],
            )
            nodes[spec.node_id] = {
                "node_id": spec.node_id,
                "quota_version": int(status["quota_version"]),
                "capacity_slots": int(status["capacity_slots"]),
                "lease_limit": int(status["lease_limit"]),
                "watermark": dict(status["watermark"]),
                "entries": entries,
                "leases": leases,
                "reservations": reservations,
            }
            for lease in leases:
                artifact_item = artifact_state.setdefault(
                    lease["artifact_key"], {"active_leases": [], "nodes": []}
                )
                artifact_item["active_leases"].append(
                    {
                        "lease_id": lease["lease_id"],
                        "node_id": spec.node_id,
                        "digest": lease["digest"],
                        "quota_version": lease["quota_version"],
                    }
                )
                if spec.node_id not in artifact_item["nodes"]:
                    artifact_item["nodes"].append(spec.node_id)
        for item in artifact_state.values():
            item["active_leases"].sort(key=lambda lease: (lease["node_id"], lease["lease_id"]))
            item["nodes"].sort()
        return {
            "format_version": 1,
            "config_digest": self.config.config_digest,
            "last_ordinal": last_ordinal,
            "last_seq": last_seq,
            "nodes": nodes,
            "artifacts": {key: artifact_state[key] for key in sorted(artifact_state)},
        }

    def _build_report(
        self,
        statuses: dict[str, dict[str, Any]],
        recovery_info: dict[str, Any],
    ) -> dict[str, Any]:
        progress = self.ledger.progress()
        latest_watermarks = self.ledger.latest_watermarks()
        return {
            "ok": True,
            "config_digest": self.config.config_digest,
            "ledger": self.ledger.path,
            "format_version": int(self.ledger.meta_int("format_version")),
            "content_digest": self.ledger.meta("content_digest"),
            "progress": {
                "last_ordinal": int(progress["last_ordinal"]),
                "last_seq": int(progress["last_seq"]),
                "last_batch": int(progress["last_batch"]),
                "ledger_digest": progress["ledger_digest"],
            },
            "events_processed_this_run": self.processed_events,
            "batches": self.batch_summaries,
            "scheduler_metrics": self.scheduler_metrics,
            "final_watermarks": {row["node_id"]: row for row in latest_watermarks},
            "node_statuses": {
                node_id: {
                    "quota_version": int(status["quota_version"]),
                    "watermark": status["watermark"],
                    "counters": status.get("counters", {}),
                }
                for node_id, status in sorted(statuses.items())
            },
            "recovery": recovery_info,
            "recovery_actions": self.ledger.recovery_actions(),
            "replay": self.ledger.replay_and_validate(set(self.config.node_by_id)),
        }


def _route_error_code(candidates: Iterable[dict[str, Any]]) -> str:
    reasons = {str(item.get("reason")) for item in candidates}
    if "lease_limit" in reasons:
        return "E_LEASE_LIMIT"
    if "no_free_slot" in reasons:
        return "E_CAPACITY"
    if "digest_missing" in reasons:
        return "E_NOT_FOUND"
    return "E_NO_NODE"


def _permit_dict(permit: QueuePermit) -> dict[str, Any]:
    return {
        "node_id": permit.node_id,
        "artifact_key": permit.artifact_key,
        "stage": permit.stage,
        "wait_start_ms": permit.wait_start_ms,
        "wait_end_ms": permit.wait_end_ms,
        "wait_ms": permit.wait_ms,
        "global_in_flight": permit.global_in_flight,
        "node_in_flight": permit.node_in_flight,
        "stage_in_flight": permit.stage_in_flight,
    }
