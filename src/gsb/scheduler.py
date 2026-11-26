"""Worker-pool scheduling with real monotonic leases and three capacity gates."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import hashlib
import threading
import time
from typing import Any, Callable, Iterable

from .errors import LeaseStateError, ScheduleError
from .fixed import analyze_fixed_error, water_level
from .ledger import LEDGER_FORMAT_VERSION
from .model import ShardSpec, Spec
from .samples import canonical_json_bytes, sha256_hex


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    units: int
    issued_ns: int
    deadline_ns: int
    renewed_ns: int
    status: str = "active"
    accounted: bool = True
    renewal_count: int = 0
    release_reason: str | None = None


class LeaseHandle:
    def __init__(self, manager: "LeaseManager", lease_id: str) -> None:
        self.manager = manager
        self.lease_id = lease_id

    def renew(self) -> None:
        self.manager.renew(self.lease_id)


class LeaseManager:
    """All capacity transitions occur exactly once under one lock."""

    def __init__(self, spec: Spec, *, session_start_ns: int | None = None) -> None:
        self.spec = spec
        self.node_map = spec.node_map()
        self.session_start_ns = session_start_ns or time.monotonic_ns()
        self._lock = threading.RLock()
        self._leases: dict[str, Lease] = {}
        self._global_units = 0
        self._node_units = {node.id: 0 for node in spec.nodes}
        self._node_lease_counts = {node.id: 0 for node in spec.nodes}
        self._node_shard_counts = {node.id: 0 for node in spec.nodes}
        self.events: list[dict[str, Any]] = []

    def _now_ms(self) -> int:
        return max(0, (time.monotonic_ns() - self.session_start_ns) // 1_000_000)

    def _event(self, lease: Lease, event: str, now_ns: int, **extra: Any) -> None:
        self.events.append(
            {
                "event": event,
                "lease_id": lease.lease_id,
                "shard_id": lease.shard_id,
                "node_id": lease.node_id,
                "units": lease.units,
                "at_ms": max(0, (now_ns - self.session_start_ns) // 1_000_000),
                **extra,
            }
        )

    def _hold(self, lease: Lease) -> None:
        self._global_units += lease.units
        self._node_units[lease.node_id] += lease.units
        self._node_lease_counts[lease.node_id] += 1
        self._node_shard_counts[lease.node_id] += 1

    def _drop_accounting(self, lease: Lease) -> None:
        if not lease.accounted:
            return
        self._global_units -= lease.units
        self._node_units[lease.node_id] -= lease.units
        self._node_lease_counts[lease.node_id] -= 1
        self._node_shard_counts[lease.node_id] -= 1
        lease.accounted = False

    def issue(self, *, lease_id: str, shard: ShardSpec, ttl_ms: int, now_ns: int | None = None) -> Lease:
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            existing = self._leases.get(lease_id)
            if existing is not None and existing.status in {"active", "expired"}:
                raise LeaseStateError("lease id is already active", details={"lease_id": lease_id})
            lease = Lease(
                lease_id=lease_id,
                shard_id=shard.id,
                node_id=shard.node_id,
                units=shard.demand,
                issued_ns=now,
                deadline_ns=now + ttl_ms * 1_000_000,
                renewed_ns=now,
            )
            self._leases[lease_id] = lease
            self._hold(lease)
            self._event(lease, "issued", now, deadline_ms=self._now_ms() + ttl_ms)
            return lease

    def renew(self, lease_id: str, *, now_ns: int | None = None) -> Lease:
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.status != "active":
                raise LeaseStateError(
                    "cannot renew a non-active lease",
                    details={"lease_id": lease_id, "status": None if lease is None else lease.status},
                )
            if now > lease.deadline_ns:
                lease.status = "expired"
                lease.release_reason = "expired_before_renew"
                self._event(lease, "expired", now, during="renew")
                raise LeaseStateError(
                    "lease expired before renewal",
                    details={"lease_id": lease_id, "now_ns": now, "deadline_ns": lease.deadline_ns},
                )
            lease.deadline_ns = now + self.spec.lease_ttl_ms * 1_000_000
            lease.renewed_ns = now
            lease.renewal_count += 1
            self._event(lease, "renewed", now, deadline_ms=self._now_ms() + self.spec.lease_ttl_ms)
            return lease

    def release(self, lease_id: str, *, reason: str, now_ns: int | None = None) -> bool:
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.status in {"released", "cancelled"}:
                return False
            was_accounted = lease.accounted
            self._drop_accounting(lease)
            lease.status = "released"
            lease.release_reason = reason
            self._event(lease, "released", now, reason=reason, capacity_released=was_accounted)
            return was_accounted

    def reclaim_expired(
        self,
        *,
        now_ns: int | None = None,
        protected_lease_ids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        now = time.monotonic_ns() if now_ns is None else now_ns
        protected = set(protected_lease_ids)
        reclaimed: list[dict[str, Any]] = []
        with self._lock:
            for lease in list(self._leases.values()):
                if lease.status != "active" or now <= lease.deadline_ns:
                    continue
                capacity_released = False
                if lease.lease_id not in protected:
                    capacity_released = lease.accounted
                    self._drop_accounting(lease)
                lease.status = "expired"
                lease.release_reason = "monotonic_deadline_exceeded"
                self._event(
                    lease,
                    "expired",
                    now,
                    capacity_released=capacity_released,
                    quarantine=lease.accounted,
                )
                reclaimed.append(
                    {
                        "lease_id": lease.lease_id,
                        "shard_id": lease.shard_id,
                        "node_id": lease.node_id,
                        "units": lease.units,
                        "capacity_released": capacity_released,
                        "quarantined_until_worker_exit": lease.accounted,
                    }
                )
        return reclaimed

    def reclaim_for_quota_change(
        self,
        node_id: str,
        *,
        now_ns: int | None = None,
    ) -> list[dict[str, Any]]:
        now = time.monotonic_ns() if now_ns is None else now_ns
        reclaimed: list[dict[str, Any]] = []
        with self._lock:
            for lease in list(self._leases.values()):
                if lease.node_id != node_id or lease.status not in {"active", "expired"}:
                    continue
                capacity_released = lease.accounted
                self._drop_accounting(lease)
                lease.status = "cancelled"
                lease.release_reason = "quota_change_reclaim"
                self._event(lease, "reclaimed_for_quota_change", now, capacity_released=capacity_released)
                reclaimed.append(
                    {
                        "lease_id": lease.lease_id,
                        "units": lease.units,
                        "capacity_released": capacity_released,
                    }
                )
        return reclaimed

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "global_units": self._global_units,
                "node_units": dict(self._node_units),
                "node_lease_counts": dict(self._node_lease_counts),
                "node_shard_counts": dict(self._node_shard_counts),
            }

    def lease(self, lease_id: str) -> Lease:
        with self._lock:
            return self._leases[lease_id]


@dataclass
class RunResult:
    records: dict[str, dict[str, Any]]
    waits: list[dict[str, Any]]
    lease_events: list[dict[str, Any]]
    session_version: int
    lease_stats: dict[str, Any] = field(default_factory=dict)

    def ordered_records(self, spec: Spec) -> list[dict[str, Any]]:
        return [self.records[shard_id] for shard_id in spec.topological_order() if shard_id in self.records]


class Scheduler:
    def __init__(
        self,
        spec: Spec,
        *,
        task: Callable[[ShardSpec, LeaseHandle], dict[str, Any]],
        session_version: int = 1,
        ledger: Any | None = None,
    ) -> None:
        self.spec = spec
        self.task = task
        self.session_version = session_version
        self.ledger = ledger
        self.node_map = spec.node_map()
        self.shard_map = spec.shard_map()
        self.topo_rank = spec.topo_rank()
        self.leases = LeaseManager(spec)

    def _lease_id(self, shard: ShardSpec) -> str:
        material = f"{self.spec.capacity_version}|{shard.node_id}|{shard.id}".encode("utf-8")
        return "L-" + hashlib.sha256(material).hexdigest()[:24]

    def _block_reasons(self, shard: ShardSpec, active_futures: int) -> list[str]:
        node = self.node_map[shard.node_id]
        stats = self.leases.stats()
        reasons: list[str] = []
        if active_futures >= self.spec.max_workers:
            reasons.append("worker_pool")
        if stats["global_units"] + shard.demand > self.spec.global_budget:
            reasons.append("global_budget")
        if stats["node_units"][node.id] + shard.demand > node.quota:
            reasons.append("node_water_quota")
        if stats["node_lease_counts"][node.id] >= node.lease_limit:
            reasons.append("node_lease_limit")
        if stats["node_shard_counts"][node.id] >= node.concurrency:
            reasons.append("node_concurrency")
        return reasons

    def _infeasible_reason(self, shard: ShardSpec) -> str | None:
        node = self.node_map[shard.node_id]
        if shard.demand > self.spec.global_budget:
            return "global_budget"
        if shard.demand > node.quota:
            return "node_water_quota"
        if node.lease_limit < 1:
            return "node_lease_limit"
        if node.concurrency < 1:
            return "node_concurrency"
        return None

    def _record(
        self,
        shard: ShardSpec,
        lease: Lease,
        aggregate: dict[str, Any],
    ) -> dict[str, Any]:
        node = self.node_map[shard.node_id]
        water = water_level(
            node.id,
            node.used + shard.demand,
            node.quota,
            self.spec.fixed_scale,
            lease_limit_units=node.lease_limit,
        )
        error_model = analyze_fixed_error(len(self.spec.nodes), self.spec.fixed_scale)
        record: dict[str, Any] = {
            "format_version": LEDGER_FORMAT_VERSION,
            "session_version": self.session_version,
            "shard_id": shard.id,
            "node_id": shard.node_id,
            "capacity_version": self.spec.capacity_version,
            "aggregate_digest": aggregate["aggregate_digest"],
            "sample_digest": aggregate["sample_digest"],
            "sample_row_count": shard.sample_count,
            "participating_rows": aggregate["participating_rows"],
            "fixed_scale": node.fixed_scale,
            "session_fixed_scale": self.spec.fixed_scale,
            "group_count": shard.group_count,
            "resource_units": shard.demand,
            "lease": {
                "lease_id": lease.lease_id,
                "node_id": lease.node_id,
                "shard_id": lease.shard_id,
                "units": lease.units,
                "status": "released",
                "reason": "completed",
            },
            "water": water.as_dict(),
            "error_attribution": {
                "segments_fp": error_model.as_dict()["components_fp"],
                "per_shard_quantization_fp": error_model.per_shard_quantization_fp,
                "per_node_comparison_bound_fp": error_model.per_node_comparison_bound_fp,
                "global_aggregation_bound_fp": error_model.global_aggregation_bound_fp,
                "total_bound_fp": error_model.total_bound_fp,
                "dominant_segment": error_model.dominant_segment,
                "tie_break_rule": error_model.tie_break_rule,
                "decision_source": "integer_fixed_point_no_float",
            },
            "progress_cursor": self.topo_rank[shard.id] + 1,
            "node_elapsed_ms": int(aggregate.get("node_elapsed_ms", 0)),
            "network_elapsed_ms": int(aggregate.get("network_elapsed_ms", 0)),
            "status": "committed",
        }
        record["content_digest"] = sha256_hex(canonical_json_bytes(record))
        return record

    def run(
        self,
        *,
        selected_shards: Iterable[str] | None = None,
        prior_done: Iterable[str] = (),
        prior_records: dict[str, dict[str, Any]] | None = None,
    ) -> RunResult:
        selected = set(self.shard_map if selected_shards is None else selected_shards)
        unknown = selected - set(self.shard_map)
        if unknown:
            raise ScheduleError("scheduler received unknown shards", details={"shards": sorted(unknown)})
        prior_done_set = set(prior_done)
        records = dict(prior_records or {})
        completed = set(prior_done_set)
        pending = set(selected)
        waiting: dict[str, dict[str, Any]] = {}
        wait_events: list[dict[str, Any]] = []
        running: dict[Future[dict[str, Any]], tuple[ShardSpec, Lease]] = {}
        first_failure: BaseException | None = None
        session_start_ns = time.monotonic_ns()

        def now_ms() -> int:
            return max(0, (time.monotonic_ns() - session_start_ns) // 1_000_000)

        def start_wait(shard_id: str, reasons: list[str]) -> None:
            current = waiting.get(shard_id)
            if current is None:
                waiting[shard_id] = {
                    "shard_id": shard_id,
                    "start_ms": now_ms(),
                    "reasons": set(reasons),
                }
            else:
                current["reasons"].update(reasons)

        def end_wait(shard_id: str) -> None:
            current = waiting.pop(shard_id, None)
            if current is None:
                return
            wait_events.append(
                {
                    "shard_id": shard_id,
                    "start_ms": current["start_ms"],
                    "end_ms": now_ms(),
                    "duration_ms": max(0, now_ms() - current["start_ms"]),
                    "reasons": sorted(current["reasons"]),
                }
            )

        with ThreadPoolExecutor(max_workers=self.spec.max_workers, thread_name_prefix="gsb-shard") as executor:
            while pending or running:
                if first_failure is not None and not running:
                    raise first_failure
                self.leases.reclaim_expired(protected_lease_ids=[lease.lease_id for _, lease in running.values()])
                ready = [
                    self.shard_map[shard_id]
                    for shard_id in pending
                    if set(self.shard_map[shard_id].dependencies) <= completed
                ]
                ready.sort(key=lambda shard: (self.topo_rank[shard.id], shard.id))

                for shard in ready:
                    if first_failure is not None:
                        break
                    reasons = self._block_reasons(shard, len(running))
                    if reasons:
                        start_wait(shard.id, reasons)
                        continue
                    end_wait(shard.id)
                    lease_id = self._lease_id(shard)
                    lease = self.leases.issue(
                        lease_id=lease_id,
                        shard=shard,
                        ttl_ms=self.spec.lease_ttl_ms,
                    )
                    if self.ledger is not None:
                        try:
                            self.ledger.reserve_lease(
                                lease_id=lease.lease_id,
                                shard_id=shard.id,
                                node_id=shard.node_id,
                                units=shard.demand,
                                issued_ns=lease.issued_ns,
                                deadline_ns=lease.deadline_ns,
                            )
                        except Exception:
                            self.leases.release(lease.lease_id, reason="ledger_reservation_failed")
                            raise
                    handle = LeaseHandle(self.leases, lease.lease_id)
                    future = executor.submit(self.task, shard, handle)
                    running[future] = (shard, lease)
                    pending.remove(shard.id)

                if not running:
                    if first_failure is not None:
                        raise first_failure
                    if pending:
                        # No task can ever become runnable without an external
                        # event.  Report an explicit infeasible capacity gate.
                        for shard_id in sorted(pending):
                            shard = self.shard_map[shard_id]
                            infeasible = self._infeasible_reason(shard)
                            if infeasible is not None:
                                raise ScheduleError(
                                    "shard is infeasible under the declared capacities",
                                    details={"shard_id": shard_id, "reason": infeasible},
                                )
                        dependency_blocked = [
                            shard_id
                            for shard_id in sorted(pending)
                            if not set(self.shard_map[shard_id].dependencies) <= completed
                        ]
                        raise ScheduleError(
                            "scheduler deadlock: no running task can release capacity",
                            details={"pending": sorted(pending), "dependency_blocked": dependency_blocked},
                        )
                    break

                finished, _ = wait(
                    running,
                    timeout=max(0.001, self.spec.poll_interval_ms / 1000),
                    return_when=FIRST_COMPLETED,
                )
                for future in finished:
                    shard, lease = running.pop(future)
                    try:
                        aggregate = future.result()
                        lease_now = self.leases.lease(lease.lease_id)
                        if lease_now.status in {"expired", "cancelled", "released"}:
                            raise LeaseStateError(
                                "task returned after its lease stopped being active",
                                details={"lease_id": lease.lease_id, "status": lease_now.status},
                            )
                        record = self._record(shard, lease, aggregate)
                        if self.ledger is not None:
                            self.ledger.complete_shard(record, lease_id=lease.lease_id)
                        self.leases.release(lease.lease_id, reason="completed")
                        records[shard.id] = record
                        completed.add(shard.id)
                    except BaseException as exc:
                        self.leases.release(lease.lease_id, reason="task_failure")
                        if self.ledger is not None:
                            try:
                                self.ledger.release_lease(lease.lease_id, reason="task_failure")
                            except Exception as ledger_exc:
                                if first_failure is None:
                                    first_failure = ledger_exc
                        if first_failure is None:
                            first_failure = exc

                if first_failure is not None and not running:
                    raise first_failure

        for shard_id in sorted(pending):
            start_wait(shard_id, ["dependency_failed"])
        for shard_id in list(waiting):
            end_wait(shard_id)
        ordered = {
            shard_id: records[shard_id]
            for shard_id in self.spec.topological_order()
            if shard_id in records
        }
        wait_events.sort(key=lambda event: (event["start_ms"], event["shard_id"]))
        return RunResult(
            records=ordered,
            waits=wait_events,
            lease_events=list(self.leases.events),
            session_version=self.session_version,
            lease_stats=self.leases.stats(),
        )
