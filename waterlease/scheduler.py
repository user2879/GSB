from __future__ import annotations

import os
import signal
import time
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from .computation import compute_shard_remotely
from .errors import CapacityError, WaterLeaseError
from .hashing import sha256_hex
from .leases import LeaseManager
from .ledger import Ledger
from .process_manager import ProcessManager
from .spec import SpecBundle
from .timeutil import monotonic_ns
from .topology import topological_order
from .water import FixedPointMeter, node_water_error_bounds, summarize_attribution, attribute_shard


class Scheduler:
    def __init__(
        self,
        spec: SpecBundle,
        ledger: Ledger,
        processes: ProcessManager,
        *,
        run_ids: set[str] | list[str] | None = None,
        force_recompute: set[str] | list[str] | None = None,
        crash_after_inflight: int = 0,
        max_workers: int = 16,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.processes = processes
        self.leases = LeaseManager(processes, spec.session.lease_ttl_ms)
        self.meter = FixedPointMeter(spec.session.scale_exp)
        self.order = topological_order(spec.shards)
        self.rank = {shard_id: index for index, shard_id in enumerate(self.order)}
        self.shards = {shard.shard_id: shard for shard in spec.shards}
        self.nodes = {node.node_id: node for node in spec.nodes}
        if run_ids is None:
            self.target = set(self.shards)
        else:
            self.target = set(run_ids)
        self.force = set(force_recompute or [])
        self.crash_after_inflight = crash_after_inflight
        self.max_workers = max_workers

        self.completed: dict[str, dict[str, Any]] = {}
        self.frozen_skipped: set[str] = set()
        self.active_futures: dict[Any, dict[str, Any]] = {}
        self.active_global_units = 0
        self.active_node_shards: Counter[str] = Counter()
        self.peak_global_units = 0
        self.peak_node_shards: Counter[str] = Counter()
        self.wait_started_ns = {shard_id: monotonic_ns() for shard_id in self.target}
        self.dispatch_trace: list[dict[str, Any]] = []
        self.queue_records: dict[str, dict[str, Any]] = {}
        self.block_reasons: dict[str, str] = {}
        self.executor: ThreadPoolExecutor | None = None

    def run(self) -> dict[str, Any]:
        already_committed = self.ledger.committed_shard_ids()
        for shard_id in list(self.target):
            shard = self.shards[shard_id]
            if shard.state == "frozen":
                self.frozen_skipped.add(shard_id)
                self._trace("frozen_skip", shard_id, detail="frozen shard never reserves budget")
                continue
            if shard_id in already_committed and shard_id not in self.force:
                record = self.ledger.current_outcome(shard_id)
                if record is not None:
                    self.completed[shard_id] = record
                    self._trace("ledger_skip_committed", shard_id)

        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="shard") as executor:
            self.executor = executor
            while True:
                expired = self.leases.expire_due()
                if expired:
                    self._trace(
                        "leases_expired",
                        None,
                        detail=[lease.lease_id for lease in expired],
                    )
                self._dispatch_ready()
                if not self.active_futures:
                    pending = self._unfinished_targets()
                    if not pending:
                        break
                    self._raise_blocked(pending)
                done, _ = wait(
                    list(self.active_futures),
                    timeout=max(0.001, self.spec.session.wait_poll_ms / 1000.0),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    self._finish_future(future)

        current = self.ledger.all_current_outcomes()
        attribution_records = [
            record["error_attribution"]
            for record in current.values()
            if isinstance(record.get("error_attribution"), dict)
        ]
        node_bounds = [
            node_water_error_bounds(
                node_id=node.node_id,
                shard_count=sum(1 for shard in self.spec.shards if shard.node_id == node.node_id),
                scale_exp=self.spec.session.scale_exp,
                concurrency_quota=node.concurrency_quota,
                max_leases=node.max_leases,
            )
            for node in self.spec.nodes
        ]
        return {
            "completed_in_run": sorted(
                shard_id for shard_id in self.completed if shard_id not in already_committed or shard_id in self.force
            ),
            "completed_total": len(current),
            "frozen_skipped": sorted(self.frozen_skipped),
            "topological_order": self.order,
            "dispatch_trace": self.dispatch_trace,
            "queue_records": self.queue_records,
            "global_water_attribution": summarize_attribution(attribution_records),
            "node_water_error_bounds": node_bounds,
            "peak_global_units": self.peak_global_units,
            "peak_node_shards": dict(self.peak_node_shards),
            "replay": self.ledger.assert_replay_ok(),
        }

    def _unfinished_targets(self) -> set[str]:
        return {
            shard_id
            for shard_id in self.target
            if shard_id not in self.completed
            and shard_id not in self.frozen_skipped
            and shard_id not in {info["shard_id"] for info in self.active_futures.values()}
        }

    def _dependencies_ready(self, shard_id: str) -> tuple[bool, dict[str, str], str | None]:
        shard = self.shards[shard_id]
        digests: dict[str, str] = {}
        for upstream in shard.upstreams:
            record = self.completed.get(upstream)
            if record is None:
                if upstream in self.frozen_skipped:
                    return False, {}, f"upstream {upstream} is frozen without a committed summary"
                if upstream in self.target and upstream not in self.completed:
                    return False, {}, f"upstream {upstream} has not been recomputed in this run"
                record = self.ledger.current_outcome(upstream)
                committed = upstream in self.ledger.committed_shard_ids()
                if record is None or not committed:
                    return False, {}, f"upstream {upstream} summary is unavailable"
            digests[upstream] = record["summary"]["aggregate_digest"]
        return True, digests, None

    def _dispatch_ready(self) -> None:
        assert self.executor is not None
        for shard_id in self.order:
            if shard_id not in self.target:
                continue
            if shard_id in self.completed or shard_id in self.frozen_skipped:
                continue
            if any(info["shard_id"] == shard_id for info in self.active_futures.values()):
                continue
            shard = self.shards[shard_id]
            if shard.state == "frozen":
                self.frozen_skipped.add(shard_id)
                self._trace("frozen_skip", shard_id, detail="frozen shard never reserves budget")
                continue

            ready, digests, dependency_reason = self._dependencies_ready(shard_id)
            if not ready:
                self.block_reasons[shard_id] = dependency_reason or "dependency not ready"
                continue

            node = self.nodes[shard.node_id]
            if shard.resource_units > self.spec.session.global_resource_budget:
                raise CapacityError(
                    "shard resource_units exceeds global budget and can never be scheduled",
                    path=f"$.shards[{shard_id}].resource_units",
                    details={"resource_units": shard.resource_units, "budget": self.spec.session.global_resource_budget},
                )
            if node.concurrency_quota == 0 or node.max_leases == 0:
                self.block_reasons[shard_id] = (
                    "node has zero concurrency_quota or max_leases, so shard is unschedulable"
                )
                continue

            reason = self._capacity_block_reason(shard_id)
            if reason:
                self.block_reasons[shard_id] = reason
                if shard_id not in self.queue_records:
                    start_ns = self.wait_started_ns.get(shard_id, monotonic_ns())
                    self.queue_records[shard_id] = {
                        "shard_id": shard_id,
                        "wait_started_monotonic_ms": start_ns // 1_000_000,
                        "wait_started_ns": start_ns,
                        "wait_end_monotonic_ms": None,
                        "wait_duration_ms": None,
                        "last_block_reason": reason,
                    }
                else:
                    self.queue_records[shard_id]["last_block_reason"] = reason
                continue

            self._dispatch_one(shard_id, digests)

    def _capacity_block_reason(self, shard_id: str) -> str | None:
        shard = self.shards[shard_id]
        node = self.nodes[shard.node_id]
        if self.active_global_units + shard.resource_units > self.spec.session.global_resource_budget:
            return "global_resource_budget"
        if self.active_node_shards[node.node_id] >= node.concurrency_quota:
            return "node_concurrency_quota"
        if self.leases.active_lease_count(node.node_id) >= node.max_leases:
            return "node_max_leases"
        return None

    def _dispatch_one(self, shard_id: str, digests: dict[str, str]) -> None:
        assert self.executor is not None
        shard = self.shards[shard_id]
        node = self.nodes[shard.node_id]
        start_ns = self.wait_started_ns.get(shard_id, monotonic_ns())
        dispatch_ns = monotonic_ns()
        contract_digest = sha256_hex(
            {
                "shard": shard.__dict__,
                "node": node.__dict__,
                "format_version": 1,
            }
        )
        lease_id = f"lease-{shard_id}-{contract_digest[:16]}"
        lease = self.leases.issue(
            shard_id=shard_id,
            node_id=node.node_id,
            capacity_version=shard.capacity_version,
            lease_id=lease_id,
        )
        reservation_id = f"resv-{uuid.uuid4().hex}"
        reservation = {
            "reservation_id": reservation_id,
            "shard_id": shard_id,
            "node_id": node.node_id,
            "lease_id": lease_id,
            "resource_units": shard.resource_units,
            "resource_units_q": shard.resource_units * self.meter.scale,
            "global_budget_units": self.spec.session.global_resource_budget,
            "spec_version": self.spec.session.spec_version,
        }
        self.ledger.commit_reservation(reservation)

        self.active_global_units += shard.resource_units
        self.active_node_shards[node.node_id] += 1
        self.peak_global_units = max(self.peak_global_units, self.active_global_units)
        self.peak_node_shards[node.node_id] = max(
            self.peak_node_shards[node.node_id], self.active_node_shards[node.node_id]
        )
        self._assert_invariants()

        wait_record = self.queue_records.get(
            shard_id,
            {
                "shard_id": shard_id,
                "wait_started_monotonic_ms": start_ns // 1_000_000,
                "wait_started_ns": start_ns,
            },
        )
        wait_record["wait_end_monotonic_ms"] = dispatch_ns // 1_000_000
        wait_record["wait_duration_ms"] = max(0, (dispatch_ns - start_ns) // 1_000_000)
        wait_record["dispatched_at_monotonic_ms"] = dispatch_ns // 1_000_000
        self.queue_records[shard_id] = wait_record

        future = self.executor.submit(
            self._execute_shard,
            shard_id,
            digests,
            lease.lease_id,
            lease.version,
        )
        self.active_futures[future] = {
            "shard_id": shard_id,
            "reservation_id": reservation_id,
            "wait": wait_record,
            "dispatch_ns": dispatch_ns,
        }
        self._trace(
            "dispatch",
            shard_id,
            detail={
                "lease_id": lease.lease_id,
                "resource_units": shard.resource_units,
                "active_global_units": self.active_global_units,
                "active_node_shards": dict(self.active_node_shards),
            },
        )
        if self.crash_after_inflight and len(self.active_futures) >= self.crash_after_inflight:
            time.sleep(0.25)
            os.kill(os.getpid(), signal.SIGKILL)

    def _execute_shard(
        self,
        shard_id: str,
        digests: dict[str, str],
        lease_id: str,
        expected_lease_version: int,
    ) -> dict[str, Any]:
        shard = self.shards[shard_id]
        node = self.nodes[shard.node_id]
        result = compute_shard_remotely(self.processes, node, shard, digests)
        lease = self.leases.renew(lease_id, expected_lease_version)
        lease = self.leases.release(lease.lease_id, lease.version)
        result["lease"] = lease.to_dict()
        return result

    def _finish_future(self, future: Any) -> None:
        info = self.active_futures.pop(future)
        shard_id = info["shard_id"]
        shard = self.shards[shard_id]
        node = self.nodes[shard.node_id]
        try:
            result = future.result()
        except Exception:
            self.active_global_units -= shard.resource_units
            self.active_node_shards[node.node_id] -= 1
            try:
                lease_id = self.ledger.active_reservations()
                matching = [item for item in lease_id if item["shard_id"] == shard_id]
                if matching:
                    self.leases.release(matching[0]["lease_id"])
            except Exception:
                pass
            raise

        capacity = self.spec.capacity(shard.capacity_version)
        active_leases_at_completion = self.leases.active_lease_count(node.node_id)
        attribution = attribute_shard(
            self.meter,
            shard_id=shard_id,
            topo_rank=self.rank[shard_id],
            total_shards=len(self.shards),
            resource_units=shard.resource_units,
            capacity_budget_units=capacity.budget_units,
            node_max_leases=node.max_leases,
        )
        wait = info["wait"]
        finish_ns = monotonic_ns()
        timing = {
            "wait_started_monotonic_ms": wait.get("wait_started_monotonic_ms"),
            "wait_ended_monotonic_ms": wait.get("wait_end_monotonic_ms"),
            "wait_duration_ms": wait.get("wait_duration_ms"),
            "dispatch_to_commit_ms": max(0, (finish_ns - int(info["dispatch_ns"])) // 1_000_000),
            "node_aggregate_elapsed_ms": result["node_elapsed_ms"],
            "rpc_receipt_count": len(result["batch_receipts"]),
        }
        resource = {
            "reservation_id": info["reservation_id"],
            "resource_units": shard.resource_units,
            "resource_units_q": shard.resource_units * self.meter.scale,
            "global_budget_units": self.spec.session.global_resource_budget,
            "global_budget_q": self.spec.session.global_resource_budget * self.meter.scale,
            "global_active_units_at_completion": self.active_global_units,
            "global_active_units_q_at_completion": self.active_global_units * self.meter.scale,
            "peak_global_units": self.peak_global_units,
            "peak_global_units_q": self.peak_global_units * self.meter.scale,
            "node_active_shards_at_completion": self.active_node_shards[node.node_id],
            "node_active_shards_q_at_completion": self.active_node_shards[node.node_id] * self.meter.scale,
            "node_active_leases_at_completion": active_leases_at_completion,
            "node_active_leases_q_at_completion": active_leases_at_completion * self.meter.scale,
            "peak_node_shards": self.peak_node_shards[node.node_id],
            "peak_node_shards_q": self.peak_node_shards[node.node_id] * self.meter.scale,
            "node_concurrency_quota": node.concurrency_quota,
            "node_concurrency_quota_q": node.concurrency_quota * self.meter.scale,
            "node_max_leases": node.max_leases,
            "node_max_leases_q": node.max_leases * self.meter.scale,
        }
        water = {
            "scale_exp": self.meter.scale_exp,
            "scale": self.meter.scale,
            "shard_water": attribution,
            "node_error_bounds": node_water_error_bounds(
                node_id=node.node_id,
                shard_count=sum(1 for item in self.spec.shards if item.node_id == node.node_id),
                scale_exp=self.spec.session.scale_exp,
                concurrency_quota=node.concurrency_quota,
                max_leases=node.max_leases,
            ),
        }
        record = {
            "shard_id": shard_id,
            "node_id": node.node_id,
            "spec_version": self.spec.session.spec_version,
            "sample_set_digest": result["sample_set_digest"],
            "summary": result["summary"],
            "dependency_digests": digests_or_none(result, info),
            "lease": result["lease"],
            "resource": resource,
            "water": water,
            "error_attribution": attribution,
            "timing": timing,
            "rpc_receipts": result["batch_receipts"],
            "reservation_id": info["reservation_id"],
        }
        committed = self.ledger.commit_shard_outcome(record, topo_order=self.order)
        self.completed[shard_id] = committed
        self.active_global_units -= shard.resource_units
        self.active_node_shards[node.node_id] -= 1
        self._assert_invariants()
        self._trace(
            "commit",
            shard_id,
            detail={
                "digest": committed["content_digest"],
                "active_global_units": self.active_global_units,
            },
        )

    def _assert_invariants(self) -> None:
        if self.active_global_units < 0 or self.active_global_units > self.spec.session.global_resource_budget:
            raise CapacityError(
                "global resource budget invariant violated",
                path="$.session.global_resource_budget",
                details={"active": self.active_global_units, "budget": self.spec.session.global_resource_budget},
            )
        for node in self.spec.nodes:
            active_shards = self.active_node_shards[node.node_id]
            if active_shards < 0 or active_shards > node.concurrency_quota:
                raise CapacityError(
                    "node concurrency quota invariant violated",
                    path=f"$.nodes[{node.node_id}].concurrency_quota",
                    details={"active": active_shards, "quota": node.concurrency_quota},
                )
            active_leases = self.leases.active_lease_count(node.node_id)
            if active_leases > node.max_leases:
                raise CapacityError(
                    "node lease cap invariant violated",
                    path=f"$.nodes[{node.node_id}].max_leases",
                    details={"active_leases": active_leases, "max_leases": node.max_leases},
                )

    def _raise_blocked(self, pending: set[str]) -> None:
        details = {shard_id: self.block_reasons.get(shard_id, "unknown") for shard_id in sorted(pending)}
        if any("zero concurrency_quota" in reason for reason in details.values()):
            raise CapacityError(
                "one or more pending shards target a node with zero scheduling capacity",
                path="$.nodes.concurrency_quota",
                details=details,
            )
        raise WaterLeaseError(
            "scheduler has no running future but shards remain blocked",
            code_override="E_SCHED_BLOCKED",
            path="$.shards",
            details={"pending": sorted(pending), "reasons": details},
        )

    def _trace(self, event: str, shard_id: str | None, detail: Any = None) -> None:
        self.dispatch_trace.append(
            {
                "event": event,
                "shard_id": shard_id,
                "monotonic_ms": monotonic_ns() // 1_000_000,
                "detail": detail,
            }
        )


def digests_or_none(result: dict[str, Any], info: dict[str, Any]) -> dict[str, str]:
    upstream = result["summary"].get("upstream_digests", [])
    return {item["shard_id"]: item["aggregate_digest"] for item in upstream}
