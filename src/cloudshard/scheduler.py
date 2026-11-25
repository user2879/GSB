from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from .adjudicate import Adjudicator
from .aggregate import aggregate_samples
from .client import dispatch_batch, grant_lease, release_lease
from .errors import CloudShardError, VerifyError
from .ledger import Ledger, monotonic_ms
from .samples import generate_samples
from .spec import ShardSpec, Spec, stable_dispatch_order, topological_shards


DispatchFunc = Callable[..., dict[str, Any]]


class DispatchEngine:
    def __init__(
        self,
        spec: Spec,
        ledger: Ledger,
        *,
        dispatcher: DispatchFunc | None = None,
        lease_client: Any | None = None,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.adj = Adjudicator.from_spec(spec)
        row = ledger.conn.execute("SELECT COALESCE(MAX(event_seq), 0) AS s FROM event_log WHERE stream='runtime'").fetchone()
        self._runtime_seq = int(row["s"])
        row = ledger.conn.execute("SELECT COALESCE(MAX(sequence_no), 0) AS s FROM ownership WHERE stream='runtime'").fetchone()
        self._ownership_seq = int(row["s"])
        self.committed_revisions = {
            row["shard_id"]: int(row["revision"])
            for row in ledger.conn.execute("SELECT shard_id, revision FROM current_view")
        }
        self.canceled_shards: set[str] = set()
        for event in spec.events:
            if event["type"] == "cancel_shard":
                self.canceled_shards.add(event["shard_id"])
        self.wait_started_ms: dict[str, int] = {}
        self.dispatch_func = dispatcher or self._rpc_dispatch
        self.lease_client = lease_client or self._rpc_lease
        self.topological_order = [shard.shard_id for shard in topological_shards(spec)]
        self.dispatch_order = stable_dispatch_order(spec)
        self.crash_before_commit: str | None = None
        self.crash_marker: Path | None = None

    def _alloc_event_seq(self) -> int:
        self._runtime_seq += 1
        return self._runtime_seq

    def _alloc_ownership_seq(self) -> int:
        self._ownership_seq += 1
        return self._ownership_seq

    def cancel_descendants(self) -> set[str]:
        canceled = set(self.canceled_shards)
        changed = True
        while changed:
            changed = False
            for shard in self.spec.shards:
                if shard.shard_id in canceled:
                    continue
                if any(dep in canceled for dep in shard.dependencies):
                    canceled.add(shard.shard_id)
                    changed = True
        self.canceled_shards = canceled
        return canceled

    def _required_revision(self, shard_id: str, target_revision: int, target_set: set[str]) -> int:
        if shard_id in target_set:
            return target_revision
        return int(self.committed_revisions.get(shard_id, target_revision))

    def _is_ready(self, shard: ShardSpec, target_revision: int, target_set: set[str]) -> bool:
        for dep_id in sorted(shard.dependencies):
            required = self._required_revision(dep_id, target_revision, target_set)
            if self.committed_revisions.get(dep_id) != required:
                return False
        return True

    def _choose_node(self, shard: ShardSpec) -> tuple[Any | None, str | None]:
        explicit_node = shard.raw.get("node_id")
        nodes = sorted(self.spec.nodes, key=lambda n: n.node_id)
        if explicit_node:
            nodes = [self.spec.node_by_id[explicit_node]]
        used_units = self.adj._used_units()
        if used_units + shard.resource_units > self.spec.global_budget_units:
            return None, "global_budget_units"
        eligible: list[tuple[int, str, Any]] = []
        for node in nodes:
            active = self.adj._active_node_claims(node.node_id)
            if len(active) >= node.lease_limit:
                continue
            if len(active) >= node.concurrency_quota:
                continue
            eligible.append((len(active), node.node_id, node))
        if eligible:
            return min(eligible, key=lambda item: (item[0], item[1]))[2], None
        return None, "node_lease_limit_or_concurrency_quota"

    def run_target(self, target_revision: int, target_shard_ids: list[str] | tuple[str, ...] | set[str]) -> dict[str, Any]:
        target_set = set(target_shard_ids)
        self.cancel_descendants()
        pending = set(target_set - self.canceled_shards)
        pending -= {sid for sid in pending if self.committed_revisions.get(sid) == target_revision}
        max_workers = max(1, sum(node.concurrency_quota for node in self.spec.nodes))
        in_flight: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        dispatch_count = 0
        wait_records: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cloudshard-dispatch") as pool:
            while pending or in_flight:
                dispatched_any = False
                for shard in self.dispatch_order:
                    if shard.shard_id not in pending:
                        continue
                    if shard.shard_id in self.canceled_shards:
                        pending.remove(shard.shard_id)
                        self.wait_started_ms.pop(shard.shard_id, None)
                        continue
                    if not self._is_ready(shard, target_revision, target_set):
                        continue
                    node, constraint = self._choose_node(shard)
                    if node is None:
                        started = self.wait_started_ms.get(shard.shard_id)
                        if started is None:
                            started = monotonic_ms()
                            self.wait_started_ms[shard.shard_id] = started
                            wait_records.append(
                                {
                                    "shard_id": shard.shard_id,
                                    "wait_started_ms": started,
                                    "constraint": constraint,
                                    "event": "wait_started",
                                }
                            )
                        continue

                    wait_started = self.wait_started_ms.pop(shard.shard_id, None)
                    wait_ended = monotonic_ms() if wait_started is not None else None
                    if wait_started is not None:
                        wait_records.append(
                            {
                                "shard_id": shard.shard_id,
                                "wait_started_ms": wait_started,
                                "wait_ended_ms": wait_ended,
                                "duration_ms": max(0, (wait_ended or wait_started) - wait_started),
                                "event": "wait_ended",
                            }
                        )
                    context = self._reserve_and_submit(
                        shard=shard,
                        node=node,
                        target_revision=target_revision,
                        wait_started=wait_started,
                        wait_ended=wait_ended,
                        pool=pool,
                        in_flight=in_flight,
                    )
                    pending.remove(shard.shard_id)
                    dispatch_count += 1
                    dispatched_any = True

                if in_flight:
                    done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                    for future in done:
                        context = in_flight.pop(future)
                        self._complete_future(future, context, target_revision)
                elif pending and not dispatched_any:
                    unready = sorted(
                        sid for sid in pending if not self._is_ready(self.spec.shard_by_id[sid], target_revision, target_set)
                    )
                    raise CloudShardError(
                        "scheduler cannot proceed: no capacity and no task in flight, or dependencies are unfinished",
                        basis={"pending": sorted(pending), "unready": unready},
                    )

        return {
            "target_revision": target_revision,
            "target_shard_ids": sorted(target_set),
            "dispatch_count": dispatch_count,
            "committed_revisions": dict(sorted(self.committed_revisions.items())),
            "wait_records": wait_records,
            "canceled_shard_ids": sorted(self.canceled_shards),
        }

    def _reserve_and_submit(
        self,
        *,
        shard: ShardSpec,
        node: Any,
        target_revision: int,
        wait_started: int | None,
        wait_ended: int | None,
        pool: ThreadPoolExecutor,
        in_flight: dict[Future[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        attempt = self.ledger.next_attempt(shard.shard_id, target_revision)
        lease_id = f"lease-r{target_revision}-{shard.shard_id}-a{attempt}"
        event_seq = self._alloc_event_seq()
        now = monotonic_ms()
        ttl_ms = self.spec.lease_ttl_ms
        event = {
            "seq": event_seq,
            "type": "lease_apply",
            "lease_id": lease_id,
            "shard_id": shard.shard_id,
            "node_id": node.node_id,
            "at_ms": now,
            "ttl_ms": ttl_ms,
            "resource_units": shard.resource_units,
            "priority": shard.priority,
            "quota_version": self.adj.quota_versions.get(node.node_id, node.capacity_version),
        }
        ownership = self.adj.handle_event(event)[0]
        ownership["sequence"] = self._alloc_ownership_seq()
        claim = self.adj.claims[lease_id]
        samples = generate_samples(self.spec, shard, node)
        revision = next(rev for rev in self.spec.revisions if rev.revision == target_revision)
        context = {
            "shard": shard,
            "node": node,
            "revision": target_revision,
            "attempt": attempt,
            "lease_id": lease_id,
            "event": event,
            "ownership": ownership,
            "samples": samples,
            "config": revision.config,
            "wait_started_ms": wait_started,
            "wait_ended_ms": wait_ended,
            "reserved_ms": now,
        }
        attempt_record = {
            "shard_id": shard.shard_id,
            "node_id": node.node_id,
            "revision": target_revision,
            "attempt": attempt,
            "lease_id": lease_id,
            "status": "DISPATCHED",
            "resource_units": shard.resource_units,
            "priority": shard.priority,
            "wait_started_ms": wait_started,
            "wait_ended_ms": wait_ended,
            "dispatched_ms": now,
            "event": event,
            "ownership": ownership,
            "lease": claim,
            "occupancy": {
                "shard_id": shard.shard_id,
                "attempt": attempt,
                "lease_id": lease_id,
                "node_id": node.node_id,
                "revision": target_revision,
                "resource_units": shard.resource_units,
                "state": "HELD",
                "quota_version": event["quota_version"],
                "event_seq": event_seq,
                    "started_ms": now,
            },
        }
        self.ledger.append_event("runtime", event)
        self.ledger.record_dispatch(attempt_record, ownership)
        future = pool.submit(self.dispatch_func, context)
        in_flight[future] = context
        return context

    @staticmethod
    def _rpc_lease(action: str, node: Any, payload: dict[str, Any]) -> dict[str, Any]:
        if action == "grant":
            return grant_lease(node, payload, timeout_ms=max(500, 2000))
        if action == "renew":
            from .client import renew_lease

            return renew_lease(node, timeout_ms=max(500, 2000), **payload)
        if action == "release":
            return release_lease(node, payload["lease_id"], timeout_ms=max(500, 2000))
        raise ValueError(action)

    def _rpc_dispatch(self, context: dict[str, Any]) -> dict[str, Any]:
        node = context["node"]
        lease = {
            "lease_id": context["lease_id"],
            "shard_id": context["shard"].shard_id,
            "lease_version": 1,
            "expires_at_ms": context["event"]["at_ms"] + context["event"]["ttl_ms"],
        }
        grant_lease(node, lease, timeout_ms=self.spec.socket_timeout_ms)
        try:
            receipt = dispatch_batch(
                node,
                shard_id=context["shard"].shard_id,
                revision=context["event"].get("revision", context.get("revision", 0)) or context.get("revision_number", 0),
                samples=context["samples"],
                config=context["config"],
                request_id=context["lease_id"],
                timeout_ms=self.spec.socket_timeout_ms,
            )
        finally:
            try:
                release_lease(node, context["lease_id"], timeout_ms=self.spec.socket_timeout_ms)
            except Exception:
                pass
        return {"receipt": receipt}

    def _complete_future(self, future: Future[dict[str, Any]], context: dict[str, Any], target_revision: int) -> None:
        shard: ShardSpec = context["shard"]
        node = context["node"]
        try:
            output = future.result()
        except Exception:
            release_seq = self._alloc_event_seq()
            release_event = {
                "seq": release_seq,
                "type": "lease_release",
                "lease_id": context["lease_id"],
                "at_ms": monotonic_ms(),
                "quota_version": context["event"]["quota_version"],
                "error_recovery": True,
            }
            try:
                self.adj.handle_event(release_event)
            except Exception:
                pass
            raise

        receipt = output["receipt"]
        expected_revision = target_revision
        if int(receipt["revision"]) != expected_revision:
            raise VerifyError(
                "node returned an unexpected revision",
                shard_id=shard.shard_id,
                field="revision",
                basis={"expected": expected_revision, "actual": receipt["revision"]},
            )
        local_summary, local_digest = aggregate_samples(context["samples"], node, context["config"])
        if local_digest != receipt["summary_digest"]:
            raise VerifyError(
                "recomputed aggregate digest does not match node receipt",
                shard_id=shard.shard_id,
                field="summary_digest",
                basis={"expected": local_digest, "receipt": receipt["summary_digest"]},
            )

        result = {
            "shard_id": shard.shard_id,
            "revision": target_revision,
            "attempt": context["attempt"],
            "node_id": node.node_id,
            "lease_id": context["lease_id"],
            "state": "COMMITTED",
            "summary": receipt["summary"],
            "summary_digest": receipt["summary_digest"],
            "receipt": receipt,
            "elapsed_ms": receipt["elapsed_ms"],
            "resource_units": shard.resource_units,
            "wait_started_ms": context.get("wait_started_ms"),
            "wait_ended_ms": context.get("wait_ended_ms"),
        }
        if self.crash_before_commit == shard.shard_id:
            self.ledger.open_uncommitted_outcome_for_crash(result)
            marker = self.crash_marker or Path("crash-marker.json")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                __import__("json").dumps(
                    {
                        "ready": True,
                        "pid": __import__("os").getpid(),
                        "shard_id": shard.shard_id,
                        "attempt": context["attempt"],
                        "ledger": str(self.ledger.path),
                    }
                ),
                encoding="utf-8",
            )
            while True:
                __import__("time").sleep(0.05)

        release_seq = self._alloc_event_seq()
        release_event = {
            "seq": release_seq,
            "type": "lease_release",
            "lease_id": context["lease_id"],
            "at_ms": monotonic_ms(),
            "quota_version": context["event"]["quota_version"],
        }
        ownership_record = self.adj.handle_event(release_event)[0]
        ownership_record["sequence"] = self._alloc_ownership_seq()
        completion_seq = self._alloc_event_seq()
        result.update(
            {
                "release_event": release_event,
                "completion_seq": completion_seq,
                "ownership_record": ownership_record,
            }
        )
        self.ledger.commit_shard_result(result, topology_order=self.topological_order, revision=target_revision)
        self.committed_revisions[shard.shard_id] = target_revision
