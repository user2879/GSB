"""Centre-side scheduling engine.

The engine drives real node subprocesses while keeping every ledger artifact
deterministic:

* admission is constrained by the global concurrency cap, each node's lease
  limit, integer capacity slots / watermarks and the per-batch point budget;
* expired leases are really reclaimed (tagged node release frees the slot and
  global token) and recorded as ``E_LEASE_EXPIRED``;
* every artifact is content-addressed by :mod:`fingerprints`, so unchanged
  nodes/shards are reused and only affected keys are recomputed;
* concurrent work is collected back in shard-index order, which makes the
  business ledger, arbitration rationale, reuse list and failure list
  field-identical to a serial full rebuild regardless of completion order.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any

from .arbitration import ArbitrationResult, ShardConclusion, arbitrate, merge_conclusions
from .clock import Clock, DeterministicClock, SystemClock
from .config import BatchConfig
from .errors import (
    E_BUDGET_EXHAUSTED,
    E_LEASE_EXPIRED,
    E_LEASE_LIMIT,
    E_NODE_UNAVAILABLE,
    EdgeCacheError,
    Failure,
)
from .fingerprints import (
    authority_key,
    conclusion_key,
    ledger_entry_key,
    node_summary_key,
    serialization_key,
)
from .ledger import Ledger
from .node_pool import NodePool
from .receipts import (
    C1,
    C2,
    C3,
    Receipt,
    ReceiptFormatCodec,
    integrity_label_for,
    precision_label_for,
)
from .leases import Lease, LeaseManager

LEASE_TTL_MS = 5_000


@dataclass
class ShardTask:
    shard_id: str
    index: int
    slots: int
    budget_points: int


@dataclass
class NodeOutcome:
    shard_id: str
    node_id: str
    receipt: Receipt | None
    compute_payload: dict[str, Any] | None
    failure: Failure | None
    lease_events: list[dict[str, Any]] = field(default_factory=list)
    reused: bool = False


class DeliveryEngine:
    def __init__(
        self,
        config: BatchConfig,
        ledger: Ledger,
        pool: NodePool,
        clock: Clock | None = None,
        lease_ttl_ms: int = LEASE_TTL_MS,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.pool = pool
        self.clock = clock or SystemClock()
        self.lease_ttl_ms = lease_ttl_ms
        self.quota_seq = {quota.version: quota.seq for quota in config.quotas}
        self.codec = ReceiptFormatCodec(self.quota_seq)
        self.lease_manager = LeaseManager(
            nodes={node.node_id: node for node in config.nodes},
            global_concurrency=config.global_concurrency,
            clock=self.clock,
            reclamation_callback=self._reclaim_on_node,
        )
        self.budget_consumed = 0
        self.arb_results: dict[str, ArbitrationResult] = {}
        self.conclusions: dict[str, ShardConclusion] = {}
        self.reuse_invalidated: list[str] = []
        self.reuse_reused: list[str] = []
        self.reuse_recomputed: list[str] = []
        self._outcomes: dict[str, list[NodeOutcome]] = {}

    # -- lease helpers -----------------------------------------------------
    def _reclaim_on_node(self, lease: Lease, now_ms: int) -> None:
        client = self.pool.client(lease.node_id)
        response = client.request({
            "type": "release",
            "lease_id": lease.lease_id,
            "tag": lease.tag,
            "at_ms": now_ms,
        })
        if response.get("ok"):
            lease.tag = int(response["tag"])
            self.ledger.add_algebra({
                "lease_id": lease.lease_id,
                "shard_id": lease.shard_id,
                "node_id": lease.node_id,
                "old_tag": response["tag"] - 1,
                "new_tag": response["tag"],
                "action": "expiry-release",
                "at_ms": now_ms,
            })
        else:
            raise EdgeCacheError(Failure.from_dict(response["error"]))

    def _lease_id(self, node_id: str, shard_id: str) -> str:
        # Deterministic per (configuration, node, shard); the engine releases
        # the lease before the next batch, so the same id can be safely reused.
        return f"{self.config.source_hash[:12]}::{node_id}::{shard_id}"

    # -- per-node task -----------------------------------------------------
    def _run_node_task(self, task: ShardTask, node_id: str) -> NodeOutcome:
        node = self.config.node(node_id)
        lease_id = self._lease_id(node_id, task.shard_id)
        events: list[dict[str, Any]] = []
        now_ms = self.clock.now_ms()
        lease = self.lease_manager.leases.get(lease_id)
        if lease is None:
            lease = Lease(
                lease_id=lease_id,
                node_id=node_id,
                shard_id=task.shard_id,
                tag=-1,
                issued_ms=now_ms,
                expires_ms=now_ms + self.lease_ttl_ms,
                slots=task.slots,
            )
            self.lease_manager.register(lease)
        try:
            client = self.pool.client(node_id)
        except EdgeCacheError as exc:
            self.lease_manager.remove(lease_id)
            return NodeOutcome(task.shard_id, node_id, None, None, exc.failure, events)

        summary_key = node_summary_key(self.config, node, task.shard_id)
        cached = self.ledger.load_artifact(summary_key)
        if cached is not None and cached.get("shard_id") == task.shard_id:
            # Cache hits still occupied a planned capacity slot during
            # admission; release it immediately so the scheduling trajectory
            # matches a clean-slate rebuild that would compute on the node.
            self.lease_manager.remove(lease_id)
            receipt = self._receipt_from_payload(cached, str(cached.get("node_id", node_id)), now_ms)
            self.ledger.record_reuse("reuse-hit", summary_key, {
                "shard_id": task.shard_id, "node_id": node_id, "scope": "node-summary",
            })
            return NodeOutcome(task.shard_id, node_id, receipt, cached, None, events, reused=True)

        try:
            acquired = client.request({
                "type": "acquire",
                "lease_id": lease_id,
                "shard_id": task.shard_id,
                "at_ms": now_ms,
            })
            if not acquired.get("ok"):
                failure = Failure.from_dict(acquired["error"])
                return NodeOutcome(task.shard_id, node_id, None, None, failure, events)
            tag = int(acquired["tag"])
            events.append({
                "lease_id": lease_id, "shard_id": task.shard_id, "node_id": node_id,
                "old_tag": -1, "new_tag": tag, "action": "acquire", "at_ms": now_ms,
            })
            lease.tag = tag
            lease.issued_ms = now_ms
            lease.expires_ms = now_ms + self.lease_ttl_ms

            compute_ms = self.clock.now_ms()
            computed = client.request({
                "type": "compute",
                "lease_id": lease_id,
                "tag": tag,
                "shard_id": task.shard_id,
                "at_ms": compute_ms,
            })
            if not computed.get("ok"):
                failure = Failure.from_dict(computed["error"])
                self._safe_release(client, lease, events)
                return NodeOutcome(task.shard_id, node_id, None, None, failure, events)

            release_ms = self.clock.now_ms()
            released = client.request({
                "type": "release",
                "lease_id": lease_id,
                "tag": tag,
                "at_ms": release_ms,
            })
            if not released.get("ok"):
                failure = Failure.from_dict(released["error"])
                return NodeOutcome(task.shard_id, node_id, None, None, failure, events)
            new_tag = int(released["tag"])
            events.append({
                "lease_id": lease_id, "shard_id": task.shard_id, "node_id": node_id,
                "old_tag": tag, "new_tag": new_tag,
                "action": "release", "at_ms": release_ms,
            })
            lease.tag = new_tag
            self.lease_manager.remove(lease_id)
            receipt = self._receipt_from_payload(
                computed, str(computed.get("node_id", node_id)), int(computed["duration_ms"])
            )
            self.ledger.store_artifact(summary_key, computed)
            self.ledger.record_reuse("recompute", summary_key, {
                "shard_id": task.shard_id, "node_id": node_id, "scope": "node-summary",
            })
            return NodeOutcome(task.shard_id, node_id, receipt, computed, None, events)
        except EdgeCacheError as exc:
            self.lease_manager.remove(lease_id)
            return NodeOutcome(task.shard_id, node_id, None, None, exc.failure, events)
        except Exception as exc:  # pragma: no cover - defensive
            self.lease_manager.remove(lease_id)
            failure = Failure(E_NODE_UNAVAILABLE, "node task failed",
                              {"error": str(exc)}, shard_id=task.shard_id, node_id=node_id)
            return NodeOutcome(task.shard_id, node_id, None, None, failure, events)

    def _safe_release(self, client: Any, lease: Lease, events: list[dict[str, Any]]) -> None:
        now_ms = self.clock.now_ms()
        try:
            response = client.request({
                "type": "release", "lease_id": lease.lease_id,
                "tag": lease.tag, "at_ms": now_ms,
            })
            if response.get("ok"):
                events.append({
                    "lease_id": lease.lease_id, "shard_id": lease.shard_id,
                    "node_id": lease.node_id, "old_tag": lease.tag,
                    "new_tag": int(response["tag"]), "action": "release", "at_ms": now_ms,
                })
                lease.tag = int(response["tag"])
        finally:
            self.lease_manager.remove(lease.lease_id)

    def _receipt_from_payload(self, payload: dict[str, Any], node_id: str,
                              clock_ms: int) -> Receipt:
        fixed_point = int(payload["fixed_point"])
        residual = int(payload["residual"])
        receipt = Receipt(
            shard_id=str(payload["shard_id"]),
            index=self.config.shard(str(payload["shard_id"])).index,
            node_id=node_id,
            check_rate_permille=int(payload["check_rate_permille"]),
            residual=residual,
            check_rate_float=float(payload["check_rate_float"]),
            quota_version=str(payload["quota_version"]),
            quota_code=self.quota_seq[str(payload["quota_version"])],
            digest=str(payload["digest"]),
            duplicate_ref=None,
            fixed_point=fixed_point,
            precision_label=precision_label_for(fixed_point, residual),
            clock_ms=clock_ms,
        )
        receipt.integrity_label = integrity_label_for(receipt)
        return receipt

    # -- deterministic planning -------------------------------------------
    def _plan_admissions(self, tasks: list[ShardTask]) -> tuple[list[tuple[ShardTask, str]], list[Failure]]:
        """Pure deterministic admission plan.

        Shards are considered in index order and assigned to the first node
        whose *planned* active-lease count and occupied integer slots stay
        within limits.  Because the plan only mutates local counters (never
        thread-visible state), two runs over the same configuration produce
        the identical (shard, node) sequence regardless of completion order.
        """
        plan: list[tuple[ShardTask, str]] = []
        failures: list[Failure] = []
        active = {node.node_id: 0 for node in self.config.nodes}
        occupied = {node.node_id: 0 for node in self.config.nodes}
        budget = self.budget_consumed
        for task in tasks:
            if budget + task.budget_points > self.config.batch_point_budget:
                failures.append(Failure(
                    E_BUDGET_EXHAUSTED,
                    "shard rejected by batch point budget",
                    {"budget": self.config.batch_point_budget,
                     "consumed": budget, "required": task.budget_points},
                    shard_id=task.shard_id,
                ))
                continue
            chosen = None
            for node in sorted(self.config.nodes, key=lambda item: item.node_id):
                if active[node.node_id] >= node.lease_limit:
                    continue
                if occupied[node.node_id] + task.slots > node.capacity_slots:
                    continue
                projected = occupied[node.node_id] + task.slots
                hard = node.capacity_slots * node.hard_watermark_permille // 1000
                if projected > hard:
                    continue
                chosen = node
                break
            if chosen is None:
                failures.append(Failure(
                    E_LEASE_LIMIT,
                    "no schedulable capacity for shard",
                    {"shard_id": task.shard_id, "slots": task.slots},
                    shard_id=task.shard_id,
                ))
                continue
            active[chosen.node_id] += 1
            occupied[chosen.node_id] += task.slots
            budget += task.budget_points
            plan.append((task, chosen.node_id))
        return plan, failures

    # -- batch orchestration ----------------------------------------------
    def run_batch(self, exchange_format: str = C3) -> dict[str, Any]:
        if exchange_format not in (C1, C2, C3):
            raise EdgeCacheError(Failure(E_NODE_UNAVAILABLE, "unknown format",
                                         {"format": exchange_format}))
        tasks = [
            ShardTask(
                shard_id=shard.shard_id,
                index=shard.index,
                slots=shard.size_units,
                budget_points=shard.size_units,
            )
            for shard in sorted(self.config.shards, key=lambda item: item.index)
        ]
        plan, failures = self._plan_admissions(tasks)
        self.budget_consumed += sum(task.budget_points for task, _ in plan)

        # Real expiry sweep before any work starts.
        now_ms = self.clock.now_ms()
        for _, expiry_failure in self.lease_manager.reclaim_expired(now_ms):
            failures.append(expiry_failure)

        # Register planned leases so runtime watermark bookkeeping mirrors the
        # deterministic plan, then execute under the global concurrency cap.
        for task, node_id in plan:
            now_ms = self.clock.now_ms()
            node = self.config.node(node_id)
            rejection = self.lease_manager.check_admission(
                node, task.slots, now_ms,
                self.config.batch_point_budget - self.budget_consumed + task.budget_points,
            )
            if rejection is not None:
                failures.append(Failure(
                    rejection.code, rejection.message,
                    {**rejection.context, "shard_id": task.shard_id},
                    shard_id=task.shard_id, node_id=node_id,
                ))
                continue
            self.lease_manager.register(Lease(
                lease_id=self._lease_id(node_id, task.shard_id),
                node_id=node_id,
                shard_id=task.shard_id,
                tag=-1,
                issued_ms=now_ms,
                expires_ms=now_ms + self.lease_ttl_ms,
                slots=task.slots,
            ))

        max_workers = max(1, min(self.config.global_concurrency, len(plan) or 1))
        outcomes: list[tuple[ShardTask, str, NodeOutcome]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._run_node_task, task, node_id): (task, node_id)
                for task, node_id in plan
            }
            for future in concurrent.futures.as_completed(futures):
                task, node_id = futures[future]
                outcomes.append((task, node_id, future.result()))

        # Absorb outcomes back in deterministic shard-index order.
        order = {(task.shard_id, node_id): index
                 for index, (task, node_id) in enumerate(plan)}
        for task, node_id, outcome in sorted(
            outcomes, key=lambda item: order[(item[0].shard_id, item[1])]
        ):
            self._absorb_outcome(task, node_id, outcome, failures)

        now_ms = self.clock.now_ms()
        for _, expiry_failure in self.lease_manager.reclaim_expired(now_ms):
            failures.append(expiry_failure)

        self._arbitrate_and_conclude(failures)
        return self._persist_batch(exchange_format, failures)

    def _absorb_outcome(self, task: ShardTask, node_id: str,
                        outcome: NodeOutcome, failures: list[Failure]) -> None:
        self.ledger.add_algebra_many(outcome.lease_events)
        if outcome.failure is not None:
            failures.append(Failure(
                outcome.failure.code, outcome.failure.message,
                {**outcome.failure.context, "shard_id": task.shard_id, "node_id": node_id},
                shard_id=task.shard_id, node_id=node_id,
            ))
            return
        if outcome.reused:
            self.reuse_reused.append(
                node_summary_key(self.config, self.config.node(node_id), task.shard_id)
            )
        else:
            self.reuse_recomputed.append(
                node_summary_key(self.config, self.config.node(node_id), task.shard_id)
            )
        self._outcomes.setdefault(task.shard_id, []).append(outcome)

    def _arbitrate_and_conclude(self, failures: list[Failure]) -> None:
        for shard in sorted(self.config.shards, key=lambda item: item.index):
            outcomes = self._outcomes.get(shard.shard_id, [])
            candidates = [outcome.receipt for outcome in outcomes if outcome.receipt is not None]
            result = arbitrate(shard.shard_id, candidates, self.quota_seq)
            self.arb_results[shard.shard_id] = result
            a_key = authority_key(self.config, shard.shard_id)
            if result.authority is not None:
                if self.ledger.has_artifact(a_key):
                    self.ledger.record_reuse("reuse-hit", a_key,
                                             {"shard_id": shard.shard_id, "scope": "authority"})
                    self.reuse_reused.append(a_key)
                else:
                    self.ledger.store_artifact(a_key, {
                        "shard_id": shard.shard_id,
                        "tied": result.tied,
                        "rationale": result.rationale,
                        "authority": result.authority.to_dict(),
                        "candidates": [receipt.to_dict() for receipt in result.candidates],
                    })
                    self.ledger.record_reuse("recompute", a_key,
                                             {"shard_id": shard.shard_id, "scope": "authority"})
                    self.reuse_recomputed.append(a_key)
            if result.failure is not None:
                failures.append(result.failure)
        self.conclusions = merge_conclusions(self.config, self.arb_results)
        for shard in sorted(self.config.shards, key=lambda item: item.index):
            conclusion = self.conclusions[shard.shard_id]
            c_key = conclusion_key(self.config, shard.shard_id)
            if self.ledger.has_artifact(c_key):
                self.ledger.record_reuse("reuse-hit", c_key,
                                         {"shard_id": shard.shard_id, "scope": "conclusion"})
                self.reuse_reused.append(c_key)
            else:
                self.ledger.store_artifact(c_key, conclusion.to_dict())
                self.ledger.record_reuse("recompute", c_key,
                                         {"shard_id": shard.shard_id, "scope": "conclusion"})
                self.reuse_recomputed.append(c_key)
            if not conclusion.available:
                failures.append(Failure(
                    E_NODE_UNAVAILABLE if conclusion.failure_code is None else conclusion.failure_code,
                    "shard conclusion unavailable",
                    {"shard_id": shard.shard_id,
                     "upstream_unavailable": list(conclusion.upstream_unavailable)},
                    shard_id=shard.shard_id,
                ))

    def _persist_batch(self, exchange_format: str,
                       failures: list[Failure]) -> dict[str, Any]:
        counts = {C1: 0, C2: 0, C3: 0}
        for shard in sorted(self.config.shards, key=lambda item: item.index):
            conclusion = self.conclusions[shard.shard_id]
            result = self.arb_results[shard.shard_id]
            wire_lines: dict[str, str] = {}
            # Persist all three formats for a complete exchange record; the
            # run's declared exchange format selects the primary wire view.
            if result.authority is not None:
                for fmt in (C1, C2, C3):
                    line = self.codec.encode(result.authority, fmt)
                    wire_lines[fmt] = line
                    counts[fmt] += 1
                    s_key = serialization_key(self.config, shard.shard_id, fmt)
                    self.ledger.store_artifact(s_key, {
                        "shard_id": shard.shard_id, "format": fmt, "line": line,
                        "sidecar": self.codec.sidecar.get(f"idx-{shard.index}", {}),
                    })
            entry = {
                "index": shard.index,
                "shard_id": shard.shard_id,
                "conclusion": conclusion.to_dict(),
                "arbitration": {
                    "tied": result.tied,
                    "rationale": result.rationale,
                    "candidate_node_ids": [candidate.node_id for candidate in result.candidates],
                },
                "wire": wire_lines,
                "exchange_format": exchange_format,
            }
            self.ledger.upsert_business(entry)
            l_key = ledger_entry_key(self.config, shard.shard_id)
            self.ledger.store_artifact(l_key, entry)
        # Deterministic failure ordering: shard index, then code, then node.
        ordered_failures = sorted(
            failures,
            key=lambda failure: (
                self.config.shard(failure.shard_id).index if failure.shard_id and any(
                    shard.shard_id == failure.shard_id for shard in self.config.shards
                ) else 10**9,
                failure.code,
                failure.node_id or "",
                failure.message,
            ),
        )
        self.ledger.add_failures(ordered_failures)
        self.ledger.flush_business()
        self.ledger.flush_failures()
        self.ledger.flush_algebra()
        self.ledger.flush_reuse()
        return {
            "format_counts": counts,
            "failures": [failure.to_dict() for failure in ordered_failures],
            "reuse": {
                "invalidated": self.reuse_invalidated,
                "reused": self.reuse_reused,
                "recomputed": self.reuse_recomputed,
            },
            "budget_consumed": self.budget_consumed,
            "conclusions": {
                shard_id: conclusion.to_dict()
                for shard_id, conclusion in sorted(
                    self.conclusions.items(),
                    key=lambda item: self.config.shard(item[0]).index,
                )
            },
        }
