"""End-to-end orchestration of the local rate-limit / breaker chain.

The engine keeps two cleanly separated layers:

* a *policy* layer (breaker, leases, rate limiting, invalidation) that is
  deterministic and byte-stable;
* a *runtime* layer (real worker subprocesses, sockets, thread pool, monotonic
  clock) that performs the NumPy compute and times waits.

Conclusions depend only on declarations, arrival order and compute digests.
Wall-clock elapsed milliseconds are recorded alongside but excluded from the
canonical content digest, so re-runs and replays stay byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .breaker import evaluate_service
from .clock import MonotonicClock, SystemMonotonicClock
from .hashing import sha256_json
from .invalidate import ChangeSet, InvalidationTrace, propagate
from .lease import LeaseManager, arbitrate_ownership
from .planner import Plan
from .ratelimit import DeterministicRateLimiter
from .scheduler import Scheduler
from .spec import SpecBundle, ServiceSpec
from .transport import ServiceClient, ServiceProcess


@dataclass
class EngineResult:
    decisions: list[dict[str, Any]]
    summaries: dict[str, dict[str, Any]]
    lease_events: list[dict[str, Any]]
    lease_ownership: dict[str, dict[str, Any]]
    waits: list[dict[str, Any]]
    plan: dict[str, Any]
    invalidation: dict[str, Any]
    workers: dict[str, dict[str, Any]]
    content_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "summaries": self.summaries,
            "lease_events": self.lease_events,
            "lease_ownership": self.lease_ownership,
            "waits": self.waits,
            "plan": self.plan,
            "invalidation": self.invalidation,
            "workers": self.workers,
            "content_digest": self.content_digest,
        }


class Engine:
    def __init__(self, bundle: SpecBundle, *,
                 clock: MonotonicClock | None = None,
                 client: ServiceClient | None = None,
                 use_workers: bool = True):
        self.bundle = bundle
        self.clock = clock or SystemMonotonicClock()
        self.client = client or ServiceClient()
        self.use_workers = use_workers
        self.lease_manager = LeaseManager(
            self.clock, bundle.global_limits.lease_ttl_ms)
        self.buckets = {
            s.service_id: DeterministicRateLimiter.create(
                s.service_id, s.rps_limit)
            for s in bundle.services
        }

    # ------------------------------------------------------------------
    def run(self, plan: Plan, *,
            change_sets: list[ChangeSet] | None = None,
            prior_decisions: dict[str, dict[str, Any]] | None = None,
            prior_ownership: dict[str, dict[str, Any]] | None = None
            ) -> EngineResult:
        prior_decisions = prior_decisions or {}
        prior_ownership = prior_ownership or {}
        services = self.bundle.services
        requests = self.bundle.requests

        if plan.mode == "incremental":
            changed = set(plan.changed_services)
            decisions, summaries, workers = self._run_subset(
                [s for s in services if s.service_id in changed])
            # Reuse byte-identical conclusions for unaffected requests.
            for request in requests:
                if request.service_id not in changed:
                    decisions.append(prior_decisions[request.request_id])
        else:
            decisions, summaries, workers = self._run_subset(services)

        decisions = sorted(
            decisions, key=lambda d: (d["service_id"], d["arrival_seq"],
                                      d["request_id"]))

        lease_events = [e.to_dict() for e in self.lease_manager.events]
        ownership = arbitrate_ownership(self.lease_manager.events)
        if plan.mode == "incremental":
            # Unaffected slot ownership is preserved from the prior run.
            merged = dict(prior_ownership)
            merged.update(ownership)
            ownership = {k: merged[k] for k in sorted(merged)}

        trace: InvalidationTrace
        if change_sets:
            lease_records = [e for e in lease_events]
            trace = propagate(services, change_sets, requests, lease_records)
        else:
            trace = InvalidationTrace()

        waits = [w.to_dict() for w in self._scheduler.waits] \
            if getattr(self, "_scheduler", None) else []

        digest = sha256_json({
            "decisions": decisions,
            "lease_ownership": ownership,
            "invalidation": trace.to_dict(),
        })
        return EngineResult(
            decisions=decisions,
            summaries=summaries,
            lease_events=lease_events,
            lease_ownership=ownership,
            waits=waits,
            plan=plan.to_dict(),
            invalidation=trace.to_dict(),
            workers=workers,
            content_digest=digest,
        )

    # ------------------------------------------------------------------
    def _run_subset(self, services: list[ServiceSpec]):
        service_concurrency = {
            s.service_id: max(1, s.concurrency_limit) for s in services
        }
        self._scheduler = Scheduler(
            clock=self.clock,
            max_active_leases=self.bundle.global_limits.max_active_leases,
            max_active_batches=self.bundle.global_limits.max_active_batches,
            service_concurrency=service_concurrency,
        )
        processes: dict[str, ServiceProcess] = {}
        workers_meta: dict[str, dict[str, Any]] = {}
        try:
            if self.use_workers:
                for spec in services:
                    sp = self.client.spawn_and_connect(
                        service_id=spec.service_id, host=spec.host,
                        port=spec.port, config_version=spec.config_version,
                        seed=spec.seed,
                        ttl_ms=self.bundle.global_limits.lease_ttl_ms)
                    processes[spec.service_id] = sp
                    workers_meta[spec.service_id] = {
                        "host": spec.host, "port": spec.port,
                        "config_version": spec.config_version,
                        "transport": "tcp-loopback-subprocess",
                    }

            all_decisions: list[dict[str, Any]] = []
            summaries: dict[str, dict[str, Any]] = {}
            lease_seq = 0
            for spec in services:
                service_requests = [
                    r for r in self.bundle.requests
                    if r.service_id == spec.service_id
                ]
                rate_rejected = self._rate_limiter_pass(spec, service_requests)
                decisions = evaluate_service(
                    spec, service_requests,
                    rate_rejected_ids=frozenset(rate_rejected))

                # Project executed batches onto the real thread-pool + sockets.
                tasks = self._build_tasks(spec, decisions, rate_rejected)
                batch_summaries = self._execute_batches(
                    spec, tasks, processes, lease_seq_counter_start=lease_seq)
                lease_seq += len(tasks)
                summaries.update(batch_summaries)
                all_decisions.extend(d.to_dict() for d in decisions)
            return all_decisions, summaries, workers_meta
        finally:
            self._scheduler.shutdown()
            for sp in processes.values():
                sp.close()

    def _rate_limiter_pass(self, spec, service_requests) -> list[str]:
        limiter = self.buckets[spec.service_id]
        rejected: list[str] = []
        for req in sorted(service_requests,
                          key=lambda r: (r.arrival_seq, r.request_id)):
            if not limiter.try_admit_seq(req.arrival_seq):
                rejected.append(req.request_id)
        return rejected

    def _build_tasks(self, spec, decisions, rate_rejected) -> list[dict[str, Any]]:
        by_batch: dict[str, list[dict[str, Any]]] = {}
        for decision in decisions:
            if decision.outcome == "executed":
                by_batch.setdefault(decision.batch_id, []).append({
                    "request_id": decision.request_id,
                    "arrival_seq": decision.arrival_seq,
                    "failed": decision.failed,
                })
        tasks = []
        limit = max(1, spec.concurrency_limit)
        for order, (batch_id, items) in enumerate(sorted(by_batch.items())):
            tasks.append({
                "_order": order,
                "service_id": spec.service_id,
                "batch_id": batch_id,
                "config_version": spec.config_version,
                "requests": items,
                "slot_index": order % limit,
            })
        return tasks

    def _execute_batches(self, spec, tasks, processes,
                         lease_seq_counter_start) -> dict[str, dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        if not tasks:
            return summaries

        def work(task: dict[str, Any]) -> dict[str, Any]:
            # Deterministic slot derived from the batch's canonical order so
            # ownership is reproducible regardless of thread interleaving.
            slot = task["slot_index"]
            lease = self.lease_manager.grant(
                service_id=spec.service_id, batch_id=task["batch_id"],
                slot_index=slot, quota_version=spec.config_version,
                lease_seq=lease_seq_counter_start + task["_order"])
            try:
                if self.use_workers:
                    sp = processes[spec.service_id]
                    reply = self.client.request(sp, {
                        "kind": "execute",
                        "batch_id": task["batch_id"],
                        "requests": task["requests"],
                    })
                    summary = reply["summary"]
                    elapsed_ms = reply["elapsed_ms"]
                else:
                    from .fixedpoint import deterministic_execute
                    results = []
                    mac_total = 0
                    for item in task["requests"]:
                        r = deterministic_execute(
                            seed=spec.seed, service_id=spec.service_id,
                            batch_id=task["batch_id"],
                            arrival_seq=item["arrival_seq"])
                        results.append({
                            "request_id": item["request_id"],
                            "arrival_seq": item["arrival_seq"],
                            "digest": r.digest, "vector": list(r.vector),
                            "failed": item["failed"],
                        })
                        mac_total += r.mac_count
                    summary = {
                        "service_id": spec.service_id,
                        "batch_id": task["batch_id"],
                        "config_version": spec.config_version,
                        "result_count": len(results),
                        "mac_count": mac_total, "results": results,
                    }
                    elapsed_ms = self.clock.now_ms()
                return {
                    "_order": task["_order"],
                    "batch_id": task["batch_id"],
                    "summary": summary,
                    "elapsed_ms": elapsed_ms,
                    "lease_id": lease.lease_id,
                }
            finally:
                # Normal deterministic release once the batch completes.
                self.lease_manager.release(lease.lease_id)

        results = self._scheduler.run(tasks, work)
        for result in results:
            summaries[result["batch_id"]] = {
                "summary": result["summary"],
                "elapsed_ms": result["elapsed_ms"],
                "lease_id": result["lease_id"],
                "digest": sha256_json(result["summary"]),
            }
        return summaries
