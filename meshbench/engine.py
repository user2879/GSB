"""Application orchestrator tying dispatch, breakers, receipts and the ledger.

The engine keeps an in-memory *cached* projection of receipts for fast views,
while the SQLite ledger remains the truth source.  Rule evolution replaces the
affected cache windows wholesale and marks the old ledger rows stale.  All view
and recomputation calls are timed on the monotonic clock and recorded.
"""

import uuid
from dataclasses import asdict

from .breaker import BreakerRegistry
from .budget import WindowCounters, compute_receipt, window_index_for
from .canonical import content_digest
from .clock import Clock, mono_ms
from .config import MeshConfig
from .dispatcher import Dispatcher
from .ledger import Ledger
from .replay import (canonical_view_digest, prevail_check,
                     replay_after_recovery)
from .snapshot import build_snapshot
from .view import build_view, render_headless


class Engine:
    def __init__(self, config: MeshConfig, ledger: Ledger, clock: Clock = None,
                 logger=None, params_version: int = 1):
        self.config = config
        self.ledger = ledger
        self.clock = clock or Clock()
        self.logger = logger
        self.params_version = int(params_version)
        self.registry = BreakerRegistry()
        self.dispatcher = Dispatcher(config, self.registry, self.clock)
        self.receipts_cache = {}
        self._restore_breakers_from_ledger()
        self._load_receipt_cache()

    # ------------------------------------------------------------------ setup
    def _restore_breakers_from_ledger(self):
        rows = self.ledger.breaker_state_rows(include_stale=False)
        if rows:
            for row in rows:
                instance = self.config.services[row["service_id"]].instance(row["instance_id"])
                breaker = self.registry.get(row["service_id"], instance)
                breaker.restore(
                    row["state"], row["consecutive_failures"], row["opened_at_ms"]
                )

    def _load_receipt_cache(self):
        self.receipts_cache = {
            row["window_id"]: row for row in self.ledger.current_receipts()
            if row["rule_version"] == self.config.version
        }

    def send_request(self, request):
        return self.dispatcher.send_request(request)

    # ------------------------------------------------------------------ batches
    def run_batch(self, requests):
        """Dispatch one batch and atomically persist results + receipts."""
        if not requests:
            return {"batch_id": None, "results": [], "receipts": [], "migrations": []}
        batch_id = f"batch-{self.ledger.cursor_sequence() + 1}-{uuid.uuid4().hex[:10]}"
        self.ledger.mark_pending(requests, self.config.version)

        results = self.dispatcher.send_batch(requests)
        migrations = self.dispatcher.drain_migrations()
        service_by_instance = self._service_by_instance()
        for migration in migrations:
            migration["service_id"] = service_by_instance.get(migration["instance_id"])

        receipts = self._receipts_for_results(results)
        breaker_snapshots = [
            dict(snap, rule_version=self.config.version)
            for snap in self.registry.all_snapshots()
        ]
        self.ledger.append_batch(
            batch_id=batch_id,
            rule_version=self.config.version,
            created_at_ms=self.clock.now_ms(),
            cursor_sequence=max(item.sequence for item in results),
            results=results,
            migrations=migrations,
            receipts=receipts,
            breaker_snapshots=breaker_snapshots,
        )
        for receipt in receipts:
            self.receipts_cache[receipt.window_id] = {
                "window_id": receipt.window_id,
                "rule_version": receipt.rule_version,
                "route_id": receipt.route_id,
                "window_index": receipt.window_index,
                "canonical_bytes": receipt.canonical_bytes,
                "digest": receipt.digest,
            }
        return {"batch_id": batch_id, "results": results,
                "receipts": receipts, "migrations": migrations}

    def _service_by_instance(self):
        mapping = {}
        for service_id, service in self.config.services.items():
            for inst in service.instances:
                mapping[inst.instance_id] = service_id
        return mapping

    def _receipts_for_results(self, results):
        grouped = {}
        for result in results:
            route = self.config.routes[result.route_id]
            window_index = window_index_for(result.started_clock_ms, route)
            key = (result.route_id, window_index)
            counters = grouped.get(key, WindowCounters())
            grouped[key] = counters.add(result.final_outcome)
        receipts = []
        for (route_id, window_index), counters in sorted(grouped.items()):
            route = self.config.routes[route_id]
            receipts.append(compute_receipt(
                route, self.config.version, window_index, counters
            ))
        return receipts

    # ------------------------------------------------------------ rule evolution
    def evolve_rules(self, new_config: MeshConfig):
        """Monotonic version bump with wholesale stale-receipt replacement."""
        old_version = self.config.version
        if new_config.version <= old_version:
            raise ValueError("rule version must be strictly monotonic")
        changed_routes, changed_services = self._changed_scope(new_config)
        affected_windows, affected_instances = self.ledger.mark_stale(
            changed_routes, old_version, new_config.version
        )
        self.config = new_config
        self.dispatcher.update_config(new_config)
        # Old breaker state is stale; current view starts from fresh breakers.
        self.registry = BreakerRegistry()
        self.dispatcher.registry = self.registry
        self._load_receipt_cache()
        report = {
            "old_version": old_version,
            "new_version": new_config.version,
            "changed_routes": changed_routes,
            "changed_services": changed_services,
            "replaced_windows": affected_windows,
            "affected_instances": [
                {"service_id": svc, "instance_id": inst}
                for svc, inst in affected_instances
            ],
        }
        if self.logger:
            self.logger.log("info", "rule_evolved", **report)
        return report

    def _changed_scope(self, new_config):
        old, new = self.config, new_config
        all_services = set(old.services) | set(new.services)
        changed_services = set()
        for sid in all_services:
            if sid not in old.services or sid not in new.services:
                changed_services.add(sid)
            elif content_digest(asdict(old.services[sid])) != \
                    content_digest(asdict(new.services[sid])):
                changed_services.add(sid)
        changed_routes = set()
        for rid in set(old.routes) | set(new.routes):
            if rid not in old.routes or rid not in new.routes:
                changed_routes.add(rid)
            elif content_digest(asdict(old.routes[rid])) != \
                    content_digest(asdict(new.routes[rid])):
                changed_routes.add(rid)
        # A changed service invalidates every route targeting it.
        for route in new.routes.values():
            if route.target_service in changed_services:
                changed_routes.add(route.route_id)
        return sorted(changed_routes), sorted(changed_services)

    # ------------------------------------------------------------------ views
    def view(self):
        started = mono_ms()
        payload = build_view(self.config, self.ledger, self.config.version)
        elapsed = mono_ms() - started
        self.ledger.record_view_timing("view_query", started, elapsed)
        return payload, elapsed

    def render_view(self):
        payload, elapsed = self.view()
        return render_headless(payload), elapsed

    # ------------------------------------------------------- authority / replay
    def authoritative_recompute(self):
        started = mono_ms()
        receipts = prevail_check(self.config, self.ledger)
        elapsed = mono_ms() - started
        self.ledger.record_view_timing("authoritative_recompute", started, elapsed)
        return receipts, elapsed

    def verify_recovery(self):
        started = mono_ms()
        receipts = replay_after_recovery(self.config, self.ledger)
        elapsed = mono_ms() - started
        self.ledger.record_view_timing("recovery_replay", started, elapsed)
        unknown = self.ledger.unknown_requests()
        return receipts, elapsed, unknown

    # ---------------------------------------------------------------- snapshot
    def snapshot(self):
        breaker = self.registry.all_snapshots()
        receipt_digests = {
            window_id: row["digest"] for window_id, row in self.receipts_cache.items()
        }
        return build_snapshot(
            rule_version=self.config.version,
            params_version=self.params_version,
            breaker_snapshots=breaker,
            receipt_digests=receipt_digests,
            cursor_sequence=self.ledger.cursor_sequence(),
            head_digest=self.ledger.head_digest(),
        )

    def cached_view_digest(self):
        from .budget import Receipt  # local import to avoid cycle surprises
        pseudo = {
            window_id: Receipt(window_id, row["route_id"], row["rule_version"],
                               row["window_index"], {}, b"", row["digest"])
            for window_id, row in self.receipts_cache.items()
        }
        return canonical_view_digest(pseudo)

    def config_digest(self):
        return content_digest({
            "version": self.config.version,
            "global_concurrency": self.config.global_concurrency,
            "routes": {rid: route.__dict__ for rid, route in self.config.routes.items()},
        })
