"""Workbench engine: orchestrates config, planning, scheduling, ledger,
versioning, invalidation, snapshots and fault injection."""

from __future__ import annotations

import os

from . import faults
from .algorithms import HashRing
from .canon import digest
from .clock import monotonic_ms
from .config_loader import ALGO_RING, load_config
from .errors import BizError, E_CONFIG, E_LEASE, E_SNAPSHOT
from .invalidation import invalidated_subset
from .ledger import Ledger
from .leases import LeaseManager
from .planner import compile_plan
from .scheduler import ProbeScheduler, ProbeTask
from .snapshot import build_snapshot, load_snapshot, save_snapshot
from .structured_log import StructuredLogger
from .versions import VersionRegistry, VersionSet, diff_units


class WorkbenchEngine:
    def __init__(self, workdir: str, global_limit: int = 4, per_db_limit: int = 2,
                 lease_ttl_ms: int = 2000, budget_version: int = 1,
                 clock=monotonic_ms):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.clock = clock
        self.logger = StructuredLogger(os.path.join(workdir, "logs"))
        self.ledger = Ledger(os.path.join(workdir, "ledger.sqlite3"), clock)
        self.lease_manager = LeaseManager(
            lease_ttl_ms, clock=clock, listener=self._on_lease_event
        )
        self.scheduler = ProbeScheduler(global_limit, per_db_limit,
                                        self.lease_manager, clock=clock)
        self.budget_version = budget_version
        self.config = None
        self.versions = None
        self.rings = {}
        self.plan_cache = {}  # unit_id -> {predicate_digest: RoutePlan}
        self.last_diff = {"added": [], "removed": [], "upgraded": []}
        self.last_invalidation = {"invalidated": [], "steps": []}

    # ---------- leases -> ledger ----------
    def _on_lease_event(self, event, lease_dict):
        v = self.versions.current() if self.versions else None
        self.ledger.record_lease(
            lease_dict, v.rule_version if v else 0, v.ring_version if v else 0)
        self.logger.log(event, lease=lease_dict)

    # ---------- config ----------
    def load_config(self, path: str):
        """Validate and install a config; any violation rejects everything."""
        try:
            config = load_config(path)
        except BizError as exc:
            self.logger.log("config_rejected", **exc.to_dict())
            raise
        old = self.config
        self.config = config
        self.rings = {
            u.unit_id: HashRing(u, config.ring_version)
            for u in config.units.values()
            if u.algo_type == ALGO_RING
        }
        new_versions = VersionSet(
            rule_version=config.config_version,
            ring_version=config.ring_version,
            dep_graph_version=config.dep_graph_version,
            budget_version=self.budget_version,
        )
        if self.versions is None:
            self.versions = VersionRegistry(new_versions)
            self.last_diff = diff_units(None, config)
        else:
            cur = self.versions.current()
            changed = (
                new_versions.rule_version != cur.rule_version
                or new_versions.ring_version != cur.ring_version
                or new_versions.dep_graph_version != cur.dep_graph_version
            )
            if changed:
                if new_versions.rule_version != cur.rule_version:
                    self.versions.bump_rule(new_versions.rule_version)
                if new_versions.ring_version != cur.ring_version:
                    self.versions.bump_ring(new_versions.ring_version)
                if new_versions.dep_graph_version != cur.dep_graph_version:
                    self.versions.bump_dep_graph(new_versions.dep_graph_version)
                self._wholesale_replace(old, config)
        self.logger.log(
            "config_loaded",
            versions=self.versions.current().as_dict(),
            diff=self.last_diff,
        )
        return config

    def _wholesale_replace(self, old, new):
        """Mark old-version ledger rows stale and drop cached plans wholesale."""
        counts = self.ledger.mark_stale(
            self.versions.current().rule_version,
            self.versions.current().ring_version,
        )
        self.plan_cache.clear()
        self.last_diff = diff_units(old, new)
        changed = self.last_diff["added"] + self.last_diff["removed"] \
            + self.last_diff["upgraded"]
        invalidated, steps = invalidated_subset(new, changed)
        self.last_invalidation = {"invalidated": invalidated, "steps": steps}
        self.logger.log(
            "wholesale_replace",
            stale_counts=counts,
            diff=self.last_diff,
            invalidated=invalidated,
        )

    # ---------- planning ----------
    def compile(self, unit_id: str, predicate) -> object:
        self._require_config()
        plan = compile_plan(self.config, unit_id, predicate, rings=self.rings)
        self.plan_cache.setdefault(unit_id, {})[plan.predicate_digest] = plan
        return plan

    def cache_digest(self) -> str:
        view = {
            u: {p: plan.plan_digest for p, plan in sorted(plans.items())}
            for u, plans in sorted(self.plan_cache.items())
        }
        return digest(view)

    # ---------- probing ----------
    def probe(self, plans, work_factory=None):
        """Run probe tasks for plans under dual quotas; commit one ledger batch."""
        self._require_config()
        versions = self.versions.current()
        tasks = []
        seq = 0
        for plan in plans:
            for target in plan.targets:
                work = (work_factory or self._default_work)(plan, target)
                tasks.append(
                    ProbeTask(
                        task_id=f"t{seq}",
                        unit_id=plan.unit_id,
                        db=target["db"],
                        table=target["table"],
                        work=work,
                    )
                )
                seq += 1
        records = self.scheduler.run_all(tasks)
        by_task = {t.task_id: t for t in tasks}
        self.ledger.begin_batch()
        try:
            for rec in records:
                task = by_task[rec.task_id]
                plan = next(p for p in plans if p.unit_id == task.unit_id)
                self.ledger.buffer_task(
                    {
                        "unit_id": rec.unit_id,
                        "predicate_digest": plan.predicate_digest,
                        "plan_digest": plan.plan_digest,
                        "targets": [{"db": rec.db, "table": rec.table}],
                        "reasons": [
                            r for t, r in zip(plan.targets, plan.reasons)
                            if t["db"] == rec.db and t["table"] == rec.table
                        ],
                        "lease": rec.lease,
                        "wait_ms": rec.wait_ms,
                        "exec_ms": rec.exec_ms,
                        "status": rec.status,
                    },
                    versions,
                )
            batch_id = self.ledger.commit_batch(versions)
        except Exception:
            self.ledger.abort_batch()
            raise
        self.logger.log(
            "batch_committed",
            batch_id=batch_id,
            tasks=len(records),
            versions=versions.as_dict(),
        )
        return batch_id, records

    @staticmethod
    def _default_work(plan, target):
        def work():
            return {
                "probe": "select 1",
                "unit": plan.unit_id,
                "db": target["db"],
                "table": target["table"],
            }
        return work

    # ---------- invalidation & subset recompute ----------
    def invalidate(self, changed_unit_ids, predicates_by_unit):
        """Recompute only the invalidated subset; verify the rest is untouched."""
        self._require_config()
        invalidated, steps = invalidated_subset(self.config, changed_unit_ids)
        before = {
            u: {p: plan.plan_digest for p, plan in plans.items()}
            for u, plans in self.plan_cache.items()
        }
        recomputed = {}
        for unit_id in invalidated:
            for pred in predicates_by_unit.get(unit_id, []):
                recomputed.setdefault(unit_id, []).append(
                    self.compile(unit_id, pred).plan_digest
                )
        unaffected_ok = all(
            before.get(u, {}) == {
                p: plan.plan_digest for p, plan in self.plan_cache.get(u, {}).items()
            }
            for u in before
            if u not in invalidated
        )
        result = {
            "invalidated": invalidated,
            "steps": steps,
            "recomputed": recomputed,
            "unaffected_byte_identical": unaffected_ok,
        }
        self.logger.log("invalidation", **result)
        return result

    # ---------- snapshots & recovery ----------
    def snapshot_path(self):
        return os.path.join(self.workdir, "snapshot.json")

    def save_snapshot(self):
        self._require_config()
        snap = build_snapshot(
            self.versions.current(),
            [l.as_dict() for l in self.lease_manager.held_leases()],
            self.cache_digest(),
            self.ledger.last_committed_batch(),
        )
        save_snapshot(self.snapshot_path(), snap)
        self.logger.log("snapshot_saved", cursor=snap["payload"]["ledger_cursor"])
        return snap

    def recover(self):
        """Restore from snapshot + committed ledger batches only."""
        self._require_config()
        payload = load_snapshot(self.snapshot_path(), self.versions.current())
        unknown = self.ledger.mark_uncommitted_leases_unknown()
        cursor = self.ledger.last_committed_batch()
        result = {
            "snapshot": payload,
            "recovered_cursor": cursor,
            "unknown_leases": unknown,
            "rerun_required": bool(unknown),
        }
        self.logger.log("recovered", **{k: v for k, v in result.items()
                                        if k != "snapshot"})
        return result

    # ---------- fault injection ----------
    def inject_config_corruption(self, path, mode="invalid_json"):
        evidence = faults.inject_config_corruption(path, mode)
        try:
            load_config(path)
        except BizError as exc:
            self.logger.log("fault_injected", kind="config_corruption",
                            evidence=evidence, **exc.to_dict())
            return exc
        raise AssertionError("corrupted config unexpectedly loaded")

    def inject_lease_version_conflict(self, lease_id):
        evidence = faults.inject_lease_version_conflict(self.lease_manager, lease_id)
        self.logger.log("fault_injected", kind="lease_version_conflict",
                        evidence=evidence)
        return evidence

    def _require_config(self):
        if self.config is None:
            raise BizError(E_CONFIG, "no config loaded", field="$config")

    def shutdown(self):
        self.scheduler.shutdown()
        self.ledger.close()
