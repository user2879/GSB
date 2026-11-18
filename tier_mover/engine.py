"""Orchestration: derive plans, move files, commit ledger, recover crashes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .errors import CapacityError, IntegrityError
from .faults import Injection
from .ledger import Ledger, ledger_path
from .logging_util import JsonlLogger
from .models import Manifest
from .planner import derive_cycle_plan, derive_initial_plan, plan_to_dict
from .storage import (
    atomic_place,
    cleanup_temp_files,
    ensure_layout,
    object_path,
    scan_tiers,
    sha256_file,
    staging_dir,
    transfer_object,
    verify_object_file,
)


class Engine:
    def __init__(
        self,
        workdir: str | Path,
        manifest: Manifest,
        logger: JsonlLogger,
        injection: Optional[Injection] = None,
    ) -> None:
        self.workdir = Path(workdir)
        self.manifest = manifest
        self.logger = logger
        self.injection = injection or Injection()
        self.tier_ids = [tier.tier_id for tier in sorted(manifest.tiers, key=lambda t: t.tier_id)]
        ensure_layout(self.workdir, self.tier_ids)

    # ------------------------------------------------------------- capacity
    def _capacity_manifest(self) -> Manifest:
        if self.injection.kind != "capacity" or self.injection.capacity_override is None:
            return self.manifest
        override = self.injection.capacity_override
        hot = self.manifest.hot()
        from dataclasses import replace

        new_hot = replace(hot, capacity_bytes=override)
        tiers = [new_hot if tier.tier_id == hot.tier_id else tier for tier in self.manifest.tiers]
        return Manifest(
            policy_version=self.manifest.policy_version,
            tier_param_version=self.manifest.tier_param_version,
            tiers=tiers,
            objects=self.manifest.objects,
            rule=self.manifest.rule,
            injection_sequence=self.manifest.injection_sequence,
        )

    # ----------------------------------------------------------- materialize
    def materialize_staging(self) -> None:
        """Verify staged input files against declared digests (E_INTEGRITY)."""
        staging = staging_dir(self.workdir)
        for obj in sorted(self.manifest.objects, key=lambda o: o.key):
            candidate = staging / obj.digest
            if not candidate.exists():
                raise IntegrityError(
                    f"staged file missing for object {obj.key}",
                    {"key": obj.key, "expected_digest": obj.digest, "location": str(candidate)},
                )
            actual = sha256_file(candidate)
            if actual != obj.digest:
                raise IntegrityError(
                    f"staged file digest mismatch for object {obj.key}",
                    {
                        "key": obj.key,
                        "expected_digest": obj.digest,
                        "actual_digest": actual,
                    },
                )

    # ------------------------------------------------------------- run init
    def run_initial(self, allow_existing: bool = False) -> Dict[str, Any]:
        manifest = self._capacity_manifest()
        self.materialize_staging()
        cleanup_temp_files(self.workdir, self.tier_ids)
        actions = derive_initial_plan(manifest)
        plan = plan_to_dict(manifest, actions, phase="initial")
        self._write_plan(plan, "initial_plan.json")
        self.logger.event("plan_derived", phase="initial", actions=len(actions))

        if allow_existing and ledger_path(self.workdir).exists():
            ledger = Ledger.open(self.workdir)
            ledger.check_versions(manifest)
        else:
            ledger = Ledger.create(self.workdir, manifest)
        cycle_id = ledger.start_cycle("initial", manifest, plan)
        committed = 0
        try:
            for seq, action in enumerate(actions, start=1):
                payload = staging_dir(self.workdir) / action.digest
                atomic_place(self.workdir, action.target_tier, payload, action.digest)
                ledger.record_placement(cycle_id, seq, manifest, action.to_dict())
                committed += 1
                self.logger.event(
                    "object_committed",
                    phase="initial",
                    key=action.key,
                    target_tier=action.target_tier,
                    seq=seq,
                )
            ledger.finish_cycle(cycle_id)
            self._verify_directory_matches_ledger(ledger, manifest)
            conclusions = ledger.export_conclusions()
            self._write_json(conclusions, "initial_conclusions.json")
            return conclusions
        finally:
            ledger.close()

    # ------------------------------------------------------------- run cycle
    def run_cycle(self) -> Dict[str, Any]:
        manifest = self._capacity_manifest()
        self.materialize_staging()
        cleanup_temp_files(self.workdir, self.tier_ids)
        ledger = Ledger.open(self.workdir)
        try:
            ledger.check_versions(manifest)
            ownership = ledger.ownership()
            plan = derive_cycle_plan(manifest, ownership)
            self._write_plan(plan, "cycle_plan.json")
            self.logger.event(
                "plan_derived",
                phase="cycle",
                actions=len(plan["actions"]),
                postponed=len(plan["postponed"]),
            )
            if not plan["actions"]:
                self.logger.event("cycle_noop", reason="no threshold breached")
                conclusions = ledger.export_conclusions()
                self._write_json(conclusions, "cycle_conclusions.json")
                return conclusions

            cycle_id = ledger.start_cycle("cycle", manifest, plan)
            committed = 0
            for action in plan["actions"]:
                ledger.mark_inflight(cycle_id, manifest, action)
                self._crash_during_transfer(committed, action)
                transfer_object(
                    self.workdir,
                    action["key"],
                    action["source_tier"],
                    action["target_tier"],
                    action["digest"],
                )
                ledger.record_migration(cycle_id, action["seq"], manifest, action)
                committed += 1
                self.logger.event(
                    "object_committed",
                    phase="cycle",
                    key=action["key"],
                    source_tier=action["source_tier"],
                    target_tier=action["target_tier"],
                    seq=action["seq"],
                )
            ledger.finish_cycle(cycle_id)
            self._verify_directory_matches_ledger(ledger, manifest)
            conclusions = ledger.export_conclusions()
            self._write_json(conclusions, "cycle_conclusions.json")
            return conclusions
        finally:
            ledger.close()

    # --------------------------------------------------------------- kill
    def _crash_during_transfer(self, committed: int, action: Mapping[str, Any]) -> None:
        """Simulate SIGKILL after the atomic target switch, before source delete.

        The object ends up physically present on both tiers while the ledger
        still marks it ``in_flight`` -- exactly the crash window that recovery
        must resolve by digest re-verification rather than a fabricated report.
        """
        if self.injection.kind != "kill" or committed != self.injection.kill_after_commit:
            return
        from .storage import atomic_place

        source = object_path(
            self.workdir, action["source_tier"], action["digest"]
        )
        atomic_place(
            self.workdir, action["target_tier"], source, action["digest"]
        )
        self.logger.event(
            "forced_termination",
            after_committed=committed,
            unknown_object=action["key"],
            window="target_switched_source_not_deleted",
            exit_code=self.injection.kill_exit_code,
        )
        self.logger.close()
        os._exit(self.injection.kill_exit_code)

    # ------------------------------------------------------------ recovery
    def recover(self) -> Dict[str, Any]:
        cleanup_temp_files(self.workdir, self.tier_ids)
        ledger = Ledger.open(self.workdir)
        try:
            ledger.check_versions(self.manifest)
            status = ledger.recovery_status()
            unknown = list(status["in_flight_objects"])
            resolved: List[Dict[str, Any]] = []
            scanned = scan_tiers(self.workdir, self.tier_ids)
            unfinished = ledger.unfinished_cycle()
            rolled_back: List[Dict[str, Any]] = []
            if unfinished is not None and unknown:
                # The interrupted cycle is not an atomic, replayable unit.
                rolled_back = ledger.rollback_unfinished_cycle(unfinished["cycle_id"])
                self.logger.event(
                    "unfinished_cycle_rolled_back",
                    cycle_id=unfinished["cycle_id"],
                    partial_migrations=len(rolled_back),
                )
            intended_target = {
                action["key"]: action["target_tier"]
                for action in derive_cycle_plan(self.manifest, ledger.ownership())["actions"]
            }
            for key in unknown:
                obj = self.manifest.object_by_key(key)
                present = {
                    tier_id: obj.digest in files
                    for tier_id, files in scanned.items()
                    if obj.digest in files
                }
                entry = self._resolve_unknown(ledger, obj, present, intended_target.get(obj.key))
                resolved.append(entry)
            if not unknown:
                self.logger.event("recovery_clean", committed=status["committed_migrations"])
            reran: List[str] = []
            if unknown or rolled_back:
                ledger.reconcile_tier_usage(self.manifest, ledger.ownership())
                # Re-derive from the now-consistent committed ownership and
                # rerun every still-pending migration in a fresh cycle.
                plan = derive_cycle_plan(self.manifest, ledger.ownership())
                baseline_rows = ledger.tier_rows()
                plan["baseline_usage"] = {
                    row["tier_id"]: row["used_bytes"] for row in baseline_rows
                }
                plan["baseline_ownership"] = {
                    key: tier_id for key, tier_id in ledger.ownership().items()
                }
                if plan["actions"]:
                    self._write_plan(plan, "recovery_plan.json")
                    cycle_id = ledger.start_cycle("recovery", self.manifest, plan)
                    for action in plan["actions"]:
                        transfer_object(
                            self.workdir,
                            action["key"],
                            action["source_tier"],
                            action["target_tier"],
                            action["digest"],
                        )
                        ledger.record_migration(
                            cycle_id, action["seq"], self.manifest, action
                        )
                        reran.append(action["key"])
                        self.logger.event(
                            "recovery_reran_migration",
                            key=action["key"],
                            target_tier=action["target_tier"],
                        )
                    ledger.finish_cycle(cycle_id)
            self._verify_directory_matches_ledger(ledger, self.manifest)
            report = {
                "policy_version": self.manifest.policy_version,
                "tier_param_version": self.manifest.tier_param_version,
                "recovery_status": ledger.recovery_status(),
                "resolved_unknown_objects": resolved,
                "rolled_back_partial_cycle": [
                    {"cycle_id": unfinished["cycle_id"], "migrations": len(rolled_back)}
                ] if unfinished is not None and unknown else [],
                "reran_migrations": reran,
                "note": "only fully committed ledger records were trusted; "
                "in-flight objects were re-verified by digest before rerun",
            }
            (self.workdir / "recovery_report.json").write_text(
                json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            self.logger.event("recovery_report_written", unknown_objects=len(unknown))
            return report
        finally:
            ledger.close()

    def _resolve_unknown(
        self,
        ledger: Ledger,
        obj: Any,
        present: Mapping[str, bool],
        intended_target: Optional[str],
    ) -> Dict[str, Any]:
        hot_id = self.manifest.hot().tier_id
        cold_id = self.manifest.cold().tier_id
        on_hot = hot_id in present
        on_cold = cold_id in present
        if on_hot and on_cold:
            # Crash after target switch but before source delete: keep target.
            target_id = intended_target or hot_id
            verify_object_file(self.workdir, target_id, obj.digest)
            redundant_id = cold_id if target_id == hot_id else hot_id
            object_path(self.workdir, redundant_id, obj.digest).unlink(missing_ok=True)
            decision = "dual_present_kept_target_deleted_source"
        elif on_hot or on_cold:
            target_id = hot_id if on_hot else cold_id
            verify_object_file(self.workdir, target_id, obj.digest)
            decision = "single_present_reverified"
        else:
            # Neither copy survives: cannot invent a result; flag for rerun.
            decision = "missing_requires_rerun"
            target_id = None
        with ledger.conn:
            ledger.conn.execute(
                "UPDATE object_state SET tier_id=?, status=? WHERE key=?",
                (target_id, "committed" if target_id else "unknown", obj.key),
            )
            ledger.conn.execute(
                "UPDATE progress SET cursor_kind=?, unknown_object=? WHERE id=1",
                ("recovered", None if target_id else obj.key),
            )
        self.logger.event(
            "unknown_object_resolved", key=obj.key, target_tier=target_id, decision=decision
        )
        return {
            "key": obj.key,
            "digest": obj.digest,
            "present_on_tiers": sorted(present),
            "target_tier": target_id,
            "decision": decision,
        }

    # ------------------------------------------------------------- verify
    def _verify_directory_matches_ledger(self, ledger: Ledger, manifest: Manifest) -> None:
        scanned = scan_tiers(self.workdir, self.tier_ids)
        actual_usage = {
            tier_id: sum(files.values()) for tier_id, files in scanned.items()
        }
        actual_counts = {tier_id: len(files) for tier_id, files in scanned.items()}
        for row in ledger.tier_rows():
            tier_id = row["tier_id"]
            if row["used_bytes"] != actual_usage.get(tier_id, 0):
                raise IntegrityError(
                    "ledger used_bytes disagrees with directory bytes",
                    {
                        "tier_id": tier_id,
                        "ledger_used_bytes": row["used_bytes"],
                        "directory_used_bytes": actual_usage.get(tier_id, 0),
                        "field": "used_bytes",
                    },
                )
            if row["object_count"] != actual_counts.get(tier_id, 0):
                raise IntegrityError(
                    "ledger object_count disagrees with directory",
                    {
                        "tier_id": tier_id,
                        "ledger_object_count": row["object_count"],
                        "directory_object_count": actual_counts.get(tier_id, 0),
                        "field": "object_count",
                    },
                )
            if row["used_bytes"] > row["capacity_bytes"]:
                raise CapacityError(
                    "tier occupancy exceeds capacity",
                    {
                        "tier_id": tier_id,
                        "used_bytes": row["used_bytes"],
                        "capacity_bytes": row["capacity_bytes"],
                    },
                )
        # Every ledger object must exist on its claimed tier with matching digest.
        for row in ledger.object_rows():
            if row["status"] != "committed":
                continue
            path = object_path(self.workdir, row["tier_id"], row["digest"])
            if not path.exists():
                raise IntegrityError(
                    "committed object missing from claimed tier",
                    {"key": row["key"], "tier_id": row["tier_id"], "field": "tier_id"},
                )
            if sha256_file(path) != row["digest"]:
                raise IntegrityError(
                    "committed object digest mismatch",
                    {"key": row["key"], "tier_id": row["tier_id"], "field": "digest"},
                )

    # -------------------------------------------------------------- helpers
    def _write_plan(self, plan: Mapping[str, Any], name: str) -> None:
        (self.workdir / name).write_text(
            json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def _write_json(self, payload: Mapping[str, Any], name: str) -> None:
        (self.workdir / name).write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
