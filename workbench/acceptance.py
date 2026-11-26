from __future__ import annotations

import copy
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .engine import WorkbenchEngine
from .invalidation import plan_evolution
from .recovery import (
    compare_conclusion_views,
    unchanged_byte_records,
    verify_ledger_replay,
)
from .snapshot import SnapshotStore
from .spec import load_spec
from .testkit import write_test_spec
from .timeutil import ScriptedClock


def _clock() -> ScriptedClock:
    return ScriptedClock(start_ms=1_700_000_100_000)


def run_acceptance(output_dir: str | Path = "artifacts/acceptance") -> dict[str, Any]:
    out = Path(output_dir)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "ok": False,
        "output_dir": str(out),
        "checks": checks,
    }

    def check(name: str, detail: dict[str, Any] | None = None) -> None:
        checks.append({"name": name, "ok": True, "detail": detail or {}})

    try:
        spec_path = out / "spec.json"
        spec = write_test_spec(spec_path)
        report["spec_digest"] = spec.digest
        report["topo_order"] = list(spec.topo_order)

        parallel = WorkbenchEngine(
            spec, out / "parallel", clock=_clock(), parallel=True, max_workers=3
        )
        try:
            parallel_result = parallel.run_until_complete()
            assert len(parallel_result["completed"]) == 6
            assert parallel.ledger.active_resource_sum() == 0
            check(
                "parallel_schedule",
                {"completed": parallel_result["completed"], "max_workers": 3},
            )
        finally:
            parallel.close()

        baseline = WorkbenchEngine(spec, out / "baseline", clock=_clock(), parallel=False)
        try:
            baseline.run_until_complete()
            baseline_replay = verify_ledger_replay(baseline.ledger, spec.topo_order)
            baseline_view = baseline.ledger.export_view()
            global_water = json.loads(
                next(
                    row
                    for row in baseline_view["water_records"]
                    if row["record_key"] == "global:global_budget"
                )["record_json"]
            )
            check("baseline_replay", baseline_replay)
            check(
                "water_error_model",
                {
                    "global_bound_fp": global_water["global_aggregate_error_bound_fp"],
                    "version_fp": global_water["version_chain_error_fp"],
                    "dependency_fp": global_water["dependency_error_fp"],
                    "lease_fp": global_water["lease_error_fp"],
                },
            )

            snapshot_store = SnapshotStore(out / "snapshot.json")
            snapshot_store.write_atomic(
                {
                    "snapshot_format_version": 999,
                    "spec_digest": "stale",
                    "view": {},
                    "content_digest": "stale",
                }
            )
            snapshot, replaced, reason = snapshot_store.open_or_replace(
                spec, baseline.ledger
            )
            assert replaced and reason == "format_version_mismatch"
            assert snapshot["spec_digest"] == spec.digest
            check("snapshot_version_replace", {"reason": reason})
        finally:
            baseline.close()

        recovery = WorkbenchEngine(spec, out / "recovery", clock=_clock(), parallel=False)
        try:
            recovery.start_nodes()
            recovery.run_shard_now("s1")
            context = recovery.admit_shard("s3")
            assert context is not None
            before_kill = recovery.ledger.export_view()
            returncode = recovery.orchestrator.kill_node("n1")
            assert returncode == -9, returncode
            unknown = recovery.recover_after_termination(restart_nodes=True)
            assert [row["shard_id"] for row in unknown] == ["s3"]
            after_recovery = recovery.ledger.export_view()
            unchanged_byte_records(before_kill, after_recovery, ["s1"])
            recovery.run_until_complete(recover=False)
            recovered_view = recovery.ledger.export_view()
            recovery_comparison = compare_conclusion_views(
                baseline_view, recovered_view, spec.topo_order
            )
            verify_ledger_replay(recovery.ledger, spec.topo_order)
            check(
                "sigkill_recovery_field_equality",
                {
                    "sigkill_returncode": returncode,
                    "unknown_rerun": ["s3"],
                    "comparison": recovery_comparison,
                },
            )
        finally:
            recovery.close()

        evolution_old = WorkbenchEngine(
            spec, out / "evolution", clock=_clock(), parallel=False
        )
        try:
            evolution_old.run_until_complete()
            attribution = {
                row["shard_id"]: row["node_id"]
                for row in evolution_old.ledger.list_shards()
            }
        finally:
            evolution_old.close()

        new_raw = copy.deepcopy(spec.raw)
        for node in new_raw["nodes"]:
            if node["node_id"] == "n1":
                node["capacity_version"] = 3
        new_spec_path = out / "new-spec.json"
        new_spec_path.write_text(
            json.dumps(new_raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        new_spec = load_spec(new_spec_path)
        plan = plan_evolution(spec, new_spec, old_attribution=attribution)
        assert plan["invalid_shards"] == ["s1", "s3", "s5", "s6"]
        assert plan["unchanged_shards"] == ["s2", "s4"]

        evolved = WorkbenchEngine(
            new_spec, out / "evolution", clock=_clock(), parallel=False
        )
        try:
            now = evolved.now_ms()
            for shard_id in plan["invalid_shards"]:
                evolved.ledger.mark_invalid_for_recompute(
                    shard_id, "E_STALE_CAPACITY_VERSION", now
                )
                evolved.ledger.record_invalidation(
                    new_spec.digest, shard_id, "capacity_version", plan, now
                )
            evolved.ledger.set_meta("spec_digest", new_spec.digest)
            evolved.ledger.set_meta("spec_json", canonical_json(new_spec.raw))
            before_evolution = evolved.ledger.export_view()
            evolved.run_until_complete(recover=False)
            after_evolution = evolved.ledger.export_view()
            unchanged = unchanged_byte_records(
                before_evolution, after_evolution, plan["unchanged_shards"]
            )
            check(
                "incremental_invalidation",
                {
                    "invalid": plan["invalid_shards"],
                    "unchanged": plan["unchanged_shards"],
                    "byte_check": unchanged,
                    "propagation_steps": len(plan["propagation_steps"]),
                },
            )
        finally:
            evolved.close()

        blank = WorkbenchEngine(new_spec, out / "blank-new", clock=_clock(), parallel=False)
        try:
            blank.run_until_complete()
            blank_view = blank.ledger.export_view()
        finally:
            blank.close()

        evolved_compare = WorkbenchEngine(
            new_spec, out / "evolution", clock=_clock(), parallel=False
        )
        try:
            evolved_view = evolved_compare.ledger.export_view()
            evolution_comparison = compare_conclusion_views(
                blank_view, evolved_view, new_spec.topo_order
            )
            check("blank_rebuild_field_equality", evolution_comparison)
        finally:
            evolved_compare.close()

        report["ok"] = True
    except Exception as exc:  # noqa: BLE001 - acceptance should record failure
        report["ok"] = False
        report["error"] = {
            "code": getattr(exc, "code", "E_ACCEPTANCE"),
            "message": str(exc),
            "details": getattr(exc, "details", {}),
            "traceback": traceback.format_exc(),
        }
    finally:
        report_path = out / "acceptance-report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report
