"""Orchestration: evolution analysis, recovery, replay and text reporting."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any, Optional

from .engine import Engine, ExecutionPlan
from .errors import WorkbenchError
from .ledger import Ledger, utc_now_iso
from .spec import PipelineSpec, load_pipeline


class Runner:
    def __init__(self, workspace: str) -> None:
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)
        self.runs_root = os.path.join(self.workspace, "runs")
        os.makedirs(self.runs_root, exist_ok=True)
        self.ledger = Ledger(self.workspace)

    def close(self) -> None:
        self.ledger.close()

    # ---------------------------------------------------------- evolution
    def _baseline(self) -> Optional[dict[str, Any]]:
        row = self.ledger.latest_completed_run()
        if row is None:
            return None
        stage_rows = self.ledger.stage_rows(row["id"])
        return {
            "run": row,
            "fingerprints": {r["stage_id"]: r["stage_fingerprint"]
                             for r in stage_rows},
            "samples_digest": row["samples_digest"],
            "shard_count": len(self.ledger.shard_rows(row["id"])),
        }

    def analyze_evolution(self, spec: PipelineSpec) -> dict[str, Any]:
        baseline = self._baseline()
        stages = list(spec.stages)
        if baseline is None:
            return {
                "baseline_run_id": None,
                "sample_changed": False,
                "changed_stage_ids": [stage.id for stage in stages],
                "affected_stage_ids": [stage.id for stage in stages],
                "unchanged_stage_ids": [],
                "invalidated_shards": 0,
                "reasons": ["cold_start"],
                "stage_changes": [],
            }
        sample_changed = baseline["samples_digest"] != spec.samples_digest
        changes: list[dict[str, str]] = []
        changed_ids: list[str] = []
        for stage in stages:
            old = baseline["fingerprints"].get(stage.id)
            new = spec.stage_fingerprints[stage.id]
            if old != new:
                changed_ids.append(stage.id)
                changes.append({"stage_id": stage.id,
                                "old_fingerprint": old or "",
                                "new_fingerprint": new})
        reasons: list[str] = []
        if sample_changed:
            reasons.append("sample_content_changed")
        if changed_ids:
            reasons.append("stage_spec_changed")
        type_by_id = {stage.id: stage.type for stage in stages}
        if any(type_by_id.get(identifier) == "shard"
               for identifier in changed_ids):
            reasons.append("shard_parameters_changed")
        # Downstream closure over declared upstream edges.
        edges = {stage.id: set(stage.upstream) for stage in stages}
        seed = set(changed_ids)
        if sample_changed:
            seed.update(stage.id for stage in stages)
        affected = set(seed)
        changed = True
        while changed:
            changed = False
            for stage in stages:
                if stage.id in affected:
                    continue
                if edges[stage.id] & affected:
                    affected.add(stage.id)
                    changed = True
        ordered_affected = [stage.id for stage in stages
                            if stage.id in affected]
        unchanged = [stage.id for stage in stages
                     if stage.id not in affected]
        invalidated = baseline["shard_count"] if affected else 0
        return {
            "baseline_run_id": baseline["run"]["id"],
            "sample_changed": sample_changed,
            "changed_stage_ids": changed_ids,
            "affected_stage_ids": ordered_affected,
            "unchanged_stage_ids": unchanged,
            "invalidated_shards": invalidated,
            "reasons": reasons or ["no_change"],
            "stage_changes": changes,
        }

    # -------------------------------------------------------------- run
    def _next_run_dir(self) -> tuple[int, str]:
        existing = [int(name[4:]) for name in os.listdir(self.runs_root)
                    if name.startswith("run_") and name[4:].isdigit()]
        next_id = (max(existing) + 1) if existing else 1
        return next_id, os.path.join(self.runs_root,
                                     f"run_{next_id:06d}")

    def run_pipeline(self, spec_path: str,
                     kill_after_ordinal: Optional[int] = None,
                     expect_recovery: bool = False) -> dict[str, Any]:
        try:
            spec = load_pipeline(spec_path)
        except WorkbenchError as error:
            # Loading faults (E_SPEC/E_DATA/E_PARAM) are still structured
            # ledger events so every business code is replayable.
            self.ledger.record_event(error.code, error.message, None,
                                     error.stage_id or "", error.field_path,
                                     {"path": spec_path, **error.details})
            raise
        plan = ExecutionPlan(spec)
        run_id, run_dir = self._next_run_dir()
        os.makedirs(run_dir, exist_ok=True)
        evolution = self.analyze_evolution(spec)
        self.ledger.create_run(spec, spec_path, evolution)
        engine = Engine(plan, run_dir)
        try:
            observations = engine.execute(kill_after_ordinal)
            for obs in observations:
                self.ledger.commit_stage(run_id, obs)
            if kill_after_ordinal is not None and \
                    kill_after_ordinal < len(spec.stages) - 1:
                # Simulated SIGKILL: process stops *before* later stages are
                # committed.  The ledger therefore cannot claim completion.
                self._mark_remaining_unknown(run_id, spec,
                                             len(observations))
                self.ledger.record_event(
                    "I_KILL",
                    f"forced termination after ordinal "
                    f"{len(observations) - 1}", run_id,
                    details={"committed_ordinals": len(observations),
                             "total_stages": len(spec.stages)})
                return {"run_id": run_id, "status": "KILLED",
                        "evolution": evolution}
            self.ledger.commit_shards(run_id, engine.shards)
            self.ledger.verify_shard_binding(run_id)
            self.ledger.complete_run(run_id)
            return {"run_id": run_id, "status": "COMPLETED",
                    "evolution": evolution}
        except WorkbenchError as error:
            self.ledger.record_event(error.code, error.message, run_id,
                                     error.stage_id or "", error.field_path,
                                     error.details)
            self.ledger.fail_run(run_id, error)
            raise

    def _mark_remaining_unknown(self, run_id: int, spec: PipelineSpec,
                                committed: int) -> None:
        for ordinal, stage in enumerate(spec.stages[committed:],
                                        start=committed):
            self.ledger.mark_unknown_stage(
                run_id, ordinal, stage.id, stage.type, stage.version,
                spec.stage_fingerprints[stage.id], list(stage.upstream),
                dict(stage.params))

    # --------------------------------------------------------- recovery
    def recover(self, spec_path: str) -> dict[str, Any]:
        spec = load_pipeline(spec_path)
        active = self.ledger.active_run()
        if active is None or active["status"] != "RUNNING":
            raise WorkbenchError("E_REPLAY",
                                 "no interrupted RUNNING ledger to recover",
                                 field_path="$.progress")
        if active["spec_fingerprint"] != spec.fingerprint:
            raise WorkbenchError(
                "E_REPLAY",
                "recovery spec fingerprint differs from interrupted run",
                field_path="$.runs.spec_fingerprint")
        run_id = active["id"]
        stage_rows = self.ledger.stage_rows(run_id)
        committed = [row for row in stage_rows if row["status"] == "COMMITTED"]
        unknown = [row for row in stage_rows if row["status"] == "UNKNOWN"]
        rerun_ids = [row["stage_id"] for row in unknown]
        self.ledger.record_event(
            "I_RECOVERY",
            "restart detected; committed stages will be replayed and "
            "compared before continuing", run_id,
            stage_id=committed[0]["stage_id"] if committed else "",
            details={"rerun_stage_ids": rerun_ids})
        # Replay committed stages into an isolated temp directory and compare
        # every digest and column statistic against the ledger.
        self._replay_committed(spec, committed)
        # Re-run the remaining chain into a fresh run directory and finish.
        plan = ExecutionPlan(spec)
        evolution = {
            "baseline_run_id": run_id,
            "sample_changed": False,
            "changed_stage_ids": [],
            "affected_stage_ids": rerun_ids,
            "unchanged_stage_ids": [row["stage_id"] for row in committed],
            "invalidated_shards": 0,
            "reasons": ["forced_termination_recovery"],
            "stage_changes": [],
        }
        new_run_id = self.ledger.create_run(spec, spec_path, evolution,
                                            recovery_of=run_id)
        run_dir = os.path.join(self.runs_root, f"run_{new_run_id:06d}")
        os.makedirs(run_dir, exist_ok=True)
        engine = Engine(plan, run_dir)
        observations = engine.execute()
        for obs in observations:
            self.ledger.commit_stage(new_run_id, obs)
        self.ledger.commit_shards(new_run_id, engine.shards)
        self.ledger.verify_shard_binding(new_run_id)
        self.ledger.complete_run(new_run_id)
        with self.ledger.conn:
            self.ledger.conn.execute(
                "UPDATE runs SET status='SUPERSEDED',completed_at=? WHERE id=?",
                (utc_now_iso(), run_id))
            self.ledger.conn.execute(
                "UPDATE progress SET status='SUPERSEDED',updated_at=? "
                "WHERE run_id=?",
                (utc_now_iso(), run_id))
        return {"recovered_run_id": run_id, "new_run_id": new_run_id,
                "replayed": [row["stage_id"] for row in committed],
                "rerun_stage_ids": rerun_ids}

    # ------------------------------------------------------------ replay
    def _replay_committed(self, spec: PipelineSpec,
                          committed_rows: list[Any]) -> None:
        if not committed_rows:
            return
        with tempfile.TemporaryDirectory(prefix="replay_",
                                         dir=self.workspace) as tmp:
            engine = Engine(ExecutionPlan(spec), tmp)
            observations = engine.execute()
            observed = {obs.stage_id: obs for obs in observations}
            for row in committed_rows:
                stage_id = row["stage_id"]
                obs = observed.get(stage_id)
                if obs is None:
                    raise WorkbenchError(
                        "E_REPLAY", f"stage {stage_id} missing from replay",
                        stage_id, f"$.stages[{row['ordinal']}]")
                if obs.output_digest != row["output_digest"]:
                    raise WorkbenchError(
                        "E_REPLAY",
                        f"first mismatch at stage {stage_id}: output_digest",
                        stage_id,
                        f"$.stages[{row['ordinal']}].output_digest",
                        {"expected": row["output_digest"],
                         "actual": obs.output_digest})
                ledger_stats = json.loads(row["column_stats_json"])
                mismatch_field = _first_stat_mismatch(ledger_stats,
                                                      obs.column_stats)
                if mismatch_field is not None:
                    raise WorkbenchError(
                        "E_REPLAY",
                        f"first mismatch at stage {stage_id}: "
                        f"column field {mismatch_field}",
                        stage_id,
                        f"$.stages[{row['ordinal']}].column_stats."
                        f"{mismatch_field}",
                        {"expected": ledger_stats,
                         "actual": obs.column_stats})

    def replay_run(self, run_id: int) -> dict[str, Any]:
        """Public replay: rerun the stored spec and compare full ledger."""
        row = self.ledger.get_run(run_id)
        snapshot = json.loads(row["spec_snapshot"])
        with tempfile.TemporaryDirectory(prefix="replay_spec_",
                                         dir=self.workspace) as tmp:
            spec_path = os.path.join(tmp, "pipeline.json")
            # Rewrite sample paths to the ledger's spec_dir layout.
            with open(spec_path, "w", encoding="utf-8",
                      newline="\n") as handle:
                json.dump(snapshot, handle, indent=2)
            spec = _reload_with_spec_dir(snapshot, row["spec_dir"], spec_path,
                                         tmp)
            stage_rows = [r for r in self.ledger.stage_rows(run_id)
                          if r["status"] == "COMMITTED"]
            self._replay_committed(spec, stage_rows)
            self.ledger.verify_shard_binding(run_id)
        self.ledger.record_event("I_REPLAY",
                                 f"replay of run {run_id} matched ledger",
                                 run_id)
        return {"run_id": run_id, "status": "MATCH",
                "stages_compared": len(stage_rows)}


def _reload_with_spec_dir(snapshot: dict[str, Any], spec_dir: str,
                          spec_path: str, tmp: str) -> PipelineSpec:
    # The stored snapshot keeps sample paths relative to the original spec
    # directory; copy samples next to the temp spec using those relative
    # paths so the standard loader validates them unchanged.
    for source in snapshot.get("sources", []):
        rel = source["path"]
        if os.path.isabs(rel):
            continue
        source_target = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(source_target), exist_ok=True)
        shutil.copy2(os.path.join(spec_dir, rel), source_target)
    return load_pipeline(spec_path)


def _first_stat_mismatch(expected: dict[str, dict[str, Any]],
                         actual: dict[str, dict[str, Any]]) -> Optional[str]:
    for column in sorted(set(expected) | set(actual)):
        if column not in expected or column not in actual:
            return column
        for field_name in sorted(set(expected[column]) | set(actual[column])):
            if expected[column].get(field_name) != \
                    actual[column].get(field_name):
                return f"{column}.{field_name}"
    return None
