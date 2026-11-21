"""Interrupted-run inspection, resumption, and item-by-item replay checks."""

from __future__ import annotations

from typing import Any

from .compute import apply_stage, table_digest, table_from_dataset
from .engine import Engine
from .errors import E_REPLAY, E_STATE, WorkbenchError
from .evolution import (
    compare_specs,
    enrich_stage_outputs,
    prepared_from_spec_row,
)
from .ledger import Ledger
from .models import BatchObservation, PreparedSpec, StageComputation, Table


class RecoveryManager:
    def __init__(self, ledger: Ledger, engine: Engine | None = None) -> None:
        self.ledger = ledger
        self.engine = engine or Engine(ledger)

    def _load_run(self, run_id: str) -> tuple[dict[str, Any], PreparedSpec]:
        run = self.ledger.get_run(run_id)
        if run is None:
            raise WorkbenchError(
                E_STATE,
                f"unknown run id {run_id!r}",
                location="ledger.runs",
            )
        spec_row = self.ledger.get_spec(int(run["spec_version"]))
        if spec_row is None:
            raise WorkbenchError(
                E_STATE,
                "run specification snapshot is missing",
                location="ledger.specs",
                details={"spec_version": run["spec_version"]},
            )
        prepared = prepared_from_spec_row(spec_row)
        if prepared.fingerprint != run["fingerprint"]:
            raise WorkbenchError(
                E_REPLAY,
                "run fingerprint does not match its specification snapshot",
                location="runs.fingerprint",
                details={"expected": run["fingerprint"], "actual": prepared.fingerprint},
            )
        if prepared.dataset_digest != run["dataset_digest"]:
            raise WorkbenchError(
                E_REPLAY,
                "run dataset digest does not match its specification snapshot",
                location="runs.dataset_digest",
                details={
                    "expected": run["dataset_digest"],
                    "actual": prepared.dataset_digest,
                },
            )
        return run, prepared

    def inspect(self, run_id: str) -> dict[str, Any]:
        run, prepared = self._load_run(run_id)
        completed = run["status"] in ("COMPLETE", "RECOVERED_COMPLETE")
        attempts = self.ledger.stage_attempts(run_id)
        recovered: list[dict[str, Any]] = []
        unknown: list[dict[str, Any]] = []
        stage_details: list[dict[str, Any]] = []
        first_unknown: int | None = None

        for stage in prepared.pipeline.stages:
            attempt_rows = attempts.get(stage.id, [])
            latest = attempt_rows[-1] if attempt_rows else None
            if latest is None:
                status = "UNKNOWN"
                attempt = 1
                # Materialise unknown state for stages that never started. This
                # makes the ledger's recovery surface explicit and auditable.
                self.ledger.start_stage(
                    run_id, stage, attempt, int(run["spec_version"]), prepared.fingerprint
                )
                self.ledger.mark_stage_unknown(run_id, stage.id, attempt)
                latest = self.ledger.latest_stage_run(run_id, stage.id)
            else:
                status = latest["status"]
                attempt = int(latest["attempt"])
            trusted = (
                latest is not None
                and latest["status"] == "COMMITTED"
                and int(latest["stage_version"]) == stage.version
                and latest["fingerprint"] == prepared.fingerprint
            )
            if trusted:
                recovered.append(
                    {
                        "stage_id": stage.id,
                        "ordinal": stage.ordinal,
                        "attempt": attempt,
                        "stage_version": stage.version,
                        "output_digest": latest["output_digest"],
                        "row_count": latest["row_count"],
                    }
                )
            else:
                if latest is not None and latest["status"] in ("RUNNING", "FAILED"):
                    self.ledger.mark_stage_unknown(run_id, stage.id, attempt)
                if first_unknown is None:
                    first_unknown = stage.ordinal
                unknown.append(
                    {
                        "stage_id": stage.id,
                        "ordinal": stage.ordinal,
                        "attempt": attempt,
                        "recorded_status": status,
                        "reason": (
                            "partial_stage_not_committed"
                            if latest is not None and latest["status"] != "COMMITTED"
                            else "stage_not_committed"
                        ),
                    }
                )
            stage_details.append(
                {
                    "ordinal": stage.ordinal,
                    "stage_id": stage.id,
                    "recorded_status": status,
                    "trusted": trusted,
                    "attempt": attempt,
                }
            )

        if not completed and unknown:
            self.ledger.update_run_status(run_id, "INTERRUPTED")
        start_ordinal = first_unknown if first_unknown is not None else len(prepared.pipeline.stages)
        rerun_plan = [stage.id for stage in prepared.pipeline.stages[start_ordinal:]]
        report = {
            "format": "mlpipe.recovery-inspection.v1",
            "run_id": run_id,
            "spec_version": int(run["spec_version"]),
            "fingerprint": run["fingerprint"],
            "run_status_at_inspection": run["status"],
            "exit_report_available": bool(completed),
            "exit_report_reason": (
                "run was already complete"
                if completed
                else "forced termination cannot flush a trustworthy exit report"
            ),
            "recovered_stages": recovered,
            "unknown_stages": unknown,
            "rerun_plan": rerun_plan,
            "stage_details": stage_details,
            "recovered_count": len(recovered),
            "unknown_count": len(unknown),
            "resume_from_stage": rerun_plan[0] if rerun_plan else None,
        }
        self.ledger.record_report(run_id, "recovery_inspection", report)
        self.ledger.record_event(
            run_id,
            "RECOVERY_INSPECTION",
            {
                "recovered_count": len(recovered),
                "unknown_count": len(unknown),
                "resume_from_stage": report["resume_from_stage"],
                "exit_report_available": report["exit_report_available"],
            },
        )
        return report

    def _compare_stage(
        self,
        run_id: str,
        computation: StageComputation,
        stage_row: dict[str, Any],
        expected_batches: list[BatchObservation],
    ) -> None:
        checks = (
            ("status", "COMMITTED", stage_row.get("status")),
            ("stage_version", computation.stage.version, stage_row.get("stage_version")),
            ("input_digest", computation.input_digest, stage_row.get("input_digest")),
            ("output_digest", computation.output_digest, stage_row.get("output_digest")),
            ("row_count", computation.row_count, stage_row.get("row_count")),
        )
        for field, expected, actual in checks:
            if expected != actual:
                self._raise_replay(computation.stage.id, field, expected, actual)
        recorded_batches = self.ledger.get_batches(
            run_id,
            computation.stage.id,
            int(stage_row["attempt"]),
            trusted_only=True,
        )
        if len(expected_batches) != len(recorded_batches):
            self._raise_replay(
                computation.stage.id,
                "batch_count",
                len(expected_batches),
                len(recorded_batches),
            )
        for index, (expected, actual) in enumerate(zip(expected_batches, recorded_batches)):
            fields = (
                ("start_row", expected.start_row, actual["start_row"]),
                ("end_row", expected.end_row, actual["end_row"]),
                ("row_count", expected.row_count, actual["row_count"]),
                ("input_digest", expected.input_digest, actual["input_digest"]),
                ("output_digest", expected.output_digest, actual["output_digest"]),
            )
            for field, exp, act in fields:
                if exp != act:
                    self._raise_replay(
                        computation.stage.id,
                        f"batches[{index}].{field}",
                        exp,
                        act,
                    )

    def _compare_shards(
        self,
        prepared: PreparedSpec,
        shard_row: dict[str, Any],
        expected: list[dict[str, Any]],
        run_id: str,
    ) -> None:
        recorded = self.ledger.get_shards(
            run_id,
            prepared.pipeline.shard_stage.id,
            int(shard_row["attempt"]),
            trusted_only=True,
        )
        stage_id = prepared.pipeline.shard_stage.id
        if len(expected) != len(recorded):
            self._raise_replay(stage_id, "shard_count", len(expected), len(recorded))
        for index, (exp, act) in enumerate(zip(expected, recorded)):
            fields = (
                ("shard_index", exp["shard_index"], act["shard_index"]),
                ("start_row", exp["start_row"], act["start_row"]),
                ("end_row", exp["end_row"], act["end_row"]),
                ("row_count", exp["row_count"], act["row_count"]),
                ("rows_digest", exp["rows_digest"], act["rows_digest"]),
                (
                    "shard_fingerprint",
                    exp["shard_fingerprint"],
                    act["shard_fingerprint"],
                ),
                (
                    "source_output_digest",
                    exp["source_output_digest"],
                    act["source_output_digest"],
                ),
                (
                    "source_stage_version",
                    exp["source_stage_version"],
                    act["source_stage_version"],
                ),
            )
            for field, exp_value, act_value in fields:
                if exp_value != act_value:
                    self._raise_replay(
                        stage_id,
                        f"shards[{index}].{field}",
                        exp_value,
                        act_value,
                    )

    def _raise_replay(self, stage_id: str, field: str, expected: Any, actual: Any) -> None:
        raise WorkbenchError(
            E_REPLAY,
            f"replay diverged at stage {stage_id!r}, field {field!r}",
            stage_id=stage_id,
            param=field,
            location=f"ledger.stage[{stage_id}].{field}",
            details={
                "first_stage_id": stage_id,
                "first_field": field,
                "expected": expected,
                "actual": actual,
            },
        )

    def _recompute(
        self,
        run_id: str,
        prepared: PreparedSpec,
        stop_ordinal: int | None = None,
        compare: bool = True,
    ) -> tuple[Table, list[tuple[StageComputation, StageComputation]], list[dict[str, Any]]]:
        run = self.ledger.get_run(run_id)
        assert run is not None
        batch_size = int(run["batch_size"])
        normalize_scale = int(run["normalize_scale"])
        table = table_from_dataset(prepared.dataset)
        computations: list[StageComputation] = []
        shards: list[dict[str, Any]] = []
        for stage in prepared.pipeline.stages:
            if stop_ordinal is not None and stage.ordinal >= stop_ordinal:
                break
            table, computation = apply_stage(
                table,
                stage,
                prepared.pipeline.fixed_scale,
                prepared.pipeline.sharding,
                batch_size,
                normalize_scale=normalize_scale,
            )
            computations.append(computation)
            stage_row = self.ledger.latest_stage_run(run_id, stage.id)
            if stage_row is None:
                self._raise_replay(stage.id, "stage_run", "COMMITTED", "MISSING")
            if compare:
                self._compare_stage(run_id, computation, stage_row, computation.batches)
            if stage.type == "shard_pack":
                shards = computation.shards
                if compare:
                    self._compare_shards(prepared, stage_row, shards, run_id)
        return table, [], shards

    def replay(self, run_id: str) -> dict[str, Any]:
        run, prepared = self._load_run(run_id)
        if run["status"] not in ("COMPLETE", "RECOVERED_COMPLETE", "RUNNING", "INTERRUPTED"):
            raise WorkbenchError(
                E_STATE,
                f"run status {run['status']!r} is not replayable",
                details={"status": run["status"]},
            )
        try:
            table, _, _ = self._recompute(run_id, prepared, compare=True)
            final_digest = table_digest(table)
            if final_digest != run["final_data_digest"]:
                self._raise_replay(
                    "<final>",
                    "final_data_digest",
                    run["final_data_digest"],
                    final_digest,
                )
        except WorkbenchError as error:
            self.ledger.record_error(run_id, error)
            error.details = {**error.details, "ledger_recorded": True}
            report = {
                "format": "mlpipe.replay-report.v1",
                "run_id": run_id,
                "status": "FAILED",
                "code": error.code,
                "error": error.to_dict(),
            }
            self.ledger.record_report(run_id, "replay", report)
            raise
        report = {
            "format": "mlpipe.replay-report.v1",
            "run_id": run_id,
            "status": "OK",
            "spec_version": int(run["spec_version"]),
            "fingerprint": run["fingerprint"],
            "stages_verified": len(prepared.pipeline.stages),
            "shards_verified": len(
                self.ledger.get_shards(
                    run_id,
                    prepared.pipeline.shard_stage.id,
                    int(self.ledger.latest_stage_run(run_id, prepared.pipeline.shard_stage.id)["attempt"]),
                )
            ),
            "final_data_digest": run["final_data_digest"],
        }
        self.ledger.record_report(run_id, "replay", report)
        self.ledger.record_event(
            run_id,
            "REPLAY_OK",
            {
                "stages_verified": report["stages_verified"],
                "shards_verified": report["shards_verified"],
                "final_data_digest": report["final_data_digest"],
            },
        )
        return report

    def _base_context(
        self, run: dict[str, Any]
    ) -> tuple[PreparedSpec | None, dict[str, Any] | None, list[dict[str, Any]]]:
        if not run.get("base_run_id") or run.get("base_spec_version") is None:
            return None, None, []
        base_spec = self.ledger.get_spec(int(run["base_spec_version"]))
        base_prepared = prepared_from_spec_row(base_spec) if base_spec else None
        base_conclusion = self.ledger.conclusion_for_run(run["base_run_id"])
        if base_conclusion is None and run.get("base_fingerprint"):
            # A history row is normally present. Falling back to the missing
            # case leaves the old view unknown rather than inventing one.
            base_conclusion = None
        old_shards = self.engine._old_shards_for_view(self.ledger, base_conclusion)
        return base_prepared, base_conclusion, old_shards

    def recover(self, run_id: str, continue_run: bool = True) -> dict[str, Any]:
        run_before, prepared = self._load_run(run_id)
        if run_before["status"] in ("COMPLETE", "RECOVERED_COMPLETE"):
            replay = self.replay(run_id)
            inspection = self.inspect(run_id)
            return {
                "format": "mlpipe.recovery-report.v1",
                "run_id": run_id,
                "resumed": False,
                "inspection": inspection,
                "replay": replay,
                "status": run_before["status"],
            }

        inspection = self.inspect(run_id)
        run, prepared = self._load_run(run_id)
        stages = prepared.pipeline.stages
        start_ordinal = (
            int(inspection["unknown_stages"][0]["ordinal"])
            if inspection["unknown_stages"]
            else len(stages)
        )
        try:
            table, _, _ = self._recompute(
                run_id, prepared, stop_ordinal=start_ordinal, compare=True
            )
            if start_ordinal < len(stages):
                attempts = {
                    stage.id: max(
                        [int(row["attempt"]) for row in self.ledger.stage_attempts(run_id).get(stage.id, [])]
                        or [0]
                    )
                    + 1
                    for stage in stages[start_ordinal:]
                }
                base_prepared, base_view_row, old_shards = self._base_context(run)
                base_view = base_view_row["view"] if base_view_row else None
                evolution = (
                    compare_specs(base_prepared, prepared, int(run["base_spec_version"]))
                    if base_prepared is not None
                    else None
                )
                if evolution is not None:
                    evolution["new_spec_version"] = int(run["spec_version"])
                table, evolution = self.engine._execute_new_stages(
                    run_id=run_id,
                    prepared=prepared,
                    spec_version=int(run["spec_version"]),
                    table=table.clone(),
                    start_ordinal=start_ordinal,
                    attempts=attempts,
                    batch_size=int(run["batch_size"]),
                    normalize_scale=int(run["normalize_scale"]),
                    crash_after_stage=None,
                    old_shards=old_shards,
                    evolution=evolution,
                )
                final_digest = table_digest(table)
                view = self.engine.build_view(
                    run_id,
                    prepared,
                    int(run["spec_version"]),
                    final_digest,
                    int(run["normalize_scale"]),
                )
                if evolution is not None:
                    stage_rows = self.ledger.latest_stage_runs(run_id)
                    evolution = enrich_stage_outputs(evolution, base_view, stage_rows)
                self.ledger.complete_run(run_id, "RECOVERED_COMPLETE", view, evolution)
            else:
                # All stages were committed before the abrupt termination; only
                # the final run/view commit was missing.
                final_digest = table_digest(table)
                base_prepared, base_view_row, old_shards = self._base_context(run)
                base_view = base_view_row["view"] if base_view_row else None
                evolution = (
                    compare_specs(base_prepared, prepared, int(run["base_spec_version"]))
                    if base_prepared is not None
                    else None
                )
                if evolution is not None:
                    evolution["new_spec_version"] = int(run["spec_version"])
                    evolution = enrich_stage_outputs(
                        evolution,
                        base_view,
                        self.ledger.latest_stage_runs(run_id),
                    )
                view = self.engine.build_view(
                    run_id,
                    prepared,
                    int(run["spec_version"]),
                    final_digest,
                    int(run["normalize_scale"]),
                )
                self.ledger.complete_run(run_id, "RECOVERED_COMPLETE", view, evolution)
            self.ledger.record_event(
                run_id,
                "RECOVERY_COMPLETED",
                {"resume_from_stage": inspection.get("resume_from_stage")},
            )
            replay = self.replay(run_id)
        except WorkbenchError as error:
            if not error.details.get("ledger_recorded"):
                self.ledger.record_error(run_id, error)
            report = {
                "format": "mlpipe.recovery-report.v1",
                "run_id": run_id,
                "resumed": False,
                "inspection": inspection,
                "replay": None,
                "status": "FAILED",
                "error": error.to_dict(),
            }
            self.ledger.record_report(run_id, "recovery", report)
            raise

        final_report = {
            "format": "mlpipe.recovery-report.v1",
            "run_id": run_id,
            "resumed": bool(inspection["unknown_stages"]),
            "inspection": inspection,
            "replay": replay,
            "status": "RECOVERED_COMPLETE",
            "recovered_stages": [stage["stage_id"] for stage in inspection["recovered_stages"]],
            "unknown_stages": [stage["stage_id"] for stage in inspection["unknown_stages"]],
            "rerun_plan": inspection["rerun_plan"],
        }
        self.ledger.record_report(run_id, "recovery", final_report)
        return final_report
