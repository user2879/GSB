"""Deterministic execution engine and conclusion-view replacement."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any

from .compute import apply_stage, table_digest, table_from_dataset
from .errors import E_OUTLIER, E_PARAM, E_STATE, WorkbenchError
from .evolution import (
    compare_specs,
    enrich_stage_outputs,
    enrich_with_shard_invalidation,
    prepared_from_spec_row,
)
from .ledger import Ledger
from .models import BatchObservation, PreparedSpec, RunResult, Table


class Engine:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    @staticmethod
    def validate_execution_parameters(prepared: PreparedSpec, normalize_scale: int | None) -> int:
        scale = prepared.pipeline.fixed_scale if normalize_scale is None else normalize_scale
        if not isinstance(scale, int) or isinstance(scale, bool):
            raise WorkbenchError(
                E_PARAM,
                "normalize_scale must be an integer",
                param="normalize_scale",
                param_value=scale,
                location="session.normalize_scale",
            )
        if not 0 <= scale <= 6:
            raise WorkbenchError(
                E_PARAM,
                "session normalize_scale must be an integer in [0, 6]",
                param="normalize_scale",
                param_value=scale,
                details={"pipeline_fixed_scale": prepared.pipeline.fixed_scale},
                location="session.normalize_scale",
            )
        return scale

    @staticmethod
    def _old_shards_for_view(ledger: Ledger, view_row: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not view_row:
            return []
        view = view_row.get("view") or json.loads(view_row["view_json"])
        shard_stage = next(
            (stage for stage in view.get("stages", []) if stage.get("type") == "shard_pack"),
            None,
        )
        if shard_stage is None:
            return []
        run_id = view.get("run_id") or view_row.get("run_id")
        attempt_row = ledger.latest_stage_run(run_id, shard_stage["id"])
        if attempt_row is None:
            return []
        return ledger.get_shards(run_id, shard_stage["id"], int(attempt_row["attempt"]))

    @staticmethod
    def _base_prepared(ledger: Ledger, view_row: dict[str, Any] | None) -> PreparedSpec | None:
        if not view_row:
            return None
        spec_version = int(view_row.get("spec_version") or view_row["view"]["spec_version"])
        spec_row = ledger.get_spec(spec_version)
        return prepared_from_spec_row(spec_row) if spec_row else None

    def run(
        self,
        prepared: PreparedSpec,
        run_id: str | None = None,
        batch_size: int = 32,
        normalize_scale: int | None = None,
        crash_after_stage: str | None = None,
        command: dict[str, Any] | None = None,
    ) -> RunResult:
        if batch_size <= 0:
            raise WorkbenchError(
                E_PARAM,
                "batch_size must be positive",
                param="batch_size",
                param_value=batch_size,
            )
        scale = self.validate_execution_parameters(prepared, normalize_scale)
        if crash_after_stage is not None and crash_after_stage not in prepared.pipeline.stage_by_id:
            raise WorkbenchError(
                E_PARAM,
                "crash_after_stage does not name a compiled stage",
                param="crash_after_stage",
                param_value=crash_after_stage,
            )
        started = time.perf_counter()
        run_id = run_id or f"run-{uuid.uuid4().hex[:16]}"
        spec_version, _is_new, base_row = self.ledger.allocate_spec(prepared)
        base_view = base_row["view"] if base_row else None
        old_prepared = self._base_prepared(self.ledger, base_row)
        old_shards = self._old_shards_for_view(self.ledger, base_row)
        evolution = (
            compare_specs(old_prepared, prepared, base_row["spec_version"])
            if old_prepared is not None
            else None
        )
        if evolution is not None:
            evolution["new_spec_version"] = spec_version
        command_payload = dict(command or {})
        command_payload.setdefault("batch_size", batch_size)
        command_payload.setdefault("normalize_scale", scale)
        self.ledger.create_run(
            run_id=run_id,
            prepared=prepared,
            spec_version=spec_version,
            batch_size=batch_size,
            normalize_scale=scale,
            base_view=base_view,
            command=command_payload,
            crash_after_stage=crash_after_stage,
        )
        self.ledger.record_event(
            run_id,
            "RUN_STARTED",
            {
                "spec_version": spec_version,
                "fingerprint": prepared.fingerprint,
                "dataset_digest": prepared.dataset_digest,
                "batch_size": batch_size,
                "normalize_scale": scale,
            },
        )

        table = table_from_dataset(prepared.dataset)
        attempts = {stage.id: 1 for stage in prepared.pipeline.stages}
        try:
            table, evolution = self._execute_new_stages(
                run_id=run_id,
                prepared=prepared,
                spec_version=spec_version,
                table=table,
                start_ordinal=0,
                attempts=attempts,
                batch_size=batch_size,
                normalize_scale=scale,
                crash_after_stage=crash_after_stage,
                old_shards=old_shards,
                evolution=evolution,
            )
        except WorkbenchError as error:
            self.ledger.record_error(run_id, error)
            error.details = {**error.details, "ledger_recorded": True}
            raise

        final_digest = table_digest(table)
        view = self.build_view(
            run_id, prepared, spec_version, final_digest, normalize_scale=scale
        )
        if evolution is not None:
            stage_rows = self.ledger.latest_stage_runs(run_id)
            evolution = enrich_stage_outputs(evolution, base_view, stage_rows)
        self.ledger.complete_run(run_id, "COMPLETE", view, evolution)
        self.ledger.record_event(
            run_id,
            "RUN_COMPLETE",
            {"final_data_digest": final_digest, "spec_version": spec_version},
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return self.ledger.run_result(run_id, elapsed_ms)

    def _execute_new_stages(
        self,
        run_id: str,
        prepared: PreparedSpec,
        spec_version: int,
        table: Table,
        start_ordinal: int,
        attempts: dict[str, int],
        batch_size: int,
        normalize_scale: int,
        crash_after_stage: str | None,
        old_shards: list[dict[str, Any]],
        evolution: dict[str, Any] | None,
    ) -> tuple[Table, dict[str, Any] | None]:
        for stage in prepared.pipeline.stages[start_ordinal:]:
            attempt = attempts[stage.id]
            self.ledger.start_stage(
                run_id, stage, attempt, spec_version, prepared.fingerprint
            )
            try:
                def on_batch(observation: BatchObservation) -> None:
                    self.ledger.commit_batch(
                        run_id,
                        stage.id,
                        attempt,
                        observation,
                        spec_version,
                        prepared.fingerprint,
                    )

                table, computation = apply_stage(
                    table,
                    stage,
                    prepared.pipeline.fixed_scale,
                    prepared.pipeline.sharding,
                    batch_size,
                    on_batch=on_batch,
                    normalize_scale=normalize_scale,
                )
            except WorkbenchError as error:
                self.ledger.fail_stage(run_id, stage.id, attempt, error.code)
                raise

            reused_versions: dict[int, int] = {}
            if computation.shards:
                if evolution is not None:
                    evolution, reused_versions = enrich_with_shard_invalidation(
                        evolution, old_shards, computation.shards
                    )
                self.ledger.commit_stage(
                    run_id,
                    computation,
                    attempt,
                    spec_version,
                    prepared.fingerprint,
                    computation.shards,
                    reused_versions,
                )
            else:
                self.ledger.commit_stage(
                    run_id,
                    computation,
                    attempt,
                    spec_version,
                    prepared.fingerprint,
                    [],
                )
            self.ledger.record_event(
                run_id,
                "STAGE_COMMITTED",
                {
                    "stage_id": stage.id,
                    "attempt": attempt,
                    "ordinal": stage.ordinal,
                    "output_digest": computation.output_digest,
                    "row_count": computation.row_count,
                    "batch_count": len(computation.batches),
                },
            )
            if crash_after_stage == stage.id:
                # Deliberate hard termination. No exit report is written: only
                # the already committed stage rows can be inspected on restart.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(75)
        return table, evolution

    def build_view(
        self,
        run_id: str,
        prepared: PreparedSpec,
        spec_version: int,
        final_data_digest: str,
        normalize_scale: int,
    ) -> dict[str, Any]:
        stage_rows = self.ledger.latest_stage_runs(run_id)
        plan = prepared.pipeline.stage_by_id
        stage_payload = []
        for row in stage_rows:
            planned = plan[row["stage_id"]]
            trusted = (
                row["status"] == "COMMITTED"
                and row["stage_version"] == planned.version
                and row["spec_version"] == spec_version
                and row["fingerprint"] == prepared.fingerprint
            )
            stage_payload.append(
                {
                    "ordinal": row["ordinal"],
                    "stage_id": row["stage_id"],
                    "id": row["stage_id"],
                    "type": row["stage_type"],
                    "version": row["stage_version"],
                    "upstream": planned.upstream,
                    "config_fingerprint": planned.config_digest,
                    "status": row["status"],
                    "trusted": trusted,
                    "row_count": row["row_count"],
                    "input_digest": row["input_digest"],
                    "output_digest": row["output_digest"],
                    "attempt": row["attempt"],
                }
            )
        shard_stage = prepared.pipeline.shard_stage
        shard_row = self.ledger.latest_stage_run(run_id, shard_stage.id)
        shard_rows = (
            self.ledger.get_shards(run_id, shard_stage.id, int(shard_row["attempt"]))
            if shard_row is not None
            else []
        )
        shards = [
            {
                "shard_index": row["shard_index"],
                "start_row": row["start_row"],
                "end_row": row["end_row"],
                "row_count": row["row_count"],
                "rows_digest": row["rows_digest"],
                "shard_fingerprint": row["shard_fingerprint"],
                "source_stage_id": row["stage_id"],
                "source_stage_version": row["source_stage_version"],
                "source_output_digest": row["source_output_digest"],
                "spec_version": row["spec_version"],
                "fingerprint": row["fingerprint"],
                "reused_from_spec_version": row["reused_from_spec_version"],
                "stale": row["source_stage_version"] != shard_stage.version
                or row["fingerprint"] != prepared.fingerprint,
            }
            for row in shard_rows
        ]
        shards.sort(key=lambda item: item["shard_index"])
        return {
            "format": "mlpipe.conclusion.v1",
            "run_id": run_id,
            "spec_version": spec_version,
            "fingerprint": prepared.fingerprint,
            "dataset_digest": prepared.dataset_digest,
            "pipeline_digest": prepared.pipeline_digest,
            "fixed_scale": prepared.pipeline.fixed_scale,
            "normalize_scale": normalize_scale,
            "sharding": prepared.pipeline.sharding.as_dict(),
            "row_count": prepared.dataset.row_count,
            "columns": [column.key for column in prepared.dataset.columns],
            "sources": [
                {"id": source.id, "rows": source.rows, "start_row": source.start_row}
                for source in prepared.dataset.sources
            ],
            "stages": sorted(stage_payload, key=lambda item: (item["ordinal"], item["stage_id"])),
            "shards": shards,
            "final_data_digest": final_data_digest,
            "business_codes": [],
        }

    @staticmethod
    def is_outlier_error(error: WorkbenchError) -> bool:
        return error.code == E_OUTLIER

    @staticmethod
    def require_complete_run(run: dict[str, Any]) -> None:
        if run["status"] not in ("COMPLETE", "RECOVERED_COMPLETE"):
            raise WorkbenchError(
                E_STATE,
                f"run {run['run_id']} is not complete",
                details={"status": run["status"]},
            )
