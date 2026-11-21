from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import PipelineConfig
from .engine import PipelineEngine, PipelineRun
from .logging_utils import StructuredLogger
from .spec import LoadedDataset
from .storage import Generation, LedgerStore
from .util import atomic_write_text, canonical_json, digest_json, write_json_file


@dataclass(frozen=True)
class ExecutionOutcome:
    generation: Generation
    report: dict[str, Any]
    run: PipelineRun | None
    already_complete: bool


def _batch_public(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_index": batch["batch_index"],
        "start_cursor": batch["start_cursor"],
        "end_cursor": batch["end_cursor"],
        "business_code": batch["business_code"],
        "summary_digest": batch["summary_digest"],
        "conclusion_digest": batch["conclusion_digest"],
        "aggregate_count": batch["aggregate_count"],
        "ledger_count": batch["ledger_count"],
        "normalized_count": batch["normalized_count"],
        "peak_global": batch["peak_global"],
        "peak_by_service": batch["peak_by_service"],
        "timeline": batch["timeline"],
        "summary": batch["summary"],
    }


def _current_conclusion_digest(store: LedgerStore, generation_id: int) -> str:
    return digest_json(
        {
            "aggregates": store.load_current_aggregates(generation_id),
            "ledger": store.load_current_ledger(generation_id),
            "batches": [
                {
                    "batch_index": batch["batch_index"],
                    "end_cursor": batch["end_cursor"],
                    "conclusion_digest": batch["conclusion_digest"],
                }
                for batch in store.committed_batches(generation_id)
            ],
        }
    )


def export_run_artifacts(
    config: PipelineConfig,
    store: LedgerStore,
    generation_id: int,
    report: dict[str, Any],
) -> dict[str, str]:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    aggregates = store.load_current_aggregates(generation_id)
    ledger = store.load_current_ledger(generation_id)
    canonical = canonical_json({"aggregates": aggregates, "ledger": ledger})
    atomic_write_text(output / "conclusions.canonical.json", canonical)
    write_json_file(output / "run_report.json", report)
    write_json_file(
        output / "timeline.json",
        {
            "generation_id": generation_id,
            "batches": [
                {
                    "batch_index": batch["batch_index"],
                    "peak_global": batch["peak_global"],
                    "peak_by_service": batch["peak_by_service"],
                    "timeline": batch["timeline"],
                }
                for batch in store.committed_batches(generation_id)
            ],
        },
    )
    return {
        "conclusions": str((output / "conclusions.canonical.json").resolve()),
        "run_report": str((output / "run_report.json").resolve()),
        "timeline": str((output / "timeline.json").resolve()),
    }


def execute_pipeline(
    config: PipelineConfig,
    dataset: LoadedDataset,
    *,
    mode: str,
    global_limit: int,
    per_service_limit: int,
    crash_before_commit_on_batch: int | None = None,
    logger: StructuredLogger | None = None,
) -> ExecutionOutcome:
    logger = logger or StructuredLogger(config.log_path)
    with LedgerStore(config.database_path) as store:
        generation = store.open_generation(config, dataset)
        if generation.status == "complete" and generation.is_current:
            report = store.generation_report(generation.generation_id)
            report["business_code"] = "OK"
            report["already_complete"] = True
            report["conclusion_digest"] = _current_conclusion_digest(
                store, generation.generation_id
            )
            report["artifacts"] = export_run_artifacts(
                config, store, generation.generation_id, report
            )
            logger.write(
                "pipeline_run",
                business_code="OK",
                generation_id=generation.generation_id,
                declaration_version=generation.declaration_version,
                already_complete=True,
            )
            return ExecutionOutcome(
                generation=generation,
                report=report,
                run=None,
                already_complete=True,
            )

        cursor, last_batch_index = store.progress(generation.generation_id)
        committed = store.committed_batches(generation.generation_id)
        if committed:
            store.replay_generation(generation, config, dataset)
        state = (
            store.load_state(generation.generation_id, last_batch_index)
            if last_batch_index >= 0
            else {}
        )
        store.crash_before_commit_on_batch = crash_before_commit_on_batch

        engine = PipelineEngine(
            config,
            dataset,
            mode=mode,
            global_limit=global_limit,
            per_service_limit=per_service_limit,
        )

        def before_batch(batch_index: int, start: int, end: int) -> None:
            store.begin_attempt(generation.generation_id, batch_index, start, end)

        def commit_batch(execution) -> None:
            store.commit_batch(generation.generation_id, execution)

        run = engine.run_all(
            start_cursor=cursor,
            state=state,
            batch_start_index=last_batch_index + 1,
            before_batch=before_batch,
            on_batch=commit_batch,
        )
        replacement = store.finalize_generation(generation, config, dataset)
        refreshed = store.generation_for_version(generation.declaration_version)
        report = store.generation_report(refreshed.generation_id)
        report["business_code"] = "OK"
        report["already_complete"] = False
        report["conclusion_digest"] = _current_conclusion_digest(
            store, refreshed.generation_id
        )
        report["replacement"] = replacement
        report["artifacts"] = export_run_artifacts(
            config, store, refreshed.generation_id, report
        )
        logger.write(
            "pipeline_run",
            business_code="OK",
            generation_id=refreshed.generation_id,
            declaration_version=refreshed.declaration_version,
            already_complete=False,
            batch_count=len(store.committed_batches(refreshed.generation_id)),
            conclusion_digest=report["conclusion_digest"],
        )
        return ExecutionOutcome(
            generation=refreshed,
            report=report,
            run=run,
            already_complete=False,
        )


def recover_pipeline(
    config: PipelineConfig,
    dataset: LoadedDataset,
    *,
    declaration_version: int,
    logger: StructuredLogger | None = None,
) -> dict[str, Any]:
    logger = logger or StructuredLogger(config.log_path)
    with LedgerStore(config.database_path) as store:
        generation = store.generation_for_version(declaration_version)
        replay = store.replay_generation(generation, config, dataset)
        unknown = store.unknown_attempts(generation.generation_id)
        report = {
            "business_code": "OK",
            "generation_id": generation.generation_id,
            "declaration_version": generation.declaration_version,
            "seed": generation.seed,
            "status": generation.status,
            "replay": replay.to_public_dict(),
            "unknown_attempts": list(unknown),
            "requires_rerun": bool(unknown),
            "exit_report_available": generation.status == "complete" and not unknown,
            "message": (
                "uncommitted batch attempts are unknown and must be rerun"
                if unknown
                else "all committed batches replayed successfully"
            ),
        }
        logger.write(
            "pipeline_recover",
            business_code="OK",
            generation_id=generation.generation_id,
            unknown_batches=[item["batch_index"] for item in unknown],
            requires_rerun=bool(unknown),
        )
        return report
