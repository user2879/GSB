from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import atomic_write_bytes, atomic_write_json, canonical_json, sha256_obj
from .dataset import apply_skew_amplification, load_dataset
from .errors import PipelineError, replay_error
from .execution import (
    build_final_result,
    make_plan,
    merge_aggregate_maps,
    result_bytes,
    result_digest,
)
from .ledger import Ledger
from .model import Dataset, RunConfig
from .replay import group_types_for_dataset, replay_ledger_run
from .scheduler import ScheduleJournal, run_scheduler
from .snapshot import (
    build_snapshot,
    compare_snapshot,
    load_snapshot,
    snapshot_config,
    write_snapshot,
)
from .structured_log import StructuredLogger


def prepare_dataset(data_path: str | Path, config: RunConfig) -> Dataset:
    dataset = load_dataset(data_path)
    if config.inject_skew_amplify > 1:
        dataset = apply_skew_amplification(
            dataset,
            config.inject_skew_amplify,
            config.skew_safety_limit,
        )
    return dataset


def deterministic_run_id(dataset: Dataset, config: RunConfig) -> str:
    fingerprint = sha256_obj(
        {
            "run_format": 1,
            "query_id": dataset.query.query_id,
            "data_digest": dataset.data_digest,
            "config_digest": config.config_digest,
        }
    )
    return "run-" + fingerprint[:20]


def _stable_task_status_counts(journal: ScheduleJournal) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in journal.records.values():
        counts[record.status] = counts.get(record.status, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _trace_payload(
    *,
    run_id: str,
    dataset: Dataset,
    config: RunConfig,
    plan_digest: str,
    plan_payload: list[dict[str, Any]],
    journal: ScheduleJournal,
    final_digest: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "query_id": dataset.query.query_id,
        "data_digest": dataset.data_digest,
        "plan_digest": plan_digest,
        "budget_version": config.budget_version,
        "split_threshold": config.split_threshold,
        "split_parts": config.split_parts,
        "split_strategy": config.split_strategy,
        "parallelism": config.parallelism,
        "memory_budget_bytes": config.memory_budget_bytes,
        "effective_memory_budget_bytes": config.effective_memory_budget(),
        "injection": dataset.injection,
        "query_plan": plan_payload,
        "tasks": [record.as_dict() for record in journal.records_in_order()],
        "skew_decisions": [
            decision.as_dict()
            for decision in sorted(journal.skew_decisions, key=lambda item: item.task_id)
        ],
        "budget_events": [event.as_dict() for event in journal.budget_events],
        "peak_memory_bytes": journal.peak_memory_bytes,
        "peak_active_tasks": journal.peak_active_tasks,
        "task_count": len(journal.records),
        "split_count": journal.split_count,
        "split_children_count": journal.split_children_count,
        "final_result_digest": final_digest,
    }


def run_pipeline(
    data_path: str | Path,
    config: RunConfig,
    *,
    out_dir: str | Path,
    ledger_path: str | Path,
    snapshot_dir: str | Path,
    log_path: str | Path | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    kind: str = "run",
) -> dict[str, Any]:
    logger = StructuredLogger(log_path)
    output_dir = Path(out_dir)
    ledger: Ledger | None = None
    journal = ScheduleJournal()
    dataset: Dataset | None = None
    effective_run_id: str | None = None
    plan_payload: list[dict[str, Any]] = []
    session_started = False
    try:
        config.validate()
        dataset = prepare_dataset(data_path, config)
        group_types = group_types_for_dataset(dataset, dataset.query)
        plan = make_plan(dataset, config.row_group_size)
        plan_payload = plan.payload()
        effective_run_id = run_id or deterministic_run_id(dataset, config)
        output_dir.mkdir(parents=True, exist_ok=True)
        ledger = Ledger(ledger_path)
        ledger.reset_run(effective_run_id)
        ledger.begin_run(
            run_id=effective_run_id,
            parent_run_id=parent_run_id,
            kind=kind,
            status="running",
            dataset=dataset,
            config=config,
            plan_payload=plan_payload,
        )
        session_started = True
        logger.emit(
            "run_started",
            code="OK",
            details={
                "run_id": effective_run_id,
                "query_id": dataset.query.query_id,
                "data_digest": dataset.data_digest,
                "plan_digest": plan.plan_digest,
                "budget_version": config.budget_version,
                "split_threshold": config.split_threshold,
                "parallelism": config.parallelism,
                "effective_memory_budget_bytes": config.effective_memory_budget(),
            },
        )

        def checkpoint(batch_no: int, current: ScheduleJournal) -> None:
            assert ledger is not None
            ledger.checkpoint(
                effective_run_id,
                batch_no,
                current,
                dataset.query,
                group_types,
            )

        schedule = run_scheduler(
            dataset,
            dataset.query,
            config,
            plan.tasks,
            checkpoint_callback=checkpoint,
            journal=journal,
        )
        merged = merge_aggregate_maps(
            schedule.included_outcomes,
            dataset.query,
            group_types,
        )
        final_result = build_final_result(dataset.query, group_types, merged)
        final_json = canonical_json(final_result)
        final_digest = result_digest(final_result)
        final_bytes = result_bytes(final_result)

        ledger.checkpoint(
            effective_run_id,
            journal.completed_markers,
            journal,
            dataset.query,
            group_types,
        )
        result_file = output_dir / "final_result.json"
        trace_file = output_dir / "trace.json"
        summary_file = output_dir / "summary.json"
        atomic_write_bytes(result_file, final_bytes)
        trace = _trace_payload(
            run_id=effective_run_id,
            dataset=dataset,
            config=config,
            plan_digest=plan.plan_digest,
            plan_payload=plan_payload,
            journal=journal,
            final_digest=final_digest,
        )
        atomic_write_json(trace_file, trace, newline=True)
        ledger.finish_run(
            effective_run_id,
            result_json=final_json,
            result_digest=final_digest,
        )
        ledger_digest = ledger.semantic_digest(effective_run_id)
        snapshot = build_snapshot(
            run_id=effective_run_id,
            dataset=dataset,
            config=config,
            result_json=final_json,
            result_digest=final_digest,
            ledger_digest=ledger_digest,
            task_count=len(journal.records),
            split_count=journal.split_count,
            data_path=str(Path(data_path).resolve()),
        )
        written_snapshot = write_snapshot(snapshot_dir, snapshot)
        ledger.register_snapshot(snapshot, written_snapshot)
        ledger.finish_run(
            effective_run_id,
            result_json=final_json,
            result_digest=final_digest,
            snapshot_id=str(snapshot["snapshot_id"]),
            snapshot_path=str(written_snapshot),
        )
        summary: dict[str, Any] = {
            "code": "OK",
            "mode": kind,
            "run_id": effective_run_id,
            "parent_run_id": parent_run_id,
            "query_id": dataset.query.query_id,
            "data_digest": dataset.data_digest,
            "plan_digest": plan.plan_digest,
            "result_digest": final_digest,
            "result_file": str(result_file),
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_path": str(written_snapshot),
            "ledger_path": str(Path(ledger_path)),
            "ledger_digest": ledger_digest,
            "budget_version": config.budget_version,
            "split_threshold": config.split_threshold,
            "split_parts": config.split_parts,
            "parallelism": config.parallelism,
            "memory_budget_bytes": config.memory_budget_bytes,
            "effective_memory_budget_bytes": config.effective_memory_budget(),
            "base_task_count": len(plan.tasks),
            "task_count": len(journal.records),
            "split_count": journal.split_count,
            "split_children_count": journal.split_children_count,
            "included_task_count": len(schedule.included_outcomes),
            "task_status_counts": _stable_task_status_counts(journal),
        }
        atomic_write_json(summary_file, summary, newline=True)
        logger.emit(
            "run_completed",
            code="OK",
            details={
                "run_id": effective_run_id,
                "result_digest": final_digest,
                "task_count": summary["task_count"],
                "split_count": summary["split_count"],
                "snapshot_id": snapshot["snapshot_id"],
            },
        )
        return {
            **summary,
            "result_json": final_json,
            "summary_file": str(summary_file),
            "trace_file": str(trace_file),
        }
    except PipelineError as error:
        logger.error(error)
        setattr(error, "_logged", True)
        if ledger is not None and session_started and effective_run_id is not None and dataset is not None:
            try:
                group_types = group_types_for_dataset(dataset, dataset.query)
                ledger.checkpoint(
                    effective_run_id,
                    journal.completed_markers,
                    journal,
                    dataset.query,
                    group_types,
                )
            except Exception:
                pass
            try:
                ledger.fail_run(effective_run_id, error.as_dict())
            except Exception:
                pass
            try:
                error_trace = _trace_payload(
                    run_id=effective_run_id,
                    dataset=dataset,
                    config=config,
                    plan_digest=make_plan(dataset, config.row_group_size).plan_digest,
                    plan_payload=plan_payload,
                    journal=journal,
                    final_digest=None,
                )
                error_trace["code"] = error.code
                error_trace["error"] = error.as_dict()
                atomic_write_json(output_dir / "error_trace.json", error_trace, newline=True)
            except Exception:
                pass
        raise
    except Exception as error:
        wrapped = PipelineError(
            "E_DATA",
            f"unexpected runtime failure: {error}",
            location="runtime",
            details={"exception_type": type(error).__name__},
        )
        logger.error(wrapped)
        if ledger is not None and session_started and effective_run_id is not None:
            try:
                ledger.fail_run(effective_run_id, wrapped.as_dict())
            except Exception:
                pass
        raise wrapped from error
    finally:
        if ledger is not None:
            ledger.close()


def replay_snapshot(
    snapshot_file: str | Path,
    *,
    data_path: str | Path | None,
    ledger_path: str | Path,
    out_dir: str | Path | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    logger = StructuredLogger(log_path)
    ledger: Ledger | None = None
    try:
        snapshot = load_snapshot(snapshot_file)
        config = snapshot_config(snapshot)
        current_data_path = data_path or snapshot.get("data_path")
        if not current_data_path:
            raise replay_error(
                "snapshot does not record a data path and --data was not supplied",
                task_id="<snapshot>",
                field_name="data_path",
            )
        dataset = prepare_dataset(current_data_path, config)
        comparison = compare_snapshot(snapshot, dataset, config)
        if not comparison["compatible"]:
            reason = comparison["reasons"][0]
            raise replay_error(
                "snapshot version does not match current data or recorded parameters",
                task_id="<snapshot>",
                field_name=reason,
                details=comparison,
            )
        ledger = Ledger(ledger_path)
        replay = replay_ledger_run(
            ledger=ledger,
            snapshot=snapshot,
            dataset=dataset,
            config=config,
        )
        output_dir = None if out_dir is None else Path(out_dir)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(
                output_dir / "final_result.json",
                replay.result_json.encode("utf-8") + b"\n",
            )
            atomic_write_json(
                output_dir / "replay_summary.json",
                {
                    "code": "OK",
                    "mode": "replay",
                    "run_id": replay.run_id,
                    "snapshot_id": replay.snapshot_id,
                    "result_digest": replay.result_digest,
                    "replayed_tasks": replay.replayed_tasks,
                    "task_count": replay.task_count,
                    "split_count": replay.split_count,
                    "ledger_digest": replay.ledger_digest,
                },
                newline=True,
            )
        logger.emit(
            "replay_completed",
            code="OK",
            details={
                "run_id": replay.run_id,
                "snapshot_id": replay.snapshot_id,
                "replayed_tasks": replay.replayed_tasks,
                "result_digest": replay.result_digest,
            },
        )
        return {
            "code": "OK",
            "mode": "replay",
            "run_id": replay.run_id,
            "snapshot_id": replay.snapshot_id,
            "query_id": dataset.query.query_id,
            "result_digest": replay.result_digest,
            "replayed_tasks": replay.replayed_tasks,
            "task_count": replay.task_count,
            "split_count": replay.split_count,
            "ledger_digest": replay.ledger_digest,
            "snapshot_path": str(Path(snapshot_file)),
            "ledger_path": str(Path(ledger_path)),
        }
    except PipelineError as error:
        logger.error(error)
        setattr(error, "_logged", True)
        raise
    except Exception as error:
        wrapped = replay_error(
            f"unexpected replay failure: {error}",
            task_id="<replay>",
            field_name="runtime",
            details={"exception_type": type(error).__name__},
        )
        logger.error(wrapped)
        raise wrapped from error
    finally:
        if ledger is not None:
            ledger.close()


def resume_snapshot(
    snapshot_file: str | Path,
    *,
    data_path: str | Path | None,
    ledger_path: str | Path,
    snapshot_dir: str | Path,
    out_dir: str | Path,
    log_path: str | Path | None = None,
    config_override: RunConfig | None = None,
) -> dict[str, Any]:
    snapshot = load_snapshot(snapshot_file)
    config = config_override or snapshot_config(snapshot)
    current_data_path = data_path or snapshot.get("data_path")
    if not current_data_path:
        raise replay_error(
            "snapshot does not record a data path and --data was not supplied",
            task_id="<snapshot>",
            field_name="data_path",
        )
    dataset = prepare_dataset(current_data_path, config)
    comparison = compare_snapshot(snapshot, dataset, config)
    if comparison["compatible"]:
        replayed = replay_snapshot(
            snapshot_file,
            data_path=current_data_path,
            ledger_path=ledger_path,
            out_dir=out_dir,
            log_path=log_path,
        )
        return {
            **replayed,
            "mode": "resume",
            "restored": True,
            "replaced": False,
            "replacement_reasons": [],
        }

    logger = StructuredLogger(log_path)
    replaced = run_pipeline(
        current_data_path,
        config,
        out_dir=out_dir,
        ledger_path=ledger_path,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
        parent_run_id=str(snapshot["run_id"]),
        kind="replacement",
    )
    with Ledger(ledger_path) as replacement_ledger:
        replacement_ledger.record_snapshot_replacement(
            old_run_id=str(snapshot["run_id"]),
            new_run_id=str(replaced["run_id"]),
            old_snapshot_id=str(snapshot["snapshot_id"]),
            reason="snapshot version mismatch; all shard and aggregate state replaced",
            details=comparison,
        )
    logger.emit(
        "snapshot_replaced",
        code="OK",
        details={
            "old_run_id": snapshot["run_id"],
            "new_run_id": replaced["run_id"],
            "old_snapshot_id": snapshot["snapshot_id"],
            "reasons": comparison["reasons"],
        },
    )
    result = {
        **replaced,
        "mode": "replacement",
        "restored": False,
        "replaced": True,
        "replacement_reasons": comparison["reasons"],
        "replaced_snapshot_id": snapshot["snapshot_id"],
        "replaced_run_id": snapshot["run_id"],
    }
    path = Path(replaced["summary_file"])
    stable_summary = {key: value for key, value in result.items() if key != "result_json"}
    atomic_write_json(path, stable_summary, newline=True)
    return result
