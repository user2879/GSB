from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .config import PipelineConfig
from .correlate import build_adjacency, build_ledger_entry
from .dedup import (
    DedupKey,
    DedupState,
    flatten_state,
    group_alerts,
    merge_group_results,
    process_group,
)
from .errors import PipelineError
from .models import AggregateAlert, BatchOutcome, LedgerEntry, ScheduleReport
from .scheduler import (
    ScheduledTask,
    combine_reports,
    conclusion_bytes,
    run_tasks,
)
from .spec import LoadedDataset, decode_alert, encode_alert
from .util import digest_json, sha256_text


@dataclass(frozen=True)
class BatchExecution:
    outcome: BatchOutcome
    state: DedupState
    aggregates: tuple[AggregateAlert, ...]
    ledger: tuple[LedgerEntry, ...]


@dataclass(frozen=True)
class PipelineRun:
    aggregates: tuple[AggregateAlert, ...]
    ledger: tuple[LedgerEntry, ...]
    batches: tuple[BatchOutcome, ...]
    conclusion_digest: str
    schedule: ScheduleReport


BatchCallback = Callable[[BatchExecution], None]
BeforeBatchCallback = Callable[[int, int, int], None]


def _declared_refs(aggregate: AggregateAlert) -> tuple[str, ...]:
    refs: set[str] = set()
    for record in aggregate.source_records:
        refs.update(record.get("root_candidate_refs", []))
    return tuple(sorted(refs))


class PipelineEngine:
    def __init__(
        self,
        config: PipelineConfig,
        dataset: LoadedDataset,
        *,
        mode: str = "parallel",
        global_limit: int = 4,
        per_service_limit: int = 1,
    ) -> None:
        if mode not in {"serial", "parallel"}:
            raise ValueError("mode must be 'serial' or 'parallel'")
        self.config = config
        self.dataset = dataset
        self.mode = mode
        self.global_limit = global_limit
        self.per_service_limit = per_service_limit

    def _execute_stage(
        self,
        tasks: list[ScheduledTask],
        stage: str,
    ) -> tuple[tuple, ScheduleReport]:
        execution = run_tasks(
            tasks,
            mode=self.mode,
            global_limit=self.global_limit,
            per_service_limit=self.per_service_limit,
        )
        if execution.error is not None:
            if isinstance(execution.error, PipelineError):
                raise execution.error
            raise PipelineError(
                "E_STATE",
                f"{stage} task failed: {execution.error}",
                details={"error_type": type(execution.error).__name__},
            ) from execution.error
        return execution.values, execution.report

    def process_batch(
        self,
        *,
        batch_index: int,
        start_cursor: int,
        end_cursor: int,
        state: Mapping[DedupKey, tuple[AggregateAlert, ...]],
    ) -> BatchExecution:
        alerts = self.dataset.alerts[start_cursor:end_cursor]
        if not alerts:
            raise PipelineError(
                "E_STATE",
                "attempted to process an empty batch",
                details={"start_cursor": start_cursor, "end_cursor": end_cursor},
            )

        normalize_tasks: list[ScheduledTask] = []
        for index, alert in enumerate(alerts, start=start_cursor):
            previous_timestamp = (
                self.dataset.alerts[index - 1].timestamp_ms if index > 0 else None
            )

            def normalize_one(
                item=alert, previous=previous_timestamp
            ):
                encoded = encode_alert(item)
                decoded = decode_alert(
                    encoded,
                    item.line_number,
                    self.dataset.graph,
                    self.config,
                    previous,
                )
                if decoded.core_view() != item.core_view():
                    raise PipelineError(
                        "E_STATE",
                        "normalization round-trip is not field-identical",
                        line=item.line_number,
                        field="core_view",
                        details={
                            "before": item.core_view(),
                            "after": decoded.core_view(),
                        },
                    )
                return item

            normalize_tasks.append(
                ScheduledTask(
                    ordinal=len(normalize_tasks),
                    task_id=f"b{batch_index}:normalize:{alert.line_number}",
                    stage="normalize",
                    service_id=alert.service_id,
                    function=normalize_one,
                )
            )
        normalized_values, normalize_report = self._execute_stage(
            normalize_tasks, "normalize"
        )
        normalized_alerts = [value for value in normalized_values if value is not None]

        grouped = group_alerts(normalized_alerts)
        dedup_tasks: list[ScheduledTask] = []
        for key, group in grouped.items():
            existing = tuple(state.get(key, tuple()))

            def dedup_one(group_key=key, items=group, prior=existing):
                return group_key, process_group(items, prior)

            dedup_tasks.append(
                ScheduledTask(
                    ordinal=len(dedup_tasks),
                    task_id=f"b{batch_index}:dedup:{key[0]}:{key[1]}",
                    stage="deduplicate",
                    service_id=key[0],
                    function=dedup_one,
                )
            )
        dedup_values, dedup_report = self._execute_stage(dedup_tasks, "deduplicate")
        updates: dict[DedupKey, tuple[AggregateAlert, ...]] = {
            key: value for key, value in dedup_values if key is not None
        }
        new_state = merge_group_results(state, updates)
        aggregates = flatten_state(new_state)

        adjacency = build_adjacency(self.dataset.graph)
        correlation_tasks: list[ScheduledTask] = []
        for aggregate in aggregates:
            refs = _declared_refs(aggregate)

            def correlate_one(item=aggregate, declared=refs):
                return build_ledger_entry(
                    item,
                    self.dataset.graph,
                    self.dataset.rules,
                    declared,
                    adjacency,
                )

            correlation_tasks.append(
                ScheduledTask(
                    ordinal=len(correlation_tasks),
                    task_id=f"b{batch_index}:correlate:{aggregate.alert_id}",
                    stage="correlate",
                    service_id=aggregate.service_id,
                    function=correlate_one,
                )
            )
        ledger_values, correlation_report = self._execute_stage(
            correlation_tasks, "correlate"
        )
        ledger = tuple(ledger_values)

        schedule = combine_reports(
            [normalize_report, dedup_report, correlation_report],
            self.mode,
            self.global_limit,
            self.per_service_limit,
        )
        conclusion_digest = sha256_text(
            conclusion_bytes(aggregates, ledger).decode("utf-8")
        )
        start_line = alerts[0].line_number
        end_line = alerts[-1].line_number
        summary = {
            "batch_index": batch_index,
            "start_cursor": start_cursor,
            "end_cursor": end_cursor,
            "start_line": start_line,
            "end_line": end_line,
            "normalized_count": len(normalized_alerts),
            "aggregate_count": len(aggregates),
            "ledger_count": len(ledger),
            "conclusion_digest": conclusion_digest,
            "business_code": schedule.business_code,
        }
        outcome = BatchOutcome(
            batch_index=batch_index,
            start_cursor=start_cursor,
            end_cursor=end_cursor,
            status="committed",
            business_code=schedule.business_code,
            schedule=schedule,
            summary_digest=digest_json(summary),
            conclusion_digest=conclusion_digest,
            aggregate_count=len(aggregates),
            ledger_count=len(ledger),
            normalized_count=len(normalized_alerts),
        )
        return BatchExecution(
            outcome=outcome,
            state={key: value for key, value in new_state.items()},
            aggregates=aggregates,
            ledger=ledger,
        )

    def run_all(
        self,
        *,
        start_cursor: int = 0,
        state: Mapping[DedupKey, tuple[AggregateAlert, ...]] | None = None,
        batch_start_index: int = 0,
        on_batch: BatchCallback | None = None,
        before_batch: BeforeBatchCallback | None = None,
        crash_before_commit_after_batch: int | None = None,
    ) -> PipelineRun:
        current_state: DedupState = dict(state or {})
        batches: list[BatchOutcome] = []
        report_parts: list[ScheduleReport] = []
        total = self.dataset.alert_count
        cursor = start_cursor
        batch_index = batch_start_index
        while cursor < total:
            end = min(cursor + self.config.batch_size, total)
            if before_batch is not None:
                before_batch(batch_index, cursor, end)
            execution = self.process_batch(
                batch_index=batch_index,
                start_cursor=cursor,
                end_cursor=end,
                state=current_state,
            )
            if on_batch is not None:
                on_batch(execution)
            current_state = execution.state
            batches.append(execution.outcome)
            report_parts.append(execution.outcome.schedule)
            if (
                crash_before_commit_after_batch is not None
                and len(batches) >= crash_before_commit_after_batch
            ):
                return PipelineRun(
                    aggregates=execution.aggregates,
                    ledger=execution.ledger,
                    batches=tuple(batches),
                    conclusion_digest=execution.outcome.conclusion_digest,
                    schedule=combine_reports(
                        report_parts,
                        self.mode,
                        self.global_limit,
                        self.per_service_limit,
                    ),
                )
            cursor = end
            batch_index += 1

        aggregates = flatten_state(current_state) if current_state else tuple()
        if aggregates:
            refs_by_id = {item.alert_id: _declared_refs(item) for item in aggregates}
            ledger = tuple(
                build_ledger_entry(
                    aggregate,
                    self.dataset.graph,
                    self.dataset.rules,
                    refs_by_id[aggregate.alert_id],
                    build_adjacency(self.dataset.graph),
                )
                for aggregate in aggregates
            )
        else:
            ledger = tuple()
        conclusion_digest = sha256_text(
            conclusion_bytes(aggregates, ledger).decode("utf-8")
        )
        schedule = combine_reports(
            report_parts,
            self.mode,
            self.global_limit,
            self.per_service_limit,
        )
        return PipelineRun(
            aggregates=aggregates,
            ledger=ledger,
            batches=tuple(batches),
            conclusion_digest=conclusion_digest,
            schedule=schedule,
        )


def run_engine(
    config: PipelineConfig,
    dataset: LoadedDataset,
    *,
    mode: str,
    global_limit: int,
    per_service_limit: int,
    on_batch: BatchCallback | None = None,
    before_batch: BeforeBatchCallback | None = None,
    start_cursor: int = 0,
    state: Mapping[DedupKey, tuple[AggregateAlert, ...]] | None = None,
    batch_start_index: int = 0,
    crash_before_commit_after_batch: int | None = None,
) -> PipelineRun:
    engine = PipelineEngine(
        config,
        dataset,
        mode=mode,
        global_limit=global_limit,
        per_service_limit=per_service_limit,
    )
    return engine.run_all(
        start_cursor=start_cursor,
        state=state,
        batch_start_index=batch_start_index,
        on_batch=on_batch,
        before_batch=before_batch,
        crash_before_commit_after_batch=crash_before_commit_after_batch,
    )
