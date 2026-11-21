from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .canonical import canonical_text, digest_text
from .errors import ArbiterError
from .jsonlog import JsonlLogger
from .ledger import LedgerDB
from .pool import BatchScheduler, ScheduledBatch, WorkerPool
from .spec import Dataset, dataset_from_payload


PROTOCOL_VERSION = 1


@dataclass
class EngineConfig:
    global_concurrency: int = 3
    per_node_concurrency: int = 2
    batch_size: int = 4
    request_timeout_ms: int = 2_000
    retries: int = 2
    handshake_timeout_ms: int = 5_000
    fault_after_commits: int | None = None
    python_executable: str | None = None
    command_factory: Callable[[Any], list[str]] | None = None

    def validate(self) -> None:
        for name in ("global_concurrency", "per_node_concurrency", "batch_size"):
            if getattr(self, name) < 1:
                raise ArbiterError("E_SPEC", f"{name} must be >= 1", path=f"/{name}")
        if self.request_timeout_ms < 1:
            raise ArbiterError("E_SPEC", "request_timeout_ms must be >= 1")
        if self.retries < 0:
            raise ArbiterError("E_SPEC", "retries must be >= 0")
        if self.handshake_timeout_ms < 1:
            raise ArbiterError("E_SPEC", "handshake_timeout_ms must be >= 1")
        if self.fault_after_commits is not None and self.fault_after_commits < 1:
            raise ArbiterError("E_SPEC", "fault_after_commits must be >= 1 when provided")

    def concurrency_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "global_concurrency": self.global_concurrency,
            "handshake_timeout_ms": self.handshake_timeout_ms,
            "per_node_concurrency": self.per_node_concurrency,
            "request_timeout_ms": self.request_timeout_ms,
            "retries": self.retries,
        }


def build_scheduled_batches(dataset: Dataset, *, batch_size: int) -> list[ScheduledBatch]:
    if batch_size < 1:
        raise ArbiterError("E_SPEC", "batch_size must be >= 1")
    nodes = dataset.nodes_payload()["nodes"]
    spans = dataset.spans_payload()
    spans_by_node: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        spans_by_node.setdefault(span["node_id"], []).append(span)

    batches: list[ScheduledBatch] = []
    batch_index = 0
    for node in sorted(dataset.nodes, key=lambda item: item.node_id):
        owned = sorted(
            spans_by_node.get(node.node_id, []),
            key=lambda span: (span["trace_id"], span["timestamp_ms"], span["span_id"]),
        )
        for offset in range(0, len(owned), batch_size):
            chunk = owned[offset : offset + batch_size]
            target_spans = sorted(
                [{"span_id": span["span_id"], "trace_id": span["trace_id"]} for span in chunk],
                key=lambda item: (item["trace_id"], item["span_id"]),
            )
            request = {
                "mode": "batch",
                "nodes": nodes,
                "protocol_version": PROTOCOL_VERSION,
                "ruleset_version": dataset.ruleset_version,
                "spans": spans,
                "target_node_id": node.node_id,
                "target_spans": target_spans,
                "worker_node_id": node.node_id,
            }
            text = canonical_text(request)
            batches.append(
                ScheduledBatch(
                    batch_index=batch_index,
                    node_id=node.node_id,
                    request=request,
                    request_text=text,
                    request_digest=digest_text(text),
                )
            )
            batch_index += 1

    finalize_node = sorted(dataset.nodes, key=lambda item: item.node_id)[0]
    finalize_request = {
        "mode": "finalize",
        "nodes": nodes,
        "protocol_version": PROTOCOL_VERSION,
        "ruleset_version": dataset.ruleset_version,
        "spans": spans,
        "worker_node_id": finalize_node.node_id,
    }
    finalize_text = canonical_text(finalize_request)
    batches.append(
        ScheduledBatch(
            batch_index=batch_index,
            node_id=finalize_node.node_id,
            request=finalize_request,
            request_text=finalize_text,
            request_digest=digest_text(finalize_text),
        )
    )
    return batches


def _fresh_submission_times(batches: list[ScheduledBatch]) -> None:
    now = time.perf_counter_ns()
    for batch in batches:
        batch.submitted_ns = now


def _scheduler(
    dataset: Dataset,
    config: EngineConfig,
    *,
    global_concurrency: int,
    per_node_concurrency: int,
    request_timeout_ms: int,
    retries: int,
    pool: WorkerPool,
) -> BatchScheduler:
    return BatchScheduler(
        pool,
        global_concurrency=global_concurrency,
        per_node_concurrency=per_node_concurrency,
        timeout_ms=request_timeout_ms,
        retries=retries,
    )


def _concurrency_result(scheduler: BatchScheduler, config: EngineConfig) -> dict[str, Any]:
    snapshot = scheduler.snapshot()
    node_ok = all(
        count <= config.per_node_concurrency for count in snapshot.max_node_running.values()
    )
    global_ok = snapshot.max_global_running <= config.global_concurrency
    result = snapshot.as_dict()
    result.update(
        {
            "global_limit": config.global_concurrency,
            "per_node_limit": config.per_node_concurrency,
            "within_limits": bool(node_ok and global_ok),
        }
    )
    if not result["within_limits"]:
        raise ArbiterError("E_PROTOCOL", "observed concurrency exceeded configured limits", details=result)
    return result


def _first_outcome_error(outcomes: list[Any], errors: list[ArbiterError]) -> ArbiterError | None:
    failed = [outcome.error for outcome in outcomes if outcome.error is not None]
    if failed:
        return min(
            failed,
            key=lambda error: (error.details.get("batch_index", 10**18), error.code),
        )
    if errors:
        return errors[0]
    return None


def _make_pool(dataset: Dataset, config: EngineConfig) -> WorkerPool:
    return WorkerPool(
        dataset.nodes,
        handshake_timeout_ms=config.handshake_timeout_ms,
        python_executable=config.python_executable,
        command_factory=config.command_factory,
    )


def run_adjudication(
    db: LedgerDB,
    dataset: Dataset,
    config: EngineConfig,
    logger: JsonlLogger,
) -> dict[str, Any]:
    config.validate()
    incomplete = db.incomplete_run()
    if incomplete is not None:
        raise ArbiterError(
            "E_RECOVERY",
            "an incomplete run exists; recover it before starting another adjudication",
            details={
                "run_id": incomplete["run_id"],
                "status": incomplete["status"],
                "progress_cursor": incomplete["progress_cursor"],
                "batch_count": incomplete["batch_count"],
            },
        )
    batches = build_scheduled_batches(dataset, batch_size=config.batch_size)
    run_id = uuid.uuid4().hex
    snapshot_id, run_id = db.create_snapshot_and_run(
        dataset,
        run_id=run_id,
        batches=[
            {
                "batch_index": batch.batch_index,
                "mode": batch.request["mode"],
                "node_id": batch.node_id,
                "request_text": batch.request_text,
            }
            for batch in batches
        ],
        concurrency=config.concurrency_dict(),
    )
    db.mark_run_running(run_id)
    logger.emit(
        "adjudication_start",
        run_id=run_id,
        snapshot_id=snapshot_id,
        ruleset_version=dataset.ruleset_version,
        batch_count=len(batches),
        global_concurrency=config.global_concurrency,
        per_node_concurrency=config.per_node_concurrency,
    )
    started_ns = time.perf_counter_ns()
    pool = _make_pool(dataset, config)
    commit_lock = threading.Lock()
    committed_successes = 0
    try:
        try:
            pool.start()
        except ArbiterError as exc:
            db.fail_run(run_id, exc)
            logger.error("adjudication_spawn_failed", exc)
            raise
        _fresh_submission_times(batches)
        scheduler = _scheduler(
            dataset,
            config,
            global_concurrency=config.global_concurrency,
            per_node_concurrency=config.per_node_concurrency,
            request_timeout_ms=config.request_timeout_ms,
            retries=config.retries,
            pool=pool,
        )

        def on_result(outcome: Any) -> None:
            nonlocal committed_successes
            db.commit_batch(run_id, outcome, operation="adjudicate")
            if outcome.error is None:
                with commit_lock:
                    committed_successes += 1
                    current = committed_successes
                logger.emit(
                    "batch_committed",
                    run_id=run_id,
                    batch_index=outcome.batch.batch_index,
                    node_id=outcome.batch.node_id,
                    attempts=outcome.attempts,
                    queue_wait_us=outcome.queue_wait_us,
                    progress_committed=current,
                )
                if (
                    config.fault_after_commits is not None
                    and current >= config.fault_after_commits
                ):
                    logger.emit("fault_injection_forced_exit", run_id=run_id)
                    os._exit(137)
            else:
                logger.error("batch_failed", outcome.error)

        outcomes, errors = scheduler.run(batches, on_result=on_result, abort_on_error=False)
    finally:
        pool.terminate()

    primary_error = _first_outcome_error(outcomes, errors)
    if primary_error is not None:
        logger.error("adjudication_failed", primary_error)
        raise primary_error
    finalize_outcome = outcomes[-1]
    if finalize_outcome.error is not None:
        logger.error("finalize_failed", finalize_outcome.error)
        raise finalize_outcome.error
    concurrency = _concurrency_result(scheduler, config)
    try:
        finalized = db.finalize_run(
            dataset=dataset,
            run_id=run_id,
            snapshot_id=snapshot_id,
            finalize_response=finalize_outcome.response,
            concurrency=concurrency,
        )
    except ArbiterError as exc:
        db.fail_run(run_id, exc)
        logger.error("ledger_finalize_failed", exc)
        raise
    ended_ns = time.perf_counter_ns()
    db.record_latency(
        snapshot_id=snapshot_id,
        operation="adjudicate",
        node_id=None,
        started_ns=started_ns,
        ended_ns=ended_ns,
        rebuild=False,
    )
    db.rebuild_latency_stats(snapshot_id)
    result = {
        "authority": "service_decision_summary",
        "batch_count": len(batches),
        "concurrency": concurrency,
        "elapsed_us": (ended_ns - started_ns) // 1000,
        "ledger": finalized,
        "ok": True,
        "operation": "adjudicate",
        "ruleset_version": dataset.ruleset_version,
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "stats": db.latency_stats(snapshot_id),
    }
    logger.emit("adjudication_complete", **result)
    return result


def _record_service_latencies(
    db: LedgerDB,
    snapshot_id: int,
    operation: str,
    outcomes: list[Any],
) -> None:
    for outcome in outcomes:
        if outcome.error is not None:
            continue
        db.record_latency(
            snapshot_id=snapshot_id,
            operation=operation,
            node_id=outcome.batch.node_id,
            started_ns=outcome.queue_exit_ns,
            ended_ns=outcome.finished_ns,
            rebuild=False,
        )
    db.rebuild_latency_stats(snapshot_id, operation=operation)


def execute_service_summary(
    db: LedgerDB,
    dataset: Dataset,
    config: EngineConfig,
    logger: JsonlLogger,
    *,
    snapshot_id: int,
    operation: str,
) -> dict[str, Any]:
    config.validate()
    batches = build_scheduled_batches(dataset, batch_size=config.batch_size)
    _fresh_submission_times(batches)
    started_ns = time.perf_counter_ns()
    pool = _make_pool(dataset, config)
    outcomes: list[Any] = []
    errors: list[ArbiterError] = []
    try:
        pool.start()
        scheduler = _scheduler(
            dataset,
            config,
            global_concurrency=config.global_concurrency,
            per_node_concurrency=config.per_node_concurrency,
            request_timeout_ms=config.request_timeout_ms,
            retries=config.retries,
            pool=pool,
        )
        outcomes, errors = scheduler.run(batches, abort_on_error=False)
        concurrency = _concurrency_result(scheduler, config)
    finally:
        pool.terminate()
        ended_ns = time.perf_counter_ns()
        _record_service_latencies(
            db,
            snapshot_id,
            operation,
            outcomes,
        )

    primary_error = _first_outcome_error(outcomes, errors)
    if primary_error is not None:
        logger.error("service_summary_failed", primary_error)
        raise primary_error
    finalize = outcomes[-1].response
    summary = finalize.get("decision_summary")
    digest = finalize.get("decision_digest")
    if not isinstance(summary, str) or digest != digest_text(summary):
        raise ArbiterError("E_PROTOCOL", "finalize summary digest is invalid")
    return {
        "concurrency": concurrency,
        "elapsed_us": (ended_ns - started_ns) // 1000,
        "finalize": finalize,
        "outcomes": outcomes,
    }


def _trace_field_difference(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    for field in (
        "root_span_id",
        "sampled_node_ids",
        "sampled_span_ids",
        "participating_nodes",
        "rule_difference",
        "span_decisions",
    ):
        if expected.get(field) != actual.get(field):
            return field
    return "decision_summary"


def verify_service_against_ledger(
    db: LedgerDB,
    *,
    snapshot_id: int,
    finalize_response: dict[str, Any],
    error_code: str,
) -> dict[str, Any]:
    db.verify_ledger_entries(snapshot_id, error_code=error_code)
    ledger_rows = db.snapshot_traces(snapshot_id)
    ledger_by_trace = {row["trace_id"]: row for row in ledger_rows}
    actual_traces = finalize_response.get("trace_results")
    if not isinstance(actual_traces, list):
        raise ArbiterError("E_PROTOCOL", "finalize response is missing trace_results")
    actual_by_trace = {trace["trace_id"]: trace for trace in actual_traces}
    for trace_id in sorted(set(ledger_by_trace) | set(actual_by_trace)):
        if trace_id not in actual_by_trace:
            raise ArbiterError(
                error_code,
                "trace exists in ledger but not in authoritative service result",
                trace_id=trace_id,
                field="trace_id",
            )
        if trace_id not in ledger_by_trace:
            raise ArbiterError(
                error_code,
                "trace exists in authoritative service result but not in ledger",
                trace_id=trace_id,
                field="trace_id",
            )
        ledger_row = ledger_by_trace[trace_id]
        entry = ledger_row["entry"]
        actual_trace = actual_by_trace[trace_id]
        try:
            ledger_summary_payload = json.loads(entry["decision_summary"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ArbiterError(
                error_code,
                "ledger entry contains an invalid decision summary",
                trace_id=trace_id,
                field="decision_summary",
            ) from exc
        for field in (
            "root_span_id",
            "sampled_node_ids",
            "sampled_span_ids",
            "participating_nodes",
            "rule_difference",
        ):
            if entry.get(field) != ledger_summary_payload.get(field):
                raise ArbiterError(
                    error_code,
                    "ledger entry fields disagree with its embedded decision summary",
                    trace_id=trace_id,
                    field=field,
                )
        if entry.get("trace_id") != trace_id:
            raise ArbiterError(
                error_code,
                "ledger entry trace_id does not match snapshot mapping",
                trace_id=trace_id,
                field="trace_id",
            )
        if entry.get("format_version") != 1:
            raise ArbiterError(
                error_code,
                "ledger entry has an unsupported integer format_version",
                trace_id=trace_id,
                field="format_version",
            )
        if entry.get("decision_digest") != digest_text(entry["decision_summary"]):
            raise ArbiterError(
                error_code,
                "ledger entry decision_digest does not match its decision summary",
                trace_id=trace_id,
                field="decision_digest",
            )
        actual_summary = canonical_text(actual_trace)
        if entry.get("decision_summary") != actual_summary:
            expected_payload = json.loads(entry.get("decision_summary", "{}"))
            field = _trace_field_difference(expected_payload, actual_trace)
            raise ArbiterError(
                error_code,
                "authoritative service summary differs from ledger bytes",
                trace_id=trace_id,
                field=field,
                details={
                    "ledger_digest": entry.get("decision_digest"),
                    "service_digest": digest_text(actual_summary),
                },
            )
        actual_rules = sorted(actual_trace.get("rule_versions", []))
        if sorted(ledger_row["rule_versions"]) != actual_rules:
            raise ArbiterError(
                error_code,
                "ledger rule version set differs from authoritative derivation",
                trace_id=trace_id,
                field="rule_versions",
                details={
                    "ledger_rule_versions": ledger_row["rule_versions"],
                    "service_rule_versions": actual_rules,
                },
            )
    return {
        "ok": True,
        "snapshot_id": snapshot_id,
        "trace_count": len(actual_by_trace),
        "truth_source": "service_decision_summary",
    }


def review_current(db: LedgerDB, config: EngineConfig, logger: JsonlLogger) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    snapshot = db.get_snapshot()
    snapshot = db.verify_snapshot_digest(snapshot["snapshot_id"])
    dataset = dataset_from_payload(
        {
            "format_version": snapshot["format_version"],
            "ruleset_version": snapshot["ruleset_version"],
            "nodes": snapshot["nodes"],
            "spans": snapshot["spans"],
        }
    )
    try:
        result = execute_service_summary(
            db,
            dataset,
            config,
            logger,
            snapshot_id=snapshot["snapshot_id"],
            operation="review",
        )
        verified = verify_service_against_ledger(
            db,
            snapshot_id=snapshot["snapshot_id"],
            finalize_response=result["finalize"],
            error_code="E_DIVERGE",
        )
    finally:
        ended_ns = time.perf_counter_ns()
        db.record_latency(
            snapshot_id=snapshot["snapshot_id"],
            operation="review",
            node_id=None,
            started_ns=started_ns,
            ended_ns=ended_ns,
            rebuild=False,
        )
        db.rebuild_latency_stats(snapshot["snapshot_id"], operation="review")
    verified.update(
        {
            "concurrency": result["concurrency"],
            "elapsed_us": result["elapsed_us"],
            "operation": "review",
            "ruleset_version": snapshot["ruleset_version"],
            "snapshot_id": snapshot["snapshot_id"],
            "stats": db.latency_stats(snapshot["snapshot_id"]),
        }
    )
    logger.emit("authoritative_review_complete", **verified)
    return verified


def replay_snapshot(
    db: LedgerDB,
    config: EngineConfig,
    logger: JsonlLogger,
    *,
    inject_snapshot_mismatch: bool = False,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    snapshot = db.get_snapshot()
    snapshot = db.verify_snapshot_digest(
        snapshot["snapshot_id"],
        inject_mismatch=inject_snapshot_mismatch,
    )
    dataset = dataset_from_payload(
        {
            "format_version": snapshot["format_version"],
            "ruleset_version": snapshot["ruleset_version"],
            "nodes": snapshot["nodes"],
            "spans": snapshot["spans"],
        }
    )
    try:
        result = execute_service_summary(
            db,
            dataset,
            config,
            logger,
            snapshot_id=snapshot["snapshot_id"],
            operation="replay",
        )
        verified = verify_service_against_ledger(
            db,
            snapshot_id=snapshot["snapshot_id"],
            finalize_response=result["finalize"],
            error_code="E_REPLAY",
        )
    finally:
        ended_ns = time.perf_counter_ns()
        db.record_latency(
            snapshot_id=snapshot["snapshot_id"],
            operation="replay",
            node_id=None,
            started_ns=started_ns,
            ended_ns=ended_ns,
            rebuild=False,
        )
        db.rebuild_latency_stats(snapshot["snapshot_id"], operation="replay")
    stats = db.latency_stats(snapshot["snapshot_id"])
    verified.update(
        {
            "concurrency": result["concurrency"],
            "elapsed_us": result["elapsed_us"],
            "operation": "replay",
            "ruleset_version": snapshot["ruleset_version"],
            "snapshot_content_digest": snapshot["content_digest"],
            "snapshot_id": snapshot["snapshot_id"],
            "stats": stats,
        }
    )
    logger.emit("replay_complete", **verified)
    return verified


def recover_incomplete(
    db: LedgerDB,
    config: EngineConfig,
    logger: JsonlLogger,
    *,
    replay_after: bool = True,
) -> dict[str, Any]:
    run = db.incomplete_run()
    if run is None:
        return {"ok": True, "status": "nothing_to_recover"}
    run_id = run["run_id"]
    unknown = db.mark_unknown_batches(run_id)
    logger.emit("recovery_unknown_batches", run_id=run_id, batch_indices=unknown)
    snapshot = db.verify_snapshot_digest(run["snapshot_id"])
    dataset = dataset_from_payload(
        {
            "format_version": snapshot["format_version"],
            "ruleset_version": snapshot["ruleset_version"],
            "nodes": snapshot["nodes"],
            "spans": snapshot["spans"],
        }
    )
    rows = db.list_batches(run_id)
    rerun_rows = [row for row in rows if row["status"] == "unknown"]
    rerun = [
        ScheduledBatch(
            batch_index=int(row["batch_index"]),
            node_id=row["node_id"],
            request=json.loads(row["request_json"]),
            request_text=row["request_json"],
            request_digest=row["request_digest"],
        )
        for row in rerun_rows
    ]
    recovery_config = EngineConfig(
        global_concurrency=int(run["global_concurrency"]),
        per_node_concurrency=int(run["per_node_concurrency"]),
        batch_size=int(run["batch_size"]),
        request_timeout_ms=int(run["request_timeout_ms"]),
        retries=int(run["retries"]),
        handshake_timeout_ms=config.handshake_timeout_ms,
        python_executable=config.python_executable,
        command_factory=config.command_factory,
    )
    started_ns = time.perf_counter_ns()
    pool = _make_pool(dataset, recovery_config)
    try:
        pool.start()
        if rerun:
            _fresh_submission_times(rerun)
            scheduler = _scheduler(
                dataset,
                recovery_config,
                global_concurrency=recovery_config.global_concurrency,
                per_node_concurrency=recovery_config.per_node_concurrency,
                request_timeout_ms=recovery_config.request_timeout_ms,
                retries=recovery_config.retries,
                pool=pool,
            )

            def on_result(outcome: Any) -> None:
                db.commit_batch(run_id, outcome, operation="recovery")
                logger.emit(
                    "recovery_batch_committed",
                    run_id=run_id,
                    batch_index=outcome.batch.batch_index,
                    node_id=outcome.batch.node_id,
                    error_code=outcome.error.code if outcome.error else None,
                )

            outcomes, errors = scheduler.run(rerun, on_result=on_result, abort_on_error=False)
            concurrency = _concurrency_result(scheduler, recovery_config)
            primary = _first_outcome_error(outcomes, errors)
            if primary is not None:
                logger.error("recovery_batch_failed", primary)
                raise primary
        else:
            concurrency = {"within_limits": True}
    finally:
        pool.terminate()

    rows = db.list_batches(run_id)
    unfinished = [row for row in rows if row["status"] != "committed"]
    if unfinished:
        raise ArbiterError(
            "E_RECOVERY",
            "recovery still has non-committed batches",
            details={"batch_indices": [row["batch_index"] for row in unfinished]},
        )
    finalize_row = next(row for row in rows if row["mode"] == "finalize")
    if finalize_row["response_json"] is None:
        raise ArbiterError("E_RECOVERY", "finalize response was not committed")
    finalize_response = json.loads(finalize_row["response_json"])
    finalized = db.finalize_run(
        dataset=dataset,
        run_id=run_id,
        snapshot_id=int(run["snapshot_id"]),
        finalize_response=finalize_response,
        concurrency=concurrency,
    )
    ended_ns = time.perf_counter_ns()
    db.record_latency(
        snapshot_id=int(run["snapshot_id"]),
        operation="recovery",
        node_id=None,
        started_ns=started_ns,
        ended_ns=ended_ns,
        rebuild=False,
    )
    db.rebuild_latency_stats(int(run["snapshot_id"]))
    result: dict[str, Any] = {
        "batch_indices_rerun": unknown,
        "concurrency": concurrency,
        "ledger": finalized,
        "ok": True,
        "run_id": run_id,
        "snapshot_id": int(run["snapshot_id"]),
        "status": "recovered",
    }
    logger.emit("recovery_complete", **result)
    if replay_after:
        result["replay"] = replay_snapshot(db, recovery_config, logger)
    return result


def view_current(db: LedgerDB) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    view = db.load_view()
    ended_ns = time.perf_counter_ns()
    snapshot_id = int(view["snapshot_id"])
    db.record_latency(
        snapshot_id=snapshot_id,
        operation="view",
        node_id=None,
        started_ns=started_ns,
        ended_ns=ended_ns,
        rebuild=False,
    )
    db.rebuild_latency_stats(snapshot_id, operation="view")
    view["latency_stats"] = db.latency_stats(snapshot_id)
    view["query_elapsed_us"] = (ended_ns - started_ns) // 1000
    return view
