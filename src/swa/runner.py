"""End-to-end orchestration: generation, scheduling, ledger commit and recovery."""

from __future__ import annotations

import os
import uuid
from typing import Any

from .engine import Engine, WindowResult
from .errors import LEDGER_FORMAT_VERSION, BusinessError
from .generator import (
    build_round_robin_plan,
    event_stream_digest,
    generate_all_events,
)
from .ledger import Ledger
from .logging_setup import StructuredLogger
from .replay import verify_committed_windows
from .spec import Spec, load_sample_overrides, load_spec
from .util import canonical_json, digest_of, monotonic_ms


def build_result_document(results: list[WindowResult]) -> dict[str, Any]:
    windows = [result.output_dict() for result in results]
    document = {"format_version": 1, "windows": windows}
    document["result_digest"] = digest_of(
        {"format_version": 1, "windows": [window_result.ledger_dict() for window_result in results]}
    )
    return document


def _reorder_plan(plan: list[Any], reorder: str | None, spec: Spec) -> list[Any]:
    if not reorder:
        return plan
    if reorder == "reverse":
        # Reverse the global delivery order while keeping per-source delivery
        # monotonic by reconstructing from source streams.
        per_source: dict[str, list[Any]] = {source.source_id: [] for source in spec.sources}
        for event in plan:
            per_source[event.source_id].append(event)
        merged: list[Any] = []
        queues = {key: list(reversed(events)) for key, events in per_source.items()}
        order = sorted(queues, reverse=True)
        while any(queues.values()):
            for source_id in order:
                if queues[source_id]:
                    merged.append(queues[source_id].pop())
        return merged
    raise BusinessError("E_SPEC", f"unknown reorder mode '{reorder}'", "reorder")


def run_pipeline(
    spec_path: str,
    ledger_path: str,
    result_path: str,
    *,
    log_path: str,
    global_limit: int = 3,
    sample_path: str | None = None,
    injection: str | None = None,
    ledger_format_version: int = LEDGER_FORMAT_VERSION,
    budget_wait_threshold_ms: float | None = None,
    crash_after_commits: int | None = None,
    reorder: str | None = None,
    schedule_jitter: bool = False,
) -> dict[str, Any]:
    logger = StructuredLogger(log_path)
    run_id = f"run-{uuid.uuid4().hex}"
    started = monotonic_ms()
    try:
        spec = load_spec(spec_path, injection=injection)
        overrides = load_sample_overrides(sample_path) if sample_path else {}
        streams = generate_all_events(spec, overrides)
        event_digest = event_stream_digest(streams)
        plan = _reorder_plan(build_round_robin_plan(streams), reorder, spec)

        ledger = Ledger(ledger_path, expected_format_version=ledger_format_version)
        replacement = ledger.register_rule_versions(spec.rules)
        ledger.start_run(run_id, spec)
        committed_before = ledger.current_committed()

        engine = Engine(
            spec,
            global_limit,
            budget_wait_threshold_ms=budget_wait_threshold_ms,
            schedule_jitter=schedule_jitter,
        )
        report = engine.ingest(plan, streams)

        ledger_actions = {"inserted": 0, "reused": 0}
        for result in report.results:
            action = ledger.commit_window(run_id, result)
            ledger_actions[action] += 1
            logger.log(
                "info",
                "window_committed",
                rule_id=result.rule_id,
                window_id=result.window_id,
                window_start_ms=result.window_start_ms,
                window_end_ms=result.window_end_ms,
                action=action,
            )
            if crash_after_commits is not None and ledger_actions["inserted"] >= crash_after_commits:
                logger.log(
                    "error",
                    "forced_termination_injected",
                    run_id=run_id,
                    inserted_windows=ledger_actions["inserted"],
                )
                logger.close()
                # Hard stop: anything not yet committed is durably unknown.
                os._exit(137)

        ledger.finish_run(run_id, "complete")
        document = build_result_document(report.results)
        with open(result_path, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(document))

        affected_windows: list[str] = []
        for rule_id, starts in replacement.items():
            affected_windows.extend(
                f"{rule_id}:v{next(rule.version for rule in spec.rules if rule.rule_id == rule_id) - 1}:{start}"
                for start in starts
            )
        payload = {
            "run_id": run_id,
            "status": "ok",
            "event_digest": event_digest,
            "result_digest": document["result_digest"],
            "result_file": os.path.abspath(result_path),
            "ledger_file": os.path.abspath(ledger_path),
            "event_report": report.event_report(),
            "scheduler": {
                key: value
                for key, value in engine.scheduler.metrics().items()
                if key != "wait_records"
            },
            "ledger_actions": ledger_actions,
            "recovery": _recovery_summary(ledger, committed_before),
            "view_replacement": {
                "replaced": bool(replacement),
                "affected_rules": sorted(replacement),
                "affected_windows": sorted(affected_windows),
                "stale_rows": sum(len(starts) for starts in replacement.values()),
            },
            "window_count": len(report.results),
            "elapsed_monotonic_ms": round(monotonic_ms() - started, 3),
        }
        logger.log("info", "run_complete", run_id=run_id, window_count=len(report.results))
        ledger.close()
        logger.close()
        return payload
    except BusinessError as exc:
        logger.log(
            "error",
            "business_error",
            code=exc.code,
            path=exc.path,
            message=exc.message,
            details=exc.details,
        )
        logger.close()
        raise


def _recovery_summary(ledger: Ledger, previous: dict[tuple[str, int], Any]) -> list[dict[str, Any]]:
    current = ledger.current_committed()
    return [
        {"window_id": row["window_id"], "rule_id": row["rule_id"], "window_start_ms": row["window_start_ms"]}
        for key, row in sorted(current.items())
        if key in previous
    ]


def replay_pipeline(
    spec_path: str,
    ledger_path: str,
    *,
    log_path: str,
    global_limit: int = 3,
    sample_path: str | None = None,
    ledger_format_version: int = LEDGER_FORMAT_VERSION,
) -> dict[str, Any]:
    logger = StructuredLogger(log_path)
    try:
        spec = load_spec(spec_path)
        overrides = load_sample_overrides(sample_path) if sample_path else {}
        streams = generate_all_events(spec, overrides)
        plan = build_round_robin_plan(streams)
        ledger = Ledger(ledger_path, expected_format_version=ledger_format_version)
        ledger.register_rule_versions(spec.rules)
        engine = Engine(spec, global_limit)
        report = engine.ingest(plan, streams)
        signatures = {rule.rule_id: rule.signature() for rule in spec.rules}
        verification = verify_committed_windows(ledger, report.results, rule_signatures=signatures)
        verification.update(
            {
                "event_report": report.event_report(),
                "scheduler": {
                    key: value
                    for key, value in engine.scheduler.metrics().items()
                    if key != "wait_records"
                },
            }
        )
        logger.log("info", "replay_verified", verified_windows=verification["verified_windows"])
        ledger.close()
        logger.close()
        return verification
    except BusinessError as exc:
        logger.log("error", "business_error", code=exc.code, path=exc.path, details=exc.details)
        logger.close()
        raise
