from __future__ import annotations

import hashlib
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .attribution import attribute_conflicts
from .canonical import canonical_bytes, sha256_canonical
from .config import SourceConfig, WorkbenchConfig
from .constants import APP_LOG_SUMMARY, DB_EXEC_STATS, SLOW_LOG_EXCERPT
from .errors import E_EMPTY, SlowDiagError
from .invalidation import (
    PlanDecision,
    commit_plan,
    decide_plan,
    derive_invalidation,
)
from .merge import merge_evidence
from .migration import migrate_conclusions
from .storage import Ledger, SourceDataset, read_all_sources


VIEW_SCHEMA = "slowdiag.view/1"


@dataclass(frozen=True)
class CoreConclusions:
    queries: list[dict[str, Any]]
    migration: dict[str, Any]
    attribution: dict[str, Any]


@dataclass(frozen=True)
class PipelineResult:
    view: dict[str, Any]
    view_sha256: str
    datasets: list[SourceDataset]
    timings: dict[str, Any]


def _ms_since(start_ns: int) -> int:
    return (time.perf_counter_ns() - start_ns) // 1_000_000


def compute_core(
    config: WorkbenchConfig,
    datasets: list[SourceDataset],
    *,
    empty_after_merge: bool = False,
) -> tuple[CoreConclusions, dict[str, int]]:
    timings: dict[str, int] = {}
    started = time.perf_counter_ns()
    queries = merge_evidence(config, datasets)
    timings["merge_ms"] = _ms_since(started)
    if empty_after_merge:
        raise SlowDiagError(
            E_EMPTY,
            "an injected parameter emptied the merged query set",
            details={"injection": "empty", "merged_query_count": len(queries)},
        )
    started = time.perf_counter_ns()
    migration_document = migrate_conclusions(config, datasets)
    timings["migration_ms"] = _ms_since(started)
    started = time.perf_counter_ns()
    attribution_document = attribute_conflicts(config, datasets, migration_document)
    timings["attribution_ms"] = _ms_since(started)
    return (
        CoreConclusions(
            queries=queries,
            migration=migration_document,
            attribution=attribution_document,
        ),
        timings,
    )


def build_view(
    config: WorkbenchConfig,
    plan_version: int,
    datasets: list[SourceDataset],
    core: CoreConclusions,
    invalidation: dict[str, Any],
) -> dict[str, Any]:
    summaries = [dataset.summary() for dataset in sorted(datasets, key=lambda d: d.config.id)]
    return {
        "schema": VIEW_SCHEMA,
        "plan_version": plan_version,
        "config_sha256": config.fingerprint_sha256(),
        "source_summaries": summaries,
        "queries": core.queries,
        "migration": core.migration,
        "attribution": core.attribution,
        "invalidation": invalidation,
    }


def run_pipeline(
    config: WorkbenchConfig,
    ledger: Ledger,
    *,
    empty_after_merge: bool = False,
) -> PipelineResult:
    total_started = time.perf_counter_ns()
    timings: dict[str, Any] = {"clock": "perf_counter_ns"}

    started = time.perf_counter_ns()
    datasets = read_all_sources(config)
    timings["read_sources_ms"] = _ms_since(started)

    old_view_meta = ledger.get_current_view()
    old_view = old_view_meta["view"] if old_view_meta else None
    decision = decide_plan(ledger, config)

    core, core_timings = compute_core(
        config, datasets, empty_after_merge=empty_after_merge
    )
    timings.update(core_timings)

    started = time.perf_counter_ns()
    invalidation = derive_invalidation(
        config,
        decision,
        old_view,
        core.queries,
        core.migration,
        core.attribution,
        datasets,
    )
    timings["invalidation_ms"] = _ms_since(started)

    view = build_view(config, decision.plan_version, datasets, core, invalidation)
    view_sha = sha256_canonical(view)
    timings["pipeline_ms"] = _ms_since(total_started)
    run_metrics = {
        **timings,
        "plan_version": decision.plan_version,
        "query_count": len(core.queries),
    }
    ledger.replace_view(view, datasets, decision.plan_version, run_metrics=run_metrics)
    commit_plan(ledger, decision, config)
    return PipelineResult(
        view=view,
        view_sha256=view_sha,
        datasets=datasets,
        timings=run_metrics,
    )


def recompute_view_from_config(
    config: WorkbenchConfig,
    plan_version: int,
) -> tuple[dict[str, Any], list[SourceDataset], dict[str, int]]:
    started = time.perf_counter_ns()
    datasets = read_all_sources(config)
    read_ms = _ms_since(started)
    core, core_timings = compute_core(config, datasets)
    invalidation = {
        "plan_version": plan_version,
        "previous_plan_version": None,
        "version_changed": False,
        "reasons": ["snapshot_replay"],
        "replacement_mode": "whole_view",
        "affected_queries": [],
        "unaffected_queries": sorted(query["query_id"] for query in core.queries),
        "stale_query_count": 0,
        "changed_evidence": [],
        "unaffected_byte_identical": True,
        "unaffected_checks": [],
    }
    view = build_view(config, plan_version, datasets, core, invalidation)
    timings = {"read_sources_ms": read_ms, **core_timings}
    return view, datasets, timings


def make_format_conflict_source(
    workspace: Path,
    source: SourceConfig,
) -> SourceConfig:
    workspace = workspace.resolve(strict=False)
    target_dir = workspace / "_injection" / "format"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.id}.sqlite3"
    shutil.copyfile(source.resolved_data_path, target)
    connection = sqlite3.connect(str(target))
    try:
        row = connection.execute(
            "SELECT payload FROM records WHERE row_no=1"
        ).fetchone()
        if row is None:
            raise SlowDiagError(
                E_EMPTY,
                "cannot inject a format conflict into an empty source",
                source_id=source.id,
            )
        import json

        envelope = json.loads(bytes(row[0]).decode("utf-8"))
        body = dict(envelope["body"])
        record_version = int(envelope["version"])
        cross_version_field = {
            APP_LOG_SUMMARY: {1: "latency_p95_ms", 2: "p95_ms"},
            DB_EXEC_STATS: {1: "latency_p95_ms", 2: "p95_ms"},
            SLOW_LOG_EXCERPT: {1: "latency_max_ms", 2: "max_ms"},
        }[source.type][record_version]
        body[cross_version_field] = 1
        envelope["body"] = body
        envelope["length"] = len(canonical_bytes(body))
        payload = canonical_bytes(envelope)
        connection.execute(
            "UPDATE records SET payload=?, payload_sha256=? WHERE row_no=1",
            (payload, hashlib.sha256(payload).hexdigest()),
        )
        connection.commit()
    finally:
        connection.close()
    return source.with_data_path(target, workspace)
