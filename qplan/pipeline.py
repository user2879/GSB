from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from .bias import analyze_bias
from .canonical import content_digest, normalize
from .cost import estimate_query_paths
from .data import data_digest_summary, materialize_database
from .executor import execute_query_paths
from .invalidation import compute_invalidation
from .ledger import Ledger, compute_snapshot_content_digest
from .paths import AccessPath, build_candidate_paths
from .search import search_indexes
from .spec import Spec
from .stats import collect_statistics, statistics_digest


PLATFORM_VERSION = "qplan-diagnostics-0.1.0"


def build_estimates(spec: Spec, statistics: dict[str, Any]) -> dict[str, Any]:
    paths_by_query: dict[str, list[AccessPath]] = {}
    estimates_by_query: dict[str, dict[str, dict[str, Any]]] = {}
    for query in spec.queries:
        paths = build_candidate_paths(spec, query)
        paths_by_query[query.id] = paths
        estimates_by_query[query.id] = estimate_query_paths(spec, statistics, query, paths)["estimates"]
    search = search_indexes(spec, paths_by_query, estimates_by_query)
    digest_payload = {
        "queries": {
            query.id: {
                "path_estimates": estimates_by_query[query.id],
                "selected_path_id": search["query_choices"][query.id]["selected_path_id"],
            }
            for query in spec.queries
        },
        "index_selection": {
            "selected_indexes": search["selected_indexes"],
            "truth_source": search["truth_source"],
        },
    }
    return {
        "paths_by_query": paths_by_query,
        "estimates_by_query": estimates_by_query,
        "index_search": search,
        "estimates_digest": content_digest(digest_payload),
    }


def _statistics_summary(statistics: dict[str, Any]) -> dict[str, Any]:
    tables = {}
    for table_name, table in statistics["tables"].items():
        columns = {}
        for column_name, column in table["columns"].items():
            columns[column_name] = {
                "type": column["type"],
                "hint_selectivity": column["hint_selectivity"],
                "exact_distinct_count": column["exact"]["distinct_count"],
                "sampled_distinct_count": column["sampled"]["distinct_count"],
                "min": column["exact"]["min"],
                "max": column["exact"]["max"],
                "exact_histogram_buckets": len(column["exact"]["histogram"]),
                "sampled_histogram_buckets": len(column["sampled"]["histogram"]),
            }
        tables[table_name] = {
            "row_count": table["row_count"],
            "adaptive_method": table["adaptive_method"],
            "sample_size": table["sample_size"],
            "sample_indices_digest": table["sample_indices_digest"],
            "columns": columns,
        }
    return {
        "stats_version": statistics["stats_version"],
        "stats_caliber_version": statistics["stats_caliber_version"],
        "statistics_model_version": statistics["statistics_model_version"],
        "histogram_model_version": statistics["histogram_model_version"],
        "config": statistics["config"],
        "tables": tables,
    }


def _batch_plan(query_ids: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        return [list(query_ids)]
    return [list(query_ids[index : index + batch_size]) for index in range(0, len(query_ids), batch_size)]


def _query_payload(
    spec: Spec,
    query,
    paths: list[AccessPath],
    estimates: dict[str, dict[str, Any]],
    executions: dict[str, Any],
    actual_filter_rows: dict[str, Any],
    index_choice: dict[str, Any],
    selected_indexes: list[str],
    bias: dict[str, Any],
) -> dict[str, Any]:
    selected_path_id = index_choice["selected_path_id"]
    selected_execution = executions[selected_path_id]
    baseline_path_id = f"scan:{query.tables[0]}"
    baseline_execution = executions[baseline_path_id]
    return {
        "query_id": query.id,
        "query": query.to_dict(),
        "path_specs": [path.to_dict() for path in sorted(paths, key=lambda item: item.id)],
        "path_estimates": estimates,
        "selected_path_id": selected_path_id,
        "selected_estimate": estimates[selected_path_id],
        "selected_path_spec": next(path.to_dict() for path in paths if path.id == selected_path_id),
        "selected_indexes": selected_indexes,
        "index_choice": index_choice,
        "executions": executions,
        "baseline_path_id": baseline_path_id,
        "baseline_digest": baseline_execution["digest"],
        "actual_table_filter_rows": actual_filter_rows,
        "result_match": selected_execution["digest"] == baseline_execution["digest"],
        "all_paths_match_baseline": all(
            execution["digest"] == baseline_execution["digest"] for execution in executions.values()
        ),
        "bias": bias,
    }


def run_pipeline(
    spec: Spec,
    *,
    spec_path: str | Path,
    ledger_path: str | Path,
    db_path: str | Path,
    batch_size: int = 2,
    force_db: bool = False,
    expected_snapshot_version: int | None = None,
    expected_snapshot_digest: str | None = None,
    force_new_version: bool = False,
    crash_after_batches: int | None = None,
) -> dict[str, Any]:
    materialized = materialize_database(spec, db_path, force=force_db)
    connection = sqlite3.connect(db_path)
    try:
        statistics = collect_statistics(spec, connection, spec.stats)
        data_summary = data_digest_summary(spec, db_path)
        stats_digest = statistics_digest(statistics)
        spec_dict = normalize(spec.to_dict())
        weights = spec.cost_weights.to_dict()
        snapshot_digest = compute_snapshot_content_digest(spec_dict, statistics, weights)
        ledger = Ledger(ledger_path)
        try:
            previous_view = ledger.load_current_view()
            previous_storage_errors = (
                ledger.verify_storage_digests(previous_view)
                if previous_view.get("snapshot") is not None
                else []
            )
            previous = ledger.latest_snapshot()
            invalidation = compute_invalidation(previous, spec, statistics)
            storage_replacement = bool(previous_storage_errors)
            prepared = ledger.prepare_snapshot(
                spec=spec_dict,
                spec_digest=spec.digest,
                stats=statistics,
                stats_digest=stats_digest,
                weights=weights,
                content_digest_value=snapshot_digest,
                invalidation=invalidation,
                expected_snapshot_version=expected_snapshot_version,
                expected_snapshot_digest=expected_snapshot_digest,
                force_new_version=force_new_version or storage_replacement,
                force_new_reason=(
                    "snapshot_or_digest_mismatch"
                    if storage_replacement
                    else ("forced_new_version" if force_new_version else None)
                ),
            )
            snapshot = prepared["snapshot"]
            estimates = build_estimates(spec, statistics)
            paths_by_query = estimates["paths_by_query"]
            estimates_by_query = estimates["estimates_by_query"]
            index_search = estimates["index_search"]
            query_ids = [query.id for query in spec.queries]
            batches = _batch_plan(query_ids, batch_size)
            session = ledger.begin_session(snapshot=snapshot, batch_plan=batches)

            execution_by_query: dict[str, dict[str, Any]] = {}
            for query in spec.queries:
                execution_by_query[query.id] = execute_query_paths(
                    spec, connection, query, paths_by_query[query.id]
                )

            query_records = []
            for query in spec.queries:
                execution = execution_by_query[query.id]
                query_records.append(
                    {
                        "query": query,
                        "paths": {path.id: path for path in paths_by_query[query.id]},
                        "estimates": estimates_by_query[query.id],
                        "executions": execution["paths"],
                        "actual_table_filter_rows": execution["actual_table_filter_rows"],
                        "selected_path_id": index_search["query_choices"][query.id]["selected_path_id"],
                    }
                )
            bias = analyze_bias(spec, query_records)
            diagnostics: list[dict[str, Any]] = []
            diagnostics.extend(index_search.get("diagnostics", []))
            diagnostics.extend(bias.get("diagnostics", []))
            for query in spec.queries:
                execution = execution_by_query[query.id]
                baseline_digest = execution["paths"][f"scan:{query.tables[0]}"]["digest"]
                mismatched = [
                    path_id
                    for path_id, path_execution in execution["paths"].items()
                    if path_execution["digest"] != baseline_digest
                ]
                if mismatched:
                    diagnostics.append(
                        {
                            "code": "E_RESULT",
                            "query_id": query.id,
                            "mismatched_paths": sorted(mismatched),
                            "baseline_digest": baseline_digest,
                        }
                    )

            query_payloads = {
                query.id: _query_payload(
                    spec,
                    query,
                    paths_by_query[query.id],
                    estimates_by_query[query.id],
                    execution_by_query[query.id]["paths"],
                    execution_by_query[query.id]["actual_table_filter_rows"],
                    index_search["query_choices"][query.id],
                    index_search["selected_indexes"],
                    bias["queries"][query.id],
                )
                for query in spec.queries
            }
            batch_commits = []
            for batch_no, batch_query_ids in enumerate(batches, start=1):
                payload = {
                    "batch_no": batch_no,
                    "snapshot_version": snapshot["snapshot_version"],
                    "plan_version": snapshot["plan_version"],
                    "statistics_caliber_version": statistics["stats_caliber_version"],
                    "index_search": index_search,
                    "queries": [query_payloads[query_id] for query_id in batch_query_ids],
                }
                commit = ledger.commit_batch(session_id=session["session_id"], batch_no=batch_no, payload=payload)
                batch_commits.append(commit)
                if crash_after_batches is not None and batch_no >= crash_after_batches:
                    os._exit(75)
            completed = ledger.complete_session(session["session_id"])
            view = ledger.load_current_view()
            storage_errors = ledger.verify_storage_digests(view)
            diagnostics.extend(storage_errors)
            report_queries = []
            for query in spec.queries:
                payload = query_payloads[query.id]
                report_queries.append(
                    {
                        "query_id": query.id,
                        "query": payload["query"],
                        "selected_path_id": payload["selected_path_id"],
                        "selected_path": payload["selected_path_spec"],
                        "selected_estimate": payload["selected_estimate"],
                        "paths": [
                            {
                                "path": path_spec,
                                "estimate": payload["path_estimates"][path_spec["id"]],
                                "execution": payload["executions"][path_spec["id"]],
                            }
                            for path_spec in payload["path_specs"]
                        ],
                        "actual_table_filter_rows": payload["actual_table_filter_rows"],
                        "result_match": payload["result_match"],
                        "all_paths_match_baseline": payload["all_paths_match_baseline"],
                        "bias": payload["bias"],
                    }
                )
            report = {
                "platform": PLATFORM_VERSION,
                "status": "OK" if not diagnostics else "OK_WITH_DIAGNOSTICS",
                "spec": {"path": str(Path(spec_path).resolve()), "name": spec.name, "digest": spec.digest},
                "database": materialized | {"summary": data_summary},
                "snapshot": {
                    "snapshot_version": snapshot["snapshot_version"],
                    "plan_version": snapshot["plan_version"],
                    "content_digest": snapshot["content_digest"],
                    "spec_digest": snapshot["spec_digest"],
                    "stats_digest": snapshot["stats_digest"],
                    "status": snapshot["status"],
                    "reused": prepared["reused"],
                    "replaced": prepared["replaced"],
                    "replacement_reason": prepared["replacement_reason"],
                    "previous_snapshot_version": prepared.get("previous_snapshot_version"),
                },
                "invalidation": invalidation,
                "replacement_detected_errors": previous_storage_errors,
                "statistics_summary": _statistics_summary(statistics),
                "index_search": index_search,
                "estimates_digest": estimates["estimates_digest"],
                "queries": report_queries,
                "bias_calibration": bias["calibration"],
                "diagnostics": diagnostics,
                "diagnostic_codes": sorted({item["code"] for item in diagnostics}),
                "session": completed,
                "batches": batch_commits,
                "ledger": {
                    "path": str(Path(ledger_path).resolve()),
                    "view_semantics": view.get("view_semantics"),
                    "history": ledger.history(),
                },
            }
            return report
        finally:
            ledger.close()
    finally:
        connection.close()
