"""Candidate generation, budgeted replay, invalidation-aware conclusions, verification."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .errors import fail
from .query import (
    ColumnRef,
    Literal,
    Predicate,
    Projection,
    Query,
    TableRef,
    is_slow,
    merge_redundant_predicates,
    parse_select,
    replace_query,
)
from .session import Execution, Session


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    query_id: str
    kind: str
    sql: str
    rationale: str
    referenced_indexes: tuple[str, ...]
    index_version: int
    stats_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "query_id": self.query_id,
            "kind": self.kind,
            "sql": self.sql,
            "rationale": self.rationale,
            "referenced_indexes": list(self.referenced_indexes),
            "index_version": self.index_version,
            "stats_version": self.stats_version,
        }


def _candidate(query: Query, kind: str, sql: str, rationale: str, indexes: Sequence[str], session: Session) -> Candidate:
    versions = session.version_info()
    digest = hashlib.sha256(f"{query.query_id}|{kind}|{sql}".encode("utf-8")).hexdigest()
    suffix = f"index_version={versions['index_version']}, stats_version={versions['stats_version']}"
    return Candidate(
        candidate_id="c_" + digest[:32],
        query_id=query.query_id,
        kind=kind,
        sql=sql,
        rationale=f"{rationale}; {suffix}",
        referenced_indexes=tuple(indexes),
        index_version=versions["index_version"],
        stats_version=versions["stats_version"],
    )


def _schema_columns(session: Session) -> dict[str, list[str]]:
    return {
        item["name"]: [column["name"] for column in item["columns"]]
        for item in session.schema_snapshot()
    }


def _resolve_scope(query: Query, column: ColumnRef, schema: Mapping[str, Sequence[str]]) -> str | None:
    if column.table is not None:
        matched = query.table_by_scope(column.table)
        return matched.scope if matched else None
    candidates = [table.scope for table in query.tables if column.name in schema.get(table.name, ())]
    return candidates[0] if len(candidates) == 1 else None


def _predicate_merge_candidate(query: Query, session: Session) -> Candidate | None:
    merged = merge_redundant_predicates(query.predicates)
    if merged is None:
        return None
    predicates, rationale = merged
    candidate_query = replace_query(query, predicates=predicates)
    if candidate_query.canonical_sql == query.canonical_sql:
        return None
    return _candidate(
        query,
        "predicate_merge",
        candidate_query.canonical_sql,
        "谓词合并: " + "; ".join(rationale),
        (),
        session,
    )


def _projection_candidate(
    query: Query, schema: Mapping[str, Sequence[str]], session: Session
) -> Candidate | None:
    star_all = any(item.kind == "star" for item in query.projection)
    star_tables = {item.table for item in query.projection if item.kind == "table_star"}
    if not star_all and not star_tables:
        return None

    used: dict[str, set[str]] = {table.scope: set() for table in query.tables}
    all_columns: Iterable[ColumnRef] = [
        predicate.column for predicate in (*query.join_conditions, *query.predicates)
    ]
    for column in all_columns:
        scope = _resolve_scope(query, column, schema)
        if scope:
            used[scope].add(column.name)
    for predicate in (*query.join_conditions, *query.predicates):
        if isinstance(predicate.right, ColumnRef):
            scope = _resolve_scope(query, predicate.right, schema)
            if scope:
                used[scope].add(predicate.right.name)
    for term in query.order_by:
        scope = _resolve_scope(query, term.column, schema)
        if scope:
            used[scope].add(term.column.name)

    projection: list[Projection] = []
    for table in query.tables:
        if not star_all and table.scope not in star_tables and table.name not in star_tables:
            continue
        available = list(schema.get(table.name, ()))
        selected = [column for column in available if column in used[table.scope]]
        if not selected and available:
            selected = [available[0]]
        for column in selected:
            projection.append(Projection(kind="column", column=ColumnRef(table.scope, column)))
    if not projection:
        return None
    candidate_query = replace_query(query, projection=tuple(projection))
    if candidate_query.canonical_sql == query.canonical_sql:
        return None
    return _candidate(
        query,
        "projection_convergence",
        candidate_query.canonical_sql,
        "投影收敛: 将 * 收敛为连接、过滤、排序实际引用的列",
        (),
        session,
    )


def _index_hint_candidate(query: Query, session: Session) -> Candidate | None:
    schema = _schema_columns(session)
    strengths: dict[str, dict[str, int]] = {table.scope: {} for table in query.tables}

    def add(column: ColumnRef, strength: int) -> None:
        scope = _resolve_scope(query, column, schema)
        if scope is None:
            return
        strengths[scope][column.name] = max(strength, strengths[scope].get(column.name, 0))

    for predicate in query.predicates:
        if predicate.kind == "in":
            add(predicate.column, 2)
        elif predicate.kind == "compare" and predicate.op == "=":
            add(predicate.column, 3)
        elif predicate.kind in {"compare", "between"}:
            add(predicate.column, 1)
    for predicate in query.join_conditions:
        add(predicate.column, 2)
        if isinstance(predicate.right, ColumnRef):
            add(predicate.right, 2)
    for term in query.order_by:
        add(term.column, 1)

    chosen: list[tuple[TableRef, dict[str, Any], int]] = []
    for index in session.list_indexes():
        if not index["columns"]:
            continue
        matches = [table for table in query.tables if table.name == index["table"]]
        if len(matches) != 1:
            continue
        table = matches[0]
        score = strengths.get(table.scope, {}).get(index["columns"][0], 0)
        if score <= 0:
            continue
        chosen.append((table, index, score))

    if not chosen:
        return None
    best_by_scope: dict[str, tuple[TableRef, dict[str, Any], int]] = {}
    for table, index, score in chosen:
        current = best_by_scope.get(table.scope)
        if current is None or score > current[2] or (score == current[2] and index["name"] < current[1]["name"]):
            best_by_scope[table.scope] = (table, index, score)
    ordered = sorted(best_by_scope.values(), key=lambda item: [table.scope for table in query.tables].index(item[0].scope))
    hints = {table.scope: index["name"] for table, index, _ in ordered}
    referenced = [index["name"] for _, index, _ in ordered]
    candidate_query = query.to_sql(hints)
    return _candidate(
        query,
        "index_hint",
        candidate_query,
        "索引提示: 对受限列强制使用索引 " + ", ".join(referenced),
        referenced,
        session,
    )


def _join_order_candidate(query: Query, session: Session) -> Candidate | None:
    if len(query.tables) != 2:
        return None
    candidate_query = replace_query(query, tables=(query.tables[1], query.tables[0]))
    return _candidate(
        query,
        "join_order",
        candidate_query.canonical_sql,
        f"连接顺序调整: {query.tables[0].scope} JOIN {query.tables[1].scope} 调整为 "
        f"{query.tables[1].scope} JOIN {query.tables[0].scope}",
        (),
        session,
    )


def generate_candidates(query: Query, session: Session) -> list[Candidate]:
    schema = _schema_columns(session)
    candidates = [
        _predicate_merge_candidate(query, session),
        _index_hint_candidate(query, session),
        _projection_candidate(query, schema, session),
        _join_order_candidate(query, session),
    ]
    result: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is not None and candidate.candidate_id not in seen:
            result.append(candidate)
            seen.add(candidate.candidate_id)
    return result


def _series_stats(values: Sequence[float]) -> dict[str, float]:
    clean = [float(value) for value in values]
    return {
        "count": float(len(clean)),
        "min_ms": min(clean),
        "median_ms": statistics.median(clean),
        "mean_ms": statistics.fmean(clean),
        "max_ms": max(clean),
        "stdev_ms": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
    }


def _choose_best(
    candidates: Sequence[Mapping[str, Any]], elapsed_by_id: Mapping[str, float]
) -> str | None:
    """Choose the strictly fastest measured valid candidate."""

    if not candidates:
        return None
    chosen = min(
        candidates,
        key=lambda item: (elapsed_by_id[item["candidate_id"]], item["candidate_id"]),
    )
    return str(chosen["candidate_id"])


class DiagnosisEngine:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _resolve_query(self, query_or_sql: Query | str) -> Query:
        if isinstance(query_or_sql, Query):
            return query_or_sql
        if query_or_sql.startswith("q_") and self.session.query_record(query_or_sql) is not None:
            return self.session.query_by_id(query_or_sql)
        return parse_select(query_or_sql)

    def diagnose(
        self,
        query_or_sql: Query | str,
        *,
        timeout_ms: int | None = None,
        budget_ms: int | None = None,
        threshold_ms: int | None = None,
        threshold_rows: int | None = None,
        fail_on_candidate_timeout: bool = False,
        candidate_repeats: int = 3,
    ) -> dict[str, Any]:
        query = self._resolve_query(query_or_sql)
        session_thresholds = self.session.thresholds()
        ms_limit = session_thresholds["threshold_ms"] if threshold_ms is None else threshold_ms
        row_limit = session_thresholds["threshold_rows"] if threshold_rows is None else threshold_rows
        if not isinstance(ms_limit, int) or isinstance(ms_limit, bool) or ms_limit < 0:
            raise fail("E_SESSION", "threshold_ms must be a non-negative integer")
        if not isinstance(row_limit, int) or isinstance(row_limit, bool) or row_limit < 0:
            raise fail("E_SESSION", "threshold_rows must be a non-negative integer")
        timeout = int(self.session.meta_value("replay_timeout_ms")) if timeout_ms is None else timeout_ms
        budget = int(self.session.meta_value("replay_budget_ms")) if budget_ms is None else budget_ms
        if not isinstance(timeout, int) or timeout <= 0:
            raise fail("E_SESSION", "timeout_ms must be a positive integer")
        if not isinstance(budget, int) or budget <= 0:
            raise fail("E_SESSION", "budget_ms must be a positive integer")
        if not isinstance(candidate_repeats, int) or candidate_repeats < 1:
            raise fail("E_SESSION", "candidate_repeats must be a positive integer")

        budget_start_ns = time.perf_counter_ns()
        original = self.session.execute_observed(
            query,
            kind="diagnosis_original",
            timeout_ms=min(timeout, budget),
        )
        if original.outcome == "timeout":
            raise fail(
                "E_TIMEOUT",
                "original query exceeded the declared replay timeout",
                {
                    "query_id": query.query_id,
                    "execution_id": original.execution_id,
                    "elapsed_ms": original.elapsed_ms,
                    "timeout_ms": original.timeout_ms,
                },
            )
        if original.outcome != "success":
            raise fail(
                "E_DATA",
                original.error_message or "original query failed",
                {"query_id": query.query_id, "execution_id": original.execution_id},
            )
        if not is_slow(original.elapsed_ms, original.scanned_rows, ms_limit, row_limit):
            return {
                "slow": False,
                "query_id": query.query_id,
                "canonical_sql": query.canonical_sql,
                "original": original.to_dict(),
                "threshold_ms": ms_limit,
                "threshold_rows": row_limit,
            }

        generated = generate_candidates(query, self.session)
        for candidate in generated:
            self.session.upsert_candidate(candidate.to_dict())

        candidate_results: list[dict[str, Any]] = []
        for candidate in generated:
            elapsed_budget_so_far = (time.perf_counter_ns() - budget_start_ns) / 1_000_000.0
            remaining = budget - elapsed_budget_so_far
            if remaining <= 0:
                self.session.record_candidate_result(
                    candidate.candidate_id,
                    status="budget_skipped",
                    reason="total replay budget exhausted before candidate execution",
                )
                candidate_results.append(
                    {
                        **candidate.to_dict(),
                        "status": "budget_skipped",
                        "elapsed_ms": None,
                        "result_rows": None,
                        "reason": "total replay budget exhausted before candidate execution",
                    }
                )
                continue
            samples: list[Execution] = []
            failure: dict[str, Any] | None = None
            for _ in range(candidate_repeats):
                remaining = budget - (time.perf_counter_ns() - budget_start_ns) / 1_000_000.0
                if remaining <= 0:
                    break
                effective_timeout = max(0.001, min(float(timeout), remaining))
                execution = self.session.execute_observed(
                    query,
                    sql=candidate.sql,
                    kind="candidate",
                    candidate_id=candidate.candidate_id,
                    timeout_ms=effective_timeout,
                )
                if execution.outcome == "timeout":
                    failure = {
                        "status": "timeout",
                        "reason": f"candidate replay timed out after {execution.elapsed_ms:.3f} ms",
                        "execution": execution,
                    }
                    break
                if execution.outcome != "success":
                    failure = {
                        "status": "error",
                        "reason": execution.error_message or "candidate execution failed",
                        "execution": execution,
                    }
                    break
                if execution.result_rows != original.result_rows:
                    failure = {
                        "status": "invalid_equiv",
                        "reason": (
                            f"result row count mismatch: original={original.result_rows}, "
                            f"candidate={execution.result_rows}"
                        ),
                        "execution": execution,
                    }
                    break
                samples.append(execution)
            if failure is not None:
                failed_execution = failure["execution"]
                self.session.record_candidate_result(
                    candidate.candidate_id,
                    status=failure["status"],
                    elapsed_ms=failed_execution.elapsed_ms,
                    result_rows=failed_execution.result_rows,
                    execution_id=failed_execution.execution_id,
                    error_code={
                        "timeout": "E_TIMEOUT",
                        "error": "E_DATA",
                        "invalid_equiv": "E_EQUIV",
                    }.get(failure["status"]),
                    reason=failure["reason"],
                )
                if failure["status"] == "invalid_equiv":
                    self.session.logger.warning(
                        "candidate_invalid_equivalence",
                        failure["reason"],
                        code="E_EQUIV",
                        details={"query_id": query.query_id, "candidate_id": candidate.candidate_id},
                    )
                candidate_results.append(
                    {
                        **candidate.to_dict(),
                        "status": failure["status"],
                        "elapsed_ms": failed_execution.elapsed_ms,
                        "samples_ms": [item.elapsed_ms for item in samples],
                        "result_rows": failed_execution.result_rows,
                        "execution_id": failed_execution.execution_id,
                        "error_code": {
                            "timeout": "E_TIMEOUT",
                            "error": "E_DATA",
                            "invalid_equiv": "E_EQUIV",
                        }.get(failure["status"]),
                        "reason": failure["reason"],
                    }
                )
                if failure["status"] == "timeout" and fail_on_candidate_timeout:
                    raise fail(
                        "E_TIMEOUT",
                        failure["reason"],
                        {
                            "query_id": query.query_id,
                            "candidate_id": candidate.candidate_id,
                            "execution_id": failed_execution.execution_id,
                        },
                    )
                continue
            if not samples:
                reason = "total replay budget exhausted before candidate sample completed"
                self.session.record_candidate_result(
                    candidate.candidate_id,
                    status="budget_skipped",
                    reason=reason,
                )
                candidate_results.append(
                    {
                        **candidate.to_dict(),
                        "status": "budget_skipped",
                        "elapsed_ms": None,
                        "samples_ms": [],
                        "result_rows": None,
                        "error_code": None,
                        "reason": reason,
                    }
                )
                continue
            median_elapsed = statistics.median(item.elapsed_ms for item in samples)
            result_rows = samples[0].result_rows
            self.session.record_candidate_result(
                candidate.candidate_id,
                status="valid",
                elapsed_ms=median_elapsed,
                result_rows=result_rows,
                execution_id=samples[-1].execution_id,
                error_code=None,
                reason=f"row-count equivalent replay median over {len(samples)} samples",
            )
            candidate_results.append(
                {
                    **candidate.to_dict(),
                    "status": "valid",
                    "elapsed_ms": median_elapsed,
                    "samples_ms": [item.elapsed_ms for item in samples],
                    "result_rows": result_rows,
                    "execution_id": samples[-1].execution_id,
                    "error_code": None,
                    "reason": f"row-count equivalent replay median over {len(samples)} samples",
                }
            )

        valid = [item for item in candidate_results if item["status"] == "valid"]
        best_id = _choose_best(valid, {item["candidate_id"]: float(item["elapsed_ms"]) for item in valid})
        best = next((item for item in valid if item["candidate_id"] == best_id), None)
        conclusion = {
            "conclusion_id": "k_" + uuid.uuid4().hex,
            "query_id": query.query_id,
            "original_execution_id": original.execution_id,
            "original_elapsed_ms": original.elapsed_ms,
            "original_scanned_rows": original.scanned_rows,
            "original_result_rows": original.result_rows,
            "candidate_ids": [candidate.candidate_id for candidate in generated],
            "optimal_candidate_id": best["candidate_id"] if best else None,
            "optimal_elapsed_ms": best["elapsed_ms"] if best else None,
            "speedup": (original.elapsed_ms / best["elapsed_ms"]) if best and best["elapsed_ms"] > 0 else None,
            "threshold_ms": ms_limit,
            "threshold_rows": row_limit,
            "data_version": original.data_version,
            "index_version": original.index_version,
            "stats_version": original.stats_version,
        }
        self.session.save_conclusion(conclusion)
        self.session.logger.info(
            "diagnosis_completed",
            "slow query candidate replay completed",
            query_id=query.query_id,
            conclusion_id=conclusion["conclusion_id"],
            optimal_candidate_id=conclusion["optimal_candidate_id"],
            candidate_count=len(generated),
        )
        return {
            "slow": True,
            "query_id": query.query_id,
            "canonical_sql": query.canonical_sql,
            "threshold_ms": ms_limit,
            "threshold_rows": row_limit,
            "original": original.to_dict(),
            "candidates": candidate_results,
            "conclusion": self.session.conclusion(query.query_id),
        }

    # ----------------------------------------------------------- replay validation
    def replay_conclusions(
        self,
        query_ids: Sequence[str] | None = None,
        *,
        repeats: int = 2,
        timeout_ms: int | None = None,
        budget_ms: int | None = None,
        tolerance_ms: float = 5.0,
        tolerance_factor: float = 2.0,
    ) -> dict[str, Any]:
        if not isinstance(repeats, int) or repeats < 1:
            raise fail("E_SESSION", "repeats must be a positive integer")
        timeout = int(self.session.meta_value("replay_timeout_ms")) if timeout_ms is None else timeout_ms
        budget = int(self.session.meta_value("replay_budget_ms")) if budget_ms is None else budget_ms
        if timeout <= 0 or budget <= 0:
            raise fail("E_SESSION", "replay timeout and budget must be positive")
        selected = set(query_ids) if query_ids is not None else None
        conclusions = [
            item for item in self.session.conclusions() if selected is None or item["query_id"] in selected
        ]
        if selected is not None:
            missing_ids = sorted(selected - {item["query_id"] for item in conclusions})
            if missing_ids:
                raise fail(
                    "E_REPLAY",
                    "conclusion is missing for replay",
                    {"unreproducible": [{"query_id": query_id, "reason": "conclusion not found"} for query_id in missing_ids]},
                )

        current = self.session.version_info()
        unreproducible: list[dict[str, Any]] = []
        for conclusion in conclusions:
            query_id = conclusion["query_id"]
            if self.session.query_record(query_id) is None:
                unreproducible.append({"query_id": query_id, "reason": "query identifier is missing"})
                continue
            version_missing = [
                field
                for field in ("data_version", "index_version", "stats_version")
                if conclusion.get(field) is None or int(conclusion[field]) != int(current[field])
            ]
            if version_missing:
                reasons = ", ".join(
                    f"{field}={conclusion.get(field)} (current {current[field]})" for field in version_missing
                )
                unreproducible.append(
                    {"query_id": query_id, "reason": f"recorded {reasons} is not available"}
                )
                continue
            missing_candidates = [
                candidate["candidate_id"] for candidate in conclusion["candidates"] if candidate.get("missing")
            ]
            if missing_candidates:
                unreproducible.append(
                    {
                        "query_id": query_id,
                        "reason": "candidate deleted: " + ", ".join(sorted(missing_candidates)),
                    }
                )
        if unreproducible:
            self.session.logger.error(
                "replay_unreproducible",
                "one or more conclusions cannot be reproduced",
                code="E_REPLAY",
                details={"unreproducible": unreproducible},
            )
            raise fail(
                "E_REPLAY",
                "diagnostic conclusions are not reproducible",
                {
                    "unreproducible": unreproducible,
                    "query_ids": sorted({item["query_id"] for item in unreproducible}),
                },
            )

        budget_start = time.perf_counter_ns()
        results: list[dict[str, Any]] = []
        for conclusion in conclusions:
            query = self.session.query_by_id(conclusion["query_id"])
            original_samples: list[Execution] = []
            for _ in range(repeats):
                remaining = budget - (time.perf_counter_ns() - budget_start) / 1_000_000.0
                if remaining <= 0:
                    raise fail("E_TIMEOUT", "replay budget exhausted", {"query_id": query.query_id})
                execution = self.session.execute_observed(
                    query,
                    kind="replay_original",
                    timeout_ms=min(float(timeout), remaining),
                )
                if execution.outcome == "timeout":
                    raise fail(
                        "E_TIMEOUT",
                        "replay original timed out",
                        {"query_id": query.query_id, "execution_id": execution.execution_id, "elapsed_ms": execution.elapsed_ms},
                    )
                if execution.outcome != "success":
                    raise fail(
                        "E_REPLAY",
                        "replay original failed",
                        {"query_id": query.query_id, "reason": execution.error_message},
                    )
                original_samples.append(execution)
            original_stats = _series_stats([item.elapsed_ms for item in original_samples])
            replay_result_rows = original_samples[0].result_rows
            if any(item.result_rows != replay_result_rows for item in original_samples):
                raise fail(
                    "E_REPLAY",
                    "replayed original returned unstable row counts",
                    {"query_id": query.query_id},
                )

            candidate_reports: list[dict[str, Any]] = []
            replayed_medians: dict[str, float] = {}
            for candidate in conclusion["candidates"]:
                if candidate.get("status") != "valid":
                    candidate_reports.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "kind": candidate.get("kind"),
                            "status": candidate.get("status"),
                            "replayed": False,
                            "stored_elapsed_ms": candidate.get("elapsed_ms"),
                            "within_tolerance": True,
                        }
                    )
                    continue
                samples: list[Execution] = []
                for _ in range(repeats):
                    remaining = budget - (time.perf_counter_ns() - budget_start) / 1_000_000.0
                    if remaining <= 0:
                        raise fail("E_TIMEOUT", "replay budget exhausted", {"query_id": query.query_id})
                    execution = self.session.execute_observed(
                        query,
                        sql=candidate["rewritten_sql"],
                        kind="replay_candidate",
                        candidate_id=candidate["candidate_id"],
                        timeout_ms=min(float(timeout), remaining),
                    )
                    if execution.outcome == "timeout":
                        raise fail(
                            "E_TIMEOUT",
                            "replay candidate timed out",
                            {
                                "query_id": query.query_id,
                                "candidate_id": candidate["candidate_id"],
                                "elapsed_ms": execution.elapsed_ms,
                            },
                        )
                    if execution.outcome != "success":
                        raise fail(
                            "E_REPLAY",
                            "replay candidate failed",
                            {
                                "query_id": query.query_id,
                                "candidate_id": candidate["candidate_id"],
                                "reason": execution.error_message,
                            },
                        )
                    if execution.result_rows != replay_result_rows:
                        raise fail(
                            "E_REPLAY",
                            "candidate result row count changed during replay",
                            {
                                "query_id": query.query_id,
                                "candidate_id": candidate["candidate_id"],
                                "original_rows": replay_result_rows,
                                "candidate_rows": execution.result_rows,
                            },
                        )
                    samples.append(execution)
                stats = _series_stats([item.elapsed_ms for item in samples])
                replayed_medians[candidate["candidate_id"]] = stats["median_ms"]
                stored_elapsed = candidate.get("elapsed_ms")
                tolerance = max(
                    tolerance_ms,
                    float(stored_elapsed) * tolerance_factor if stored_elapsed is not None else 0.0,
                )
                within = stored_elapsed is None or abs(stats["median_ms"] - float(stored_elapsed)) <= tolerance
                candidate_reports.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "kind": candidate.get("kind"),
                        "status": candidate.get("status"),
                        "replayed": True,
                        "stored_elapsed_ms": stored_elapsed,
                        "replay_stats": stats,
                        "row_count_match": True,
                        "within_tolerance": within,
                        "tolerance_ms": tolerance,
                    }
                )

            replay_candidates = [
                item
                for item in conclusion["candidates"]
                if item["candidate_id"] in replayed_medians
            ]
            replay_best = _choose_best(replay_candidates, replayed_medians)
            original_tolerance = max(tolerance_ms, float(conclusion["original_elapsed_ms"]) * tolerance_factor)
            original_within = abs(original_stats["median_ms"] - float(conclusion["original_elapsed_ms"])) <= original_tolerance
            stored_best = conclusion.get("optimal_candidate_id")
            best_match = replay_best == stored_best
            replay_minimum = min(replayed_medians.values()) if replayed_medians else 0.0
            best_equivalence_tolerance = max(tolerance_ms, replay_minimum * 0.50)
            stored_best_replay = replayed_medians.get(stored_best) if stored_best else None
            best_equivalent = best_match or (
                stored_best_replay is not None
                and replay_best is not None
                and abs(stored_best_replay - replayed_medians[replay_best]) <= best_equivalence_tolerance
            )
            replay_timing_match = all(item["within_tolerance"] for item in candidate_reports)
            matched = original_within and best_equivalent and replay_timing_match
            results.append(
                {
                    "query_id": query.query_id,
                    "conclusion_id": conclusion["conclusion_id"],
                    "stored_versions": {
                        "data_version": conclusion["data_version"],
                        "index_version": conclusion["index_version"],
                        "stats_version": conclusion["stats_version"],
                    },
                    "original": {
                        "stored_elapsed_ms": conclusion["original_elapsed_ms"],
                        "stored_result_rows": conclusion["original_result_rows"],
                        "replay_stats": original_stats,
                        "replay_result_rows": replay_result_rows,
                        "within_tolerance": original_within,
                        "tolerance_ms": original_tolerance,
                    },
                    "stored_best_candidate_id": stored_best,
                    "replay_best_candidate_id": replay_best,
                    "best_candidate_match": best_match,
                    "best_candidate_equivalent": best_equivalent,
                    "best_equivalence_tolerance_ms": best_equivalence_tolerance,
                    "candidates": candidate_reports,
                    "matched": matched,
                }
            )

        report = {
            "matched": all(item["matched"] for item in results),
            "repeats": repeats,
            "clock_basis": self.session.meta_value("clock_basis"),
            "versions": current,
            "conclusions": results,
        }
        report["validation_id"] = self.session.save_replay_validation(report)
        if not report["matched"]:
            self.session.logger.warning(
                "replay_mismatch",
                "replay completed with statistical or optimal-candidate mismatch",
                details=report,
            )
        else:
            self.session.logger.info(
                "replay_verified",
                "all snapshot conclusions reproduced",
                validation_id=report["validation_id"],
                conclusion_count=len(results),
            )
        return report
