from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import INT64_MAX, INT64_MIN, atomic_write, canonical_json, digest_value
from .codec import read_column
from .errors import DivergenceError, SpecError, SpecIssue
from .executor import (
    BatchResult,
    QueryResult,
    batch_fingerprint,
    merge_query_result,
)
from .ledger import Ledger
from .planner import BatchPlan, ExecutionPlan, compile_plan
from .scheduler import SchedulerStats, SchedulerTask, run_batches
from .spec import QuerySpec, ScanSpec, parse_spec
from .store import StoreState, build_store, column_path, load_store


SESSION_FORMAT_VERSION = 1


def _query_columns(query: QuerySpec) -> set[str]:
    result = set(query.project) | set(query.predicate_columns) | set(query.group_by)
    result.update(agg.column for agg in query.aggregates if agg.column is not None)
    return result


def _batch_key_dict(key: tuple[str, str, int], plan: BatchPlan | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "query_id": key[0],
        "segment_id": key[1],
        "batch_no": key[2],
    }
    if plan is not None:
        result.update({"start_row": plan.start_row, "end_row": plan.end_row})
    return result


@dataclass
class FullSnapshot:
    store: StoreState
    plan: ExecutionPlan
    batches: dict[tuple[str, str, int], BatchResult]
    queries: dict[str, QueryResult]


class Session:
    """Stateful scan session with deterministic incremental invalidation."""

    def __init__(
        self,
        spec: ScanSpec | str | Path,
        data_root: str | Path,
        ledger_path: str | Path,
        *,
        overrides: Mapping[str, Mapping[str, Mapping[int, int | None]]] | None = None,
        rebuild_store: bool = True,
    ) -> None:
        from .spec import load_spec

        self.spec: ScanSpec = load_spec(spec) if isinstance(spec, (str, Path)) else spec
        self.data_root = Path(data_root)
        self.ledger_path = Path(ledger_path)
        self.overrides: dict[str, dict[str, dict[int, int | None]]] = {
            segment: {
                column: {int(row): value for row, value in rows.items()}
                for column, rows in segment_overrides.items()
            }
            for segment, segment_overrides in (overrides or {}).items()
        }
        if rebuild_store:
            self.store = build_store(self.spec, self.data_root, self.overrides)
        else:
            self.store = load_store(self.data_root)
        self.ledger = Ledger(
            self.ledger_path,
            snapshot_digest=self.store.snapshot_digest,
            spec_digest=self._segment_spec_digest(),
        )
        self.plan: ExecutionPlan = compile_plan(self.spec)
        self.results: dict[tuple[str, str, int], BatchResult] = {}
        self.query_results: dict[str, QueryResult] = {}
        self.batch_fingerprints: dict[tuple[str, str, int], str] = {}
        self.last_scheduler_stats: SchedulerStats | None = None
        self.last_invalidation: dict[str, Any] = {}
        self.last_run_report: dict[str, Any] = {}
        self.recovery_report = None
        self._generation = 0

    def _segment_spec_digest(self) -> str:
        return digest_value(
            {
                "segments": [segment.to_dict() for segment in self.spec.segments],
                "format_version": self.spec.format_version,
            }
        )

    def _new_plan(self, spec: ScanSpec) -> ExecutionPlan:
        return compile_plan(spec)

    def _plan_keys(self, plan: ExecutionPlan, query_ids: set[str] | None = None) -> set[tuple[str, str, int]]:
        return {
            batch.key
            for batch in plan.batches()
            if query_ids is None or batch.query_id in query_ids
        }

    def _invalidate_for_query_changes(
        self,
        old_plan: ExecutionPlan,
        new_plan: ExecutionPlan,
        query_ids: set[str],
        reason: str,
    ) -> tuple[set[tuple[str, str, int]], list[dict[str, Any]]]:
        old_keys = self._plan_keys(old_plan, query_ids)
        new_keys = self._plan_keys(new_plan, query_ids)
        all_keys = old_keys | new_keys
        details = []
        old_batches = {batch.key: batch for batch in old_plan.batches()}
        new_batches = {batch.key: batch for batch in new_plan.batches()}
        for key in sorted(all_keys):
            plan = new_batches.get(key) or old_batches.get(key)
            details.append(
                {
                    **_batch_key_dict(key, plan),
                    "reasons": [reason],
                }
            )
        return all_keys, details

    def _propagation(
        self,
        *,
        kind: str,
        column_updates: list[dict[str, Any]],
        affected_queries: set[str],
        affected_batches: set[tuple[str, str, int]],
    ) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = [{"stage": "edit", "kind": kind}]
        for update in column_updates:
            steps.append(
                {
                    "stage": "column",
                    "segment_id": update["segment_id"],
                    "column": update["column"],
                    "basis": "explicit value replacement invalidates only this column file digest",
                    "old_digest": update.get("old_digest"),
                    "new_digest": update.get("new_digest"),
                }
            )
        for query_id in sorted(affected_queries):
            query = self.spec.query_map()[query_id]
            steps.append(
                {
                    "stage": "query",
                    "query_id": query_id,
                    "basis": "query references an edited column or its projection/predicate/batch partition changed",
                    "referenced_columns": sorted(_query_columns(query)),
                }
            )
        for segment_id in sorted({key[1] for key in affected_batches}):
            batches = sorted(key[2] for key in affected_batches if key[1] == segment_id)
            steps.append(
                {
                    "stage": "batch",
                    "segment_id": segment_id,
                    "batch_numbers": batches,
                    "basis": "edited row falls in the batch or the query plan fingerprint changed",
                }
            )
        if affected_batches:
            steps.append(
                {
                    "stage": "aggregate_and_attribution",
                    "basis": "query aggregate and attribution are rebuilt only from the affected batch results",
                    "affected_batch_count": len(affected_batches),
                }
            )
        return steps

    def _build_invalidation(
        self,
        *,
        kind: str,
        changed: dict[str, Any],
        column_updates: list[dict[str, Any]],
        affected_queries: set[str],
        affected_batches: set[tuple[str, str, int]],
        new_plan: ExecutionPlan | None = None,
        extra_reason: str | None = None,
    ) -> dict[str, Any]:
        details = []
        old_batches = {batch.key: batch for batch in self.plan.batches()}
        if new_plan is not None:
            old_batches.update({batch.key: batch for batch in new_plan.batches()})
        for key in sorted(affected_batches):
            details.append(
                {
                    **_batch_key_dict(key, old_batches.get(key)),
                    "reasons": [extra_reason or kind],
                }
            )
        return {
            "kind": kind,
            "changed": changed,
            "column_invalidations": column_updates,
            "query_invalidations": [
                {"query_id": query_id, "reason": extra_reason or kind}
                for query_id in sorted(affected_queries)
            ],
            "batch_invalidations": details,
            "propagation": self._propagation(
                kind=kind,
                column_updates=column_updates,
                affected_queries=affected_queries,
                affected_batches=affected_batches,
            ),
            "invalid_batch_count": len(affected_batches),
        }

    def _validate_column_edit(
        self,
        segment_id: str,
        column: str,
        row: int,
        value: int | None,
    ) -> None:
        segment_map = self.spec.segment_map()
        if segment_id not in segment_map:
            raise SpecError(SpecIssue(segment_id, "segment_id", "unknown segment"))
        if column not in segment_map[segment_id].column_map():
            raise SpecError(SpecIssue(segment_id, "column", f"unknown column {column!r}"))
        if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < segment_map[segment_id].rows:
            raise SpecError(SpecIssue(segment_id, "row", "row must be an in-range integer"))
        if value is not None and (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < INT64_MIN
            or value > INT64_MAX
        ):
            raise SpecError(SpecIssue(segment_id, "value", "value must be null or a signed int64"))

    def edit_column_value(
        self,
        segment_id: str,
        column: str,
        row: int,
        value: int | None,
        *,
        run: bool = True,
    ) -> dict[str, Any]:
        self._validate_column_edit(segment_id, column, row, value)
        old_segment = self.store.segment_map()[segment_id]
        old_column_state = old_segment.column_map()[column]
        current = read_column(column_path(self.data_root, segment_id, column))
        current_value = None if bool(current.null_mask[row]) else int(current.values[row])
        if current_value == value:
            self.last_invalidation = {
                "kind": "column_value",
                "changed": {"segment_id": segment_id, "column": column, "row": row, "value": value},
                "column_invalidations": [],
                "query_invalidations": [],
                "batch_invalidations": [],
                "propagation": [{"stage": "noop", "basis": "replacement equals the current value"}],
                "invalid_batch_count": 0,
            }
            if run:
                return self.run()
            return self.last_invalidation

        old_plan = self.plan
        self.overrides.setdefault(segment_id, {}).setdefault(column, {})[row] = value
        self.store = build_store(
            self.spec,
            self.data_root,
            self.overrides,
            previous=self.store,
            columns_by_segment={segment_id: {column}},
        )
        new_column_state = self.store.segment_map()[segment_id].column_map()[column]
        affected_queries = {
            query.id
            for query in self.spec.queries
            if segment_id in query.segments and column in _query_columns(query)
        }
        batch_no = row // self.spec.execution.batch_rows
        affected_batches = {
            (query_id, segment_id, batch_no) for query_id in affected_queries
        }
        column_updates = [
            {
                "segment_id": segment_id,
                "column": column,
                "old_digest": old_column_state.column_digest,
                "new_digest": new_column_state.column_digest,
                "affected_queries": sorted(affected_queries),
            }
        ]
        self.last_invalidation = self._build_invalidation(
            kind="column_value",
            changed={"segment_id": segment_id, "column": column, "row": row, "value": value},
            column_updates=column_updates,
            affected_queries=affected_queries,
            affected_batches=affected_batches,
            new_plan=old_plan,
        )
        self._generation += 1
        if run:
            return self.run(extra_invalid=affected_batches)
        return self.last_invalidation

    def _spec_with_query(self, query_id: str, replacement: QuerySpec) -> ScanSpec:
        queries = [replacement if query.id == query_id else query for query in self.spec.queries]
        return parse_spec(
            {
                **self.spec.canonical_dict(),
                "queries": [query.to_dict() for query in queries],
            }
        )

    def _query_edit(
        self,
        query_id: str,
        *,
        project: Iterable[str] | None = None,
        predicate: Any = ...,
        run: bool,
        kind: str,
    ) -> dict[str, Any]:
        query_map = self.spec.query_map()
        if query_id not in query_map:
            raise SpecError(SpecIssue(query_id, "query_id", "unknown query"))
        old_query = query_map[query_id]
        replacement = replace(old_query)
        if project is not None:
            replacement = replace(replacement, project=tuple(sorted(set(project))))
        if predicate is not ...:
            if predicate is None:
                replacement = replace(replacement, predicate=None, predicate_columns=())
            else:
                from .spec import canonicalize_predicate, predicate_columns as pred_columns

                canonical = canonicalize_predicate(predicate, query_id, "predicate")
                replacement = replace(
                    replacement,
                    predicate=canonical,
                    predicate_columns=pred_columns(canonical),
                )
        new_spec = self._spec_with_query(query_id, replacement)
        new_query = new_spec.query_map()[query_id]
        if new_query == old_query:
            self.last_invalidation = {
                "kind": kind,
                "changed": {"query_id": query_id},
                "column_invalidations": [],
                "query_invalidations": [],
                "batch_invalidations": [],
                "propagation": [{"stage": "noop", "basis": "query is unchanged"}],
                "invalid_batch_count": 0,
            }
            if run:
                return self.run()
            return self.last_invalidation
        old_plan = self.plan
        new_plan = self._new_plan(new_spec)
        affected_batches, _ = self._invalidate_for_query_changes(
            old_plan, new_plan, {query_id}, kind
        )
        self.spec = new_spec
        self.plan = new_plan
        self.last_invalidation = self._build_invalidation(
            kind=kind,
            changed={"query_id": query_id},
            column_updates=[],
            affected_queries={query_id},
            affected_batches=affected_batches,
            new_plan=new_plan,
        )
        self._generation += 1
        if run:
            return self.run(extra_invalid=affected_batches)
        return self.last_invalidation

    def edit_projection(
        self, query_id: str, project: Iterable[str], *, run: bool = True
    ) -> dict[str, Any]:
        return self._query_edit(
            query_id, project=project, run=run, kind="projection"
        )

    def edit_predicate(
        self, query_id: str, predicate: Any, *, run: bool = True
    ) -> dict[str, Any]:
        return self._query_edit(
            query_id, predicate=predicate, run=run, kind="predicate"
        )

    def set_batch_rows(self, batch_rows: int, *, run: bool = True) -> dict[str, Any]:
        old_plan = self.plan
        data = self.spec.canonical_dict()
        data["execution"]["batch_rows"] = batch_rows
        new_spec = parse_spec(data)
        new_plan = self._new_plan(new_spec)
        affected_queries = set(self.spec.query_map())
        affected_batches = self._plan_keys(old_plan) | self._plan_keys(new_plan)
        self.spec = new_spec
        self.plan = new_plan
        self.last_invalidation = self._build_invalidation(
            kind="batch_rows",
            changed={"batch_rows": batch_rows},
            column_updates=[],
            affected_queries=affected_queries,
            affected_batches=affected_batches,
            new_plan=new_plan,
        )
        self._generation += 1
        if run:
            return self.run(extra_invalid=affected_batches)
        return self.last_invalidation

    def _build_tasks(self, plan: ExecutionPlan) -> dict[str, SchedulerTask]:
        query_map = plan.query_map()
        tasks: dict[tuple[str, str, int], SchedulerTask] = {}
        for batch in plan.batches():
            query = query_map[batch.query_id].query
            tasks[batch.key] = SchedulerTask(
                plan=batch,
                query=query,
                fingerprint=batch_fingerprint(self.spec, self.store, self.data_root, batch),
                ready=True,
            )
        return tasks

    def run(
        self,
        *,
        force_full: bool = False,
        extra_invalid: Iterable[tuple[str, str, int]] | None = None,
    ) -> dict[str, Any]:
        self.plan = compile_plan(self.spec)
        task_map = self._build_tasks(self.plan)
        expected = [
            (task.plan, task.fingerprint)
            for task in sorted(task_map.values(), key=lambda item: item.key)
        ]
        recovery = self.ledger.reconcile_expected(expected)
        self.recovery_report = recovery
        committed = self.ledger.committed_records()

        invalid = set(extra_invalid or ())
        if force_full:
            invalid |= set(task_map)
        for key, task in task_map.items():
            if key not in self.results:
                invalid.add(key)
            elif self.batch_fingerprints.get(key) != task.fingerprint:
                invalid.add(key)
            record = committed.get(key)
            if record is None or record.get("state") != "committed":
                invalid.add(key)

        recompute = [
            task_map[key] for key in sorted(task_map) if key in invalid
        ]
        reused = [key for key in sorted(task_map) if key not in invalid]
        for task in recompute:
            self.ledger.begin_batch(task.plan, task.fingerprint)

        new_results, stats = run_batches(
            self.spec,
            self.store,
            self.data_root,
            recompute,
            on_commit=self.ledger.commit_batch,
        )
        self.last_scheduler_stats = stats
        recovery_comparisons = 0
        for result in new_results:
            old_record = committed.get(result.key)
            if (
                old_record is not None
                and old_record.get("state") == "committed"
                and old_record.get("fingerprint") == result.fingerprint
            ):
                self.ledger.verify_result_against_record(result, old_record)
                recovery_comparisons += 1
            self.results[result.key] = result
            self.batch_fingerprints[result.key] = task_map[result.key].fingerprint

        self.results = {
            key: result for key, result in self.results.items() if key in task_map
        }
        self.batch_fingerprints = {
            key: fingerprint
            for key, fingerprint in self.batch_fingerprints.items()
            if key in task_map
        }

        query_results: dict[str, QueryResult] = {}
        for query in self.spec.queries:
            query_batches = [
                self.results[key]
                for key in sorted(self.results)
                if key[0] == query.id
            ]
            query_results[query.id] = merge_query_result(self.spec, query, query_batches)
        self.query_results = query_results

        for result in self.results.values():
            if result.key in invalid:
                self.ledger.verify_results([result])

        report = {
            "format_version": 1,
            "spec_digest": self.spec.digest,
            "store": self.store.to_dict(),
            "columns": [
                {
                    "segment_id": segment.id,
                    "column": column.name,
                    "encoding": column.encoding,
                    "non_null_count": column.non_null_count,
                    "column_digest": column.column_digest,
                }
                for segment in self.store.segments
                for column in segment.columns
            ],
            "plan": self.plan.to_dict(include_batches=False),
            "execution": self.spec.execution.to_dict(),
            "invalidation": self.last_invalidation,
            "incremental": {
                "recomputed_batches": [
                    _batch_key_dict(result.key, result.plan) for result in new_results
                ],
                "reused_batches": [_batch_key_dict(key) for key in reused],
                "recomputed_batch_count": len(new_results),
                "reused_batch_count": len(reused),
                "recovery_comparisons": recovery_comparisons,
            },
            "scheduler": stats.to_dict(),
            "ledger": {
                **self.ledger.open_result.to_dict(),
                "content_digest": self.ledger.current_content_digest(),
                "committed_batch_count": len(
                    self.ledger.committed_records()
                ),
            },
            "recovery": recovery.to_dict(),
            "queries": [
                query_results[query_id].to_dict()
                for query_id in sorted(query_results)
            ],
            "batch_results": [
                self.results[key].to_dict() for key in sorted(self.results)
            ],
        }
        self.last_run_report = report
        return report

    def state_dict(self) -> dict[str, Any]:
        body = {
            "format_version": SESSION_FORMAT_VERSION,
            "spec": self.spec.canonical_dict(),
            "overrides": {
                segment: {
                    column: {str(row): value for row, value in sorted(rows.items())}
                    for column, rows in sorted(columns.items())
                }
                for segment, columns in sorted(self.overrides.items())
            },
        }
        return {**body, "content_digest": digest_value(body)}

    def save_state(self, path: str | Path) -> None:
        atomic_write(path, (canonical_json(self.state_dict()) + "\n").encode("utf-8"))

    def full_rebuild_snapshot(self) -> FullSnapshot:
        with tempfile.TemporaryDirectory(prefix="vscan-full-") as temp:
            root = Path(temp)
            store = build_store(self.spec, root, self.overrides)
            plan = compile_plan(self.spec)
            tasks = []
            query_map = plan.query_map()
            for batch in plan.batches():
                tasks.append(
                    SchedulerTask(
                        plan=batch,
                        query=query_map[batch.query_id].query,
                        fingerprint=batch_fingerprint(self.spec, store, root, batch),
                        ready=True,
                    )
                )
            batch_results, _ = run_batches(self.spec, store, root, tasks)
            by_key = {result.key: result for result in batch_results}
            queries = {
                query.id: merge_query_result(
                    self.spec,
                    query,
                    [by_key[key] for key in sorted(by_key) if key[0] == query.id],
                )
                for query in self.spec.queries
            }
            return FullSnapshot(store, plan, by_key, queries)

    def verify_against_full_rebuild(self) -> dict[str, Any]:
        full = self.full_rebuild_snapshot()
        left_segments = self.store.segment_map()
        right_segments = full.store.segment_map()
        for segment_id in sorted(set(left_segments) | set(right_segments)):
            if segment_id not in left_segments or segment_id not in right_segments:
                raise DivergenceError(
                    "segment", segment_id=segment_id, expected="present", actual="missing"
                )
            left_columns = left_segments[segment_id].column_map()
            right_columns = right_segments[segment_id].column_map()
            for column_name in sorted(set(left_columns) | set(right_columns)):
                if column_name not in left_columns or column_name not in right_columns:
                    raise DivergenceError(
                        "column",
                        segment_id=segment_id,
                        column=column_name,
                        expected="present",
                        actual="missing",
                    )
                left = left_columns[column_name]
                right = right_columns[column_name]
                for field in ("encoding", "non_null_count", "column_digest", "file_digest"):
                    if getattr(left, field) != getattr(right, field):
                        raise DivergenceError(
                            f"column.{field}",
                            segment_id=segment_id,
                            column=column_name,
                            expected=getattr(left, field),
                            actual=getattr(right, field),
                        )

        left_batches = self.results
        right_batches = full.batches
        for key in sorted(set(left_batches) | set(right_batches)):
            if key not in left_batches or key not in right_batches:
                raise DivergenceError(
                    "batch",
                    segment_id=key[1],
                    batch_no=key[2],
                    expected="present" if key in left_batches else "missing",
                    actual="present" if key in right_batches else "missing",
                )
            left = left_batches[key]
            right = right_batches[key]
            batch_fields = {
                "start_row": (left.plan.start_row, right.plan.start_row),
                "end_row": (left.plan.end_row, right.plan.end_row),
                "plan_digest": (left.plan.plan_digest, right.plan.plan_digest),
                "passed_rows": (left.computation.passed_rows, right.computation.passed_rows),
                "decoded_bytes": (
                    left.computation.actual_decoded_bytes,
                    right.computation.actual_decoded_bytes,
                ),
                "aggregate_digest": (
                    left.computation.aggregate_digest,
                    right.computation.aggregate_digest,
                ),
                "aggregate_json": (
                    canonical_json(left.computation.aggregate_payload()),
                    canonical_json(right.computation.aggregate_payload()),
                ),
                "attribution": (
                    canonical_json(left.computation.attribution),
                    canonical_json(right.computation.attribution),
                ),
            }
            for field, (left_value, right_value) in batch_fields.items():
                if left_value != right_value:
                    column = next(iter(left.plan.decode_columns), None)
                    raise DivergenceError(
                        f"batch.{field}",
                        segment_id=key[1],
                        column=column,
                        batch_no=key[2],
                        expected=left_value,
                        actual=right_value,
                    )

        for query_id in sorted(set(self.query_results) | set(full.queries)):
            left = self.query_results.get(query_id)
            right = full.queries.get(query_id)
            if left is None or right is None:
                raise DivergenceError(
                    "query", expected="present" if left else "missing", actual="present" if right else "missing"
                )
            if left.aggregate_digest != right.aggregate_digest:
                raise DivergenceError(
                    "query.aggregate_digest",
                    expected=left.aggregate_digest,
                    actual=right.aggregate_digest,
                )
            left_attribution = left.attribution.to_dict(self.spec.execution.attribution_tolerance_ppm)
            right_attribution = right.attribution.to_dict(self.spec.execution.attribution_tolerance_ppm)
            if canonical_json(left_attribution) != canonical_json(right_attribution):
                raise DivergenceError(
                    "query.attribution",
                    expected=left_attribution,
                    actual=right_attribution,
                )
        return {
            "consistent": True,
            "segments_checked": len(self.store.segments),
            "columns_checked": sum(len(segment.columns) for segment in self.store.segments),
            "batches_checked": len(self.results),
            "queries_checked": len(self.query_results),
            "fields": [
                "column.encoding",
                "column.non_null_count",
                "column.column_digest",
                "column.file_digest",
                "batch.start_row",
                "batch.end_row",
                "batch.plan_digest",
                "batch.passed_rows",
                "batch.decoded_bytes",
                "batch.aggregate_digest",
                "batch.aggregate_json",
                "batch.attribution",
                "query.aggregate_digest",
                "query.attribution",
            ],
        }

    def close(self) -> None:
        self.ledger.close()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
