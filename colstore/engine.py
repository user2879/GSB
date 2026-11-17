"""Scan engine: ties storage, query planning, encoding/memory estimation and
the parallel scheduler together, with versioned stale-result protection."""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .encoding import estimate_encoding, select_encoding
from .errors import DivergenceError
from .model import (
    BlockId,
    BudgetParams,
    Encoding,
    ScanResult,
    ScanStats,
    SegmentDefinition,
    TaskRecord,
)
from .query import build_scan_plan, evaluate, query_id
from .scheduler import Scheduler, TaskSpec
from .segment import (
    BlockMeta,
    LoadedSegment,
    decode_segment_block,
    load_segment,
    write_segment,
)


@dataclass
class ReplacementReport:
    replaced_result_id: Optional[str]
    budget_version: int
    affected_blocks: List[Dict[str, Any]]
    abandoned_tasks: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replaced_result_id": self.replaced_result_id,
            "budget_version": self.budget_version,
            "affected_blocks": list(self.affected_blocks),
            "abandoned_tasks": self.abandoned_tasks,
            "reason": self.reason,
        }


class ScanEngine:
    """Owns segment files, encoding choices, budgets and scan generations."""

    def __init__(self, data_dir: str, budget: Optional[BudgetParams] = None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.budget = budget or BudgetParams()
        Scheduler._validate(self.budget)
        self.budget_version = 1
        self.encoding_selection_version = 1
        self.definitions: Dict[str, SegmentDefinition] = {}
        self.segments: Dict[str, LoadedSegment] = {}
        self.encoding_maps: Dict[str, Dict[str, Encoding]] = {}
        self.encoding_decisions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.current_result: Optional[ScanResult] = None
        self.generations: Dict[str, ScanResult] = {}
        self._active_scheduler: Optional[Scheduler] = None

    # ------------------------------------------------------------------
    # Segment / encoding lifecycle
    # ------------------------------------------------------------------

    def register_definition(self, definition: SegmentDefinition) -> None:
        self.definitions[definition.name] = definition

    def build_segments(
        self, force_encoding: Optional[Encoding] = None
    ) -> Dict[str, Dict[str, Encoding]]:
        """Write every registered definition to a segment file and validate."""
        for name, definition in self.definitions.items():
            path = self.segment_path(name)
            selected: Dict[str, Encoding] = {}
            decisions: Dict[str, Dict[str, Any]] = {}
            bounds = definition.row_group_bounds()
            for col in definition.columns:
                values = definition.values[col.name]
                for row_group, (start, end) in enumerate(bounds):
                    key = f"{row_group}:{col.name}"
                    block_values = list(values[start:end])
                    explicit = definition.block_encoding(row_group, col.name)
                    if force_encoding is not None:
                        chosen = force_encoding
                        decision = {
                            "chosen": chosen.value,
                            "reason": f"forced {chosen.value} for consistency check",
                        }
                    elif explicit is not None:
                        chosen = explicit
                        decision = {
                            "chosen": chosen.value,
                            "reason": "explicit encoding in segment definition",
                        }
                    else:
                        chosen, decision = select_encoding(block_values, col.dtype)
                    selected[key] = chosen
                    decisions[key] = decision
            write_segment(definition, path, selected)
            self.encoding_maps[name] = selected
            self.encoding_decisions[name] = decisions
            self.segments[name] = load_segment(path, definition)
        return self.encoding_maps

    def rebuild_with_encoding(self, force_encoding: Encoding) -> None:
        """Rewrite all segments with one encoding and bump selection version."""
        self.build_segments(force_encoding=force_encoding)
        self.encoding_selection_version += 1
        self._invalidate_results("encoding selection changed")

    def segment_path(self, name: str) -> str:
        return os.path.join(self.data_dir, f"{name}.colseg")

    def encoding_estimate(self, segment: str, row_group: int, column: str) -> Dict[str, Any]:
        definition = self.definitions[segment]
        start, end = definition.row_group_bounds()[row_group]
        values = definition.values[column][start:end]
        col = definition.column(column)
        key = f"{row_group}:{column}"
        chosen = self.encoding_maps[segment][key]
        estimate = estimate_encoding(values, col.dtype, chosen)
        return {
            "block": BlockId(segment, row_group, column).to_dict(),
            "chosen_encoding": chosen.value,
            "selection": self.encoding_decisions[segment][key],
            "estimate": estimate.to_dict(),
        }

    # ------------------------------------------------------------------
    # Budget / version management
    # ------------------------------------------------------------------

    def update_budget(self, budget: BudgetParams) -> ReplacementReport:
        if budget == self.budget:
            return self._empty_report("budget unchanged")
        abandoned = self._abort_active_scan()
        self.budget = budget
        self.budget_version += 1
        report = self._invalidate_results(
            "budget parameters changed", abandoned_override=abandoned
        )
        return report

    def _abort_active_scan(self) -> int:
        if self._active_scheduler is None:
            return 0
        return self._active_scheduler.cancel()

    def _invalidate_results(
        self, reason: str, abandoned_override: Optional[int] = None
    ) -> ReplacementReport:
        affected: List[Dict[str, Any]] = []
        abandoned = 0
        replaced_id: Optional[str] = None
        if self.current_result is not None:
            old = self.current_result
            replaced_id = old.result_id
            old.stats.stale = True
            for task in old.tasks:
                if task.status == "queued":
                    task.status = "abandoned"
                    abandoned += 1
            affected = list(old.plan_summary.get("candidate_blocks", []))
            self.generations[old.result_id] = old
            self.current_result = None
        if abandoned_override is not None:
            abandoned = max(abandoned, abandoned_override)
        return ReplacementReport(
            replaced_result_id=replaced_id,
            budget_version=self.budget_version,
            affected_blocks=affected,
            abandoned_tasks=abandoned,
            reason=reason,
        )

    def _empty_report(self, reason: str) -> ReplacementReport:
        return ReplacementReport(
            replaced_result_id=None,
            budget_version=self.budget_version,
            affected_blocks=[],
            abandoned_tasks=0,
            reason=reason,
        )

    def current_summary(self) -> Dict[str, Any]:
        """Presentation guard: stale generations never enter the summary."""
        if self.current_result is None:
            return {"current": None, "stored_generations": sorted(self.generations)}
        result = self.current_result
        if result.budget_version != self.budget_version:
            return {
                "current": None,
                "blocked": "result budget version differs from current budget version",
                "stored_generations": sorted(self.generations),
            }
        return {"current": result.to_dict(), "stored_generations": sorted(self.generations)}

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(
        self,
        query_raw: Dict[str, Any],
        budget: Optional[BudgetParams] = None,
        label: str = "",
    ) -> ScanResult:
        if budget is not None and budget != self.budget:
            self.update_budget(budget)
        loaded: List[LoadedSegment] = [
            self.segments[name] for name in self.definitions if name in self.segments
        ]
        plan = build_scan_plan(query_raw, loaded)

        # Whole-result replacement: a new generation never merges into an old one.
        abandoned = self._abort_active_scan()
        if self.current_result is not None:
            report = self._invalidate_results("new scan generation")
            abandoned = max(abandoned, report.abandoned_tasks)

        specs: List[TaskSpec] = []
        meta_by_block: Dict[Tuple[str, int, str], BlockMeta] = {}
        for segment in loaded:
            for meta in segment.blocks:
                meta_by_block[
                    (meta.block_id.segment, meta.block_id.row_group, meta.block_id.column)
                ] = meta
        for block_id in plan.candidate_blocks:
            meta = meta_by_block[(block_id.segment, block_id.row_group, block_id.column)]
            definition = self.definitions[block_id.segment]
            col = definition.column(block_id.column)
            estimate = estimate_encoding(
                self._block_values(block_id), col.dtype, meta.encoding
            )
            specs.append(
                TaskSpec(
                    block=block_id,
                    encoding=meta.encoding,
                    memory_bytes=estimate.decoded_peak_bytes,
                    rows=meta.rows,
                )
            )

        # Decoded per-block columns keyed by block coordinate.
        decoded_cache: Dict[Tuple[str, int, str], List[int]] = {}

        def worker(record: TaskRecord) -> int:
            block_id = record.block
            segment = self.segments[block_id.segment]
            meta = segment.block_meta(block_id.row_group, block_id.column)
            values = decode_segment_block(segment, meta)
            decoded_cache[(block_id.segment, block_id.row_group, block_id.column)] = values
            return len(values)

        wall_start = time.monotonic() * 1000.0
        scheduler = Scheduler(self.budget, self.budget_version, worker)
        self._active_scheduler = scheduler
        try:
            task_records = scheduler.run(specs)
        finally:
            self._active_scheduler = None
        elapsed = time.monotonic() * 1000.0 - wall_start

        rows, rows_matched = self._assemble_rows(loaded, plan, decoded_cache)
        result_hash = self._hash_rows(plan.selected_columns, rows)
        stats = ScanStats(
            budget_version=self.budget_version,
            encoding_selection_version=self.encoding_selection_version,
            segment_format_versions={
                name: seg.definition.format_version for name, seg in self.segments.items()
            },
            tasks_total=len(task_records),
            tasks_done=sum(1 for t in task_records if t.status == "done"),
            tasks_abandoned=abandoned,
            blocks_scanned=len(plan.candidate_blocks),
            blocks_skipped_by_pushdown=len(plan.skipped_blocks),
            columns_pruned=list(plan.pruned_columns),
            rows_completed=sum(spec.rows for spec in specs),
            rows_matched=rows_matched,
            peak_memory_bytes=scheduler.peak_memory_bytes,
            total_memory_reserved_ms=round(scheduler.memory_time_integral_ms, 6),
            elapsed_ms=round(elapsed, 6),
        )
        result = ScanResult(
            result_id=uuid.uuid4().hex[:16],
            query_id=plan.query_hash,
            budget_version=self.budget_version,
            encoding_selection_version=self.encoding_selection_version,
            selected_columns=list(plan.selected_columns),
            rows=rows,
            stats=stats,
            plan_summary=plan.summary(),
            tasks=task_records,
            segment_versions={
                name: seg.definition.format_version for name, seg in self.segments.items()
            },
            result_hash=result_hash,
        )
        self.current_result = result
        self.generations[result.result_id] = result
        return result

    def _block_values(self, block_id: BlockId) -> List[int]:
        definition = self.definitions[block_id.segment]
        start, end = definition.row_group_bounds()[block_id.row_group]
        return list(definition.values[block_id.column][start:end])

    def _assemble_rows(
        self,
        loaded: Sequence[LoadedSegment],
        plan,
        decoded_cache: Dict[Tuple[str, int, str], List[int]],
    ) -> Tuple[List[Tuple[Any, ...]], int]:
        rows: List[Tuple[Any, ...]] = []
        matched = 0
        for segment in loaded:
            definition = segment.definition
            available = {col.name for col in definition.columns}
            segment_columns = [c for c in plan.selected_columns if c in available]
            if not segment_columns:
                continue
            for row_group, (start, end) in enumerate(definition.row_group_bounds()):
                # Filter predicates only touch selected columns, all decoded.
                block_columns: Dict[str, List[int]] = {}
                for column in segment_columns:
                    key = (definition.name, row_group, column)
                    if key not in decoded_cache:
                        meta = segment.block_meta(row_group, column)
                        decoded_cache[key] = decode_segment_block(segment, meta)
                    block_columns[column] = decoded_cache[key]
                for local_index in range(end - start):
                    row = {
                        column: block_columns[column][local_index]
                        for column in segment_columns
                    }
                    # If a predicate column is absent in this segment, the
                    # block-level planning never selected those blocks; here we
                    # conservatively skip evaluation rather than KeyError.
                    if not self._predicate_columns_available(plan.predicate, row):
                        continue
                    if evaluate(plan.predicate, row):
                        matched += 1
                        rows.append(
                            tuple(row[column] for column in segment_columns)
                        )
        return rows, matched

    @staticmethod
    def _predicate_columns_available(predicate, row: Dict[str, int]) -> bool:
        from .query import _referenced_columns

        return all(column in row for column in _referenced_columns(predicate))

    @staticmethod
    def _hash_rows(columns: List[str], rows: List[Tuple[Any, ...]]) -> str:
        payload = repr((columns, rows)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------
    # Cross-parallelism / cross-encoding consistency
    # ------------------------------------------------------------------

    def consistency_check(
        self, query_raw: Dict[str, Any], high_parallelism: int
    ) -> Dict[str, Any]:
        """Run the same query on parallel vs serial and all three encodings.

        Results are compared byte-for-byte (column order + row tuples + hash)
        and completed-row statistics are compared.  Any mismatch raises
        ``E_DIVERGE`` annotated with the first differing row id, column and the
        two values, and tagged ``semantic`` vs ``statistical``.
        """
        original_budget = self.budget
        original_encoding_version = self.encoding_selection_version
        paths: Dict[str, ScanResult] = {}
        try:
            serial_budget = BudgetParams(
                max_parallelism=1,
                memory_budget_bytes=original_budget.memory_budget_bytes,
                per_segment_concurrency=1,
            )
            paths["parallelism_1"] = self.scan(query_raw, budget=serial_budget, label="p1")
            parallel_budget = BudgetParams(
                max_parallelism=high_parallelism,
                memory_budget_bytes=original_budget.memory_budget_bytes,
                per_segment_concurrency=original_budget.per_segment_concurrency,
            )
            paths[f"parallelism_{high_parallelism}"] = self.scan(
                query_raw, budget=parallel_budget, label="phigh"
            )
            for encoding in (Encoding.RAW, Encoding.DICTIONARY, Encoding.RLE):
                self.rebuild_with_encoding(encoding)
                paths[f"encoding_{encoding.value}"] = self.scan(
                    query_raw, budget=parallel_budget, label=encoding.value
                )
            self._compare_paths(paths)
        finally:
            # Restore the caller's encoding selection and budget state.
            self.build_segments()
            self.encoding_selection_version = original_encoding_version
            self.budget = original_budget
            self.budget_version += 1
            self._invalidate_results("consistency check state restore")
        return {
            "ok": True,
            "paths": {name: result.result_hash for name, result in paths.items()},
            "rows_matched": next(iter(paths.values())).stats.rows_matched,
            "rows_completed": next(iter(paths.values())).stats.rows_completed,
        }

    def _compare_paths(self, paths: Dict[str, ScanResult]) -> None:
        baseline_name = next(iter(paths))
        baseline = paths[baseline_name]
        for name, result in list(paths.items())[1:]:
            if result.selected_columns != baseline.selected_columns:
                raise DivergenceError(
                    "column order diverges between scan paths",
                    {
                        "kind": "semantic",
                        "first_row_id": 0,
                        "left_path": baseline_name,
                        "right_path": name,
                        "left_columns": baseline.selected_columns,
                        "right_columns": result.selected_columns,
                    },
                )
            first = self._first_row_diff(
                baseline.rows, result.rows, baseline.selected_columns
            )
            if first is not None:
                row_index, column, left_value, right_value = first
                raise DivergenceError(
                    "result rows diverge between scan paths",
                    {
                        "kind": "semantic",
                        "first_row_id": row_index,
                        "column": column,
                        "left_path": baseline_name,
                        "right_path": name,
                        "left_value": left_value,
                        "right_value": right_value,
                    },
                )
            if result.result_hash != baseline.result_hash:
                raise DivergenceError(
                    "row tuples match but result hashes diverge",
                    {
                        "kind": "semantic",
                        "first_row_id": -1,
                        "left_path": baseline_name,
                        "right_path": name,
                    },
                )
            for field_name in ("rows_completed", "rows_matched"):
                left = getattr(baseline.stats, field_name)
                right = getattr(result.stats, field_name)
                if left != right:
                    raise DivergenceError(
                        f"statistic {field_name} diverges between scan paths",
                        {
                            "kind": "statistical",
                            "statistic": field_name,
                            "first_row_id": -1,
                            "column": None,
                            "left_path": baseline_name,
                            "right_path": name,
                            "left_value": left,
                            "right_value": right,
                        },
                    )

    @staticmethod
    def _first_row_diff(
        left: List[Tuple[Any, ...]],
        right: List[Tuple[Any, ...]],
        columns: List[str],
    ) -> Optional[Tuple[int, Optional[str], Any, Any]]:
        if len(left) != len(right):
            return (min(len(left), len(right)), None, len(left), len(right))
        for row_index, (left_row, right_row) in enumerate(zip(left, right)):
            for column, left_value, right_value in zip(
                columns, left_row, right_row
            ):
                if left_value != right_value:
                    return row_index, column, left_value, right_value
        return None
