from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np

from .config import SpMMConfig
from .csr import CSRMatrix
from .diagnostics import (
    Diagnostic,
    E_BLOCK_COMPUTE_FAILED,
    E_CONFIG_INVALID,
    E_CSR_INVALID,
    E_MATRIX_NONFINITE,
    E_RECOMPUTE_PENDING,
    sort_diagnostics,
)
from .fingerprint import array_bytes, digest_json
from .graph import BlockDependency, derive_dependency
from .hot import HotBlockTable
from .kernels import (
    BlockResult,
    compute_block,
    row_blocks,
    validate_dense,
)


class InputValidationError(ValueError):
    def __init__(self, diagnostics: Sequence[Diagnostic]) -> None:
        self.diagnostics = sort_diagnostics(diagnostics)
        message = "; ".join(item.code for item in self.diagnostics)
        super().__init__(message or "input validation failed")


def validate_bundle(
    csr: CSRMatrix,
    b_matrix: np.ndarray,
    c_matrix: np.ndarray,
    config: SpMMConfig,
) -> list[Diagnostic]:
    failures = list(config.validate())
    failures.extend(csr.validate())
    b = np.asarray(b_matrix)
    c = np.asarray(c_matrix)
    if b.ndim != 2:
        failures.append(
            Diagnostic(
                code=E_CONFIG_INVALID,
                phase="dense_validation",
                message="B must be a two-dimensional dense matrix",
                details={"shape": list(b.shape)},
            )
        )
        return sort_diagnostics(failures)
    if tuple(b.shape[:1]) != (csr.cols,):
        failures.append(
            Diagnostic(
                code=E_CSR_INVALID,
                phase="dense_validation",
                message="B row count must equal CSR column count",
                details={"b_rows": int(b.shape[0]), "csr_cols": csr.cols},
            )
        )
    for name, matrix, expected in (
        ("B", b, (csr.cols, int(b.shape[1]))),
        ("C", c, (csr.rows, int(b.shape[1]))),
    ):
        for message in validate_dense(matrix, expected, name):
            code = E_MATRIX_NONFINITE if "non-finite" in message else E_CONFIG_INVALID
            failures.append(
                Diagnostic(
                    code=code,
                    phase="dense_validation",
                    message=message,
                    details={"matrix": name, "shape": list(matrix.shape)},
                )
            )
    return sort_diagnostics(failures)


@dataclass
class RunReport:
    run_id: int
    targets: tuple[int, ...]
    recomputed: tuple[int, ...]
    reused: tuple[int, ...]
    pending: tuple[int, ...]
    budget_deductions: list[dict[str, Any]]
    failures: list[Diagnostic]
    block_fingerprints: dict[int, str]
    overflow_records: list[dict[str, Any]]
    task_keys: list[dict[str, Any]]
    max_workers_used: int
    complete: bool
    output_fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "targets": list(self.targets),
            "recomputed": list(self.recomputed),
            "reused": list(self.reused),
            "pending": list(self.pending),
            "budget_deductions": self.budget_deductions,
            "failures": [item.to_dict() for item in self.failures],
            "block_fingerprints": {str(key): value for key, value in sorted(self.block_fingerprints.items())},
            "overflow_records": self.overflow_records,
            "task_keys": self.task_keys,
            "max_workers_used": self.max_workers_used,
            "complete": self.complete,
            "output_fingerprint": self.output_fingerprint,
        }


@dataclass
class FullRebuildResult:
    config: SpMMConfig
    blocks: dict[int, BlockResult]
    dependencies: dict[int, BlockDependency]
    c_matrix: np.ndarray
    row_fingerprints: tuple[str, ...]
    overflow_records: list[dict[str, Any]]
    hot_table: HotBlockTable
    budget_ledger: list[dict[str, Any]]
    output_fingerprint: str
    task_keys: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "block_fingerprints": {
                str(block_id): block.output_fingerprint
                for block_id, block in sorted(self.blocks.items())
            },
            "row_fingerprints": list(self.row_fingerprints),
            "overflow_records": self.overflow_records,
            "hot_table": self.hot_table.to_dict(),
            "budget_ledger": self.budget_ledger,
            "output_fingerprint": self.output_fingerprint,
            "task_keys": self.task_keys,
        }


class IncrementalEngine:
    """Deterministic block-cached CSR SpMM engine with audit-oriented diagnostics."""

    def __init__(
        self,
        csr: CSRMatrix,
        b_matrix: np.ndarray,
        c_matrix: np.ndarray,
        config: SpMMConfig,
        *,
        bootstrap_diagnostics: Sequence[Diagnostic] = (),
    ) -> None:
        failures = validate_bundle(csr, b_matrix, c_matrix, config)
        if failures:
            raise InputValidationError(failures)
        self.csr = csr
        self.b_matrix = np.asarray(b_matrix, dtype=np.float64).copy()
        self.c_matrix = np.asarray(c_matrix, dtype=np.float64).copy()
        self.config = config
        self.hot = HotBlockTable(config.hot_capacity)
        self.output_blocks: dict[int, BlockResult] = {}
        self.dependencies: dict[int, BlockDependency] = {}
        self.invalid_blocks: set[int] = set()
        self.pending: set[int] = set()
        self.run_id = 0
        self.edit_version = 0
        self.compute_invocations: Counter[int] = Counter()
        self.budget_ledger: list[dict[str, Any]] = []
        self.invalidation_events: list[dict[str, Any]] = []
        self.recompute_events: list[dict[str, Any]] = []
        self.hot_event_log: list[dict[str, Any]] = []
        self.bootstrap_diagnostics = sort_diagnostics(bootstrap_diagnostics)
        self._refresh_topology()
        self.invalid_blocks = set(self.dependencies)
        self.pending = set(self.dependencies)

    @property
    def n_columns(self) -> int:
        return int(self.b_matrix.shape[1])

    @property
    def block_ranges(self) -> list[tuple[int, int]]:
        return row_blocks(self.csr.rows, self.config.row_block_size)

    @property
    def current_block_ids(self) -> tuple[int, ...]:
        return tuple(range(len(self.block_ranges)))

    def _refresh_topology(self) -> None:
        ranges = self.block_ranges
        current_ids = set(range(len(ranges)))
        self.output_blocks = {
            block_id: block
            for block_id, block in self.output_blocks.items()
            if block_id in current_ids
            and (block.row_start, block.row_end) == ranges[block_id]
        }
        self.dependencies = {
            block_id: derive_dependency(
                self.csr,
                self.b_matrix,
                self.c_matrix,
                self.config,
                block_id,
                row_start,
                row_end,
            )
            for block_id, (row_start, row_end) in enumerate(ranges)
        }
        for block_id, block in self.output_blocks.items():
            if block.dependency_fingerprint == self.dependencies[block_id].input_fingerprint:
                self.dependencies[block_id] = self.dependencies[block_id].with_output(
                    block.output_fingerprint
                )
        self.hot.prune(current_ids)
        self.hot_event_log.append({"type": "prune", "valid_blocks": sorted(current_ids)})
        self.invalid_blocks.intersection_update(current_ids)
        self.pending.intersection_update(current_ids)

    def _record_invalidation(self, block_ids: Iterable[int], reason: str) -> None:
        ids = tuple(sorted(set(block_ids)))
        if not ids:
            return
        self.edit_version += 1
        for block_id in ids:
            self.hot.invalidate(block_id, reason)
            self.hot_event_log.append(
                {
                    "type": "invalidate",
                    "block_id": block_id,
                    "reason": reason,
                }
            )
            self.invalid_blocks.add(block_id)
            self.pending.add(block_id)
        self.invalidation_events.append(
            {"edit_version": self.edit_version, "reason": reason, "blocks": list(ids)}
        )

    def _invalidate_all(self, reason: str) -> None:
        self._record_invalidation(self.current_block_ids, reason)

    def replace_row(self, row: int, columns: Iterable[int], values: Iterable[float]) -> None:
        block_id = row // self.config.row_block_size
        updated = self.csr.replace_row(row, columns, values)
        failures = updated.validate()
        if failures:
            raise InputValidationError(failures)
        self.csr = updated
        self._refresh_topology()
        self._record_invalidation((block_id,), "replace_row")

    def delete_row(self, row: int) -> None:
        first_block = row // self.config.row_block_size
        self.csr = self.csr.delete_row(row)
        self.c_matrix = np.delete(self.c_matrix, row, axis=0)
        self._refresh_topology()
        affected = tuple(range(first_block, len(self.block_ranges)))
        self._record_invalidation(affected, "delete_row")

    def append_row(
        self,
        columns: Iterable[int],
        values: Iterable[float],
        c_init_row: Iterable[float] | None = None,
    ) -> None:
        updated = self.csr.append_row(columns, values)
        failures = updated.validate()
        if failures:
            raise InputValidationError(failures)
        if c_init_row is None:
            c_row = np.zeros(self.n_columns, dtype=np.float64)
        else:
            c_row = np.asarray(list(c_init_row), dtype=np.float64)
            if c_row.shape != (self.n_columns,):
                raise ValueError("c_init_row shape does not match B columns")
        if not np.all(np.isfinite(c_row)):
            raise InputValidationError(
                [
                    Diagnostic(
                        code=E_MATRIX_NONFINITE,
                        phase="dense_validation",
                        message="c_init_row contains a non-finite value",
                        details={"matrix": "C"},
                    )
                ]
            )
        self.csr = updated
        self.c_matrix = np.vstack((self.c_matrix, c_row))
        self._refresh_topology()
        new_block = (self.csr.rows - 1) // self.config.row_block_size
        self._record_invalidation((new_block,), "append_row")

    def set_config(self, new_config: SpMMConfig) -> None:
        failures = new_config.validate()
        if failures:
            raise InputValidationError(failures)
        old = self.config
        input_changed = (
            old.alpha != new_config.alpha
            or old.beta != new_config.beta
            or old.nnz_chunk_size != new_config.nnz_chunk_size
            or old.order != new_config.order
            or old.precision_mode != new_config.precision_mode
            or old.fractional_bits != new_config.fractional_bits
            or old.accumulator_bits != new_config.accumulator_bits
            or old.row_block_size != new_config.row_block_size
        )
        topology_changed = old.row_block_size != new_config.row_block_size
        capacity_changed = old.hot_capacity != new_config.hot_capacity
        if capacity_changed:
            self.hot = HotBlockTable(new_config.hot_capacity, self.hot.entries())
            self.hot_event_log.append(
                {
                    "type": "resize",
                    "capacity": new_config.hot_capacity,
                    "entries": self.hot.to_dict()["entries"],
                }
            )
        self.config = new_config
        if topology_changed:
            self.hot.clear()
            self.hot_event_log.append({"type": "clear", "reason": "row_block_size"})
        self._refresh_topology()
        if input_changed:
            reason = "config_topology" if topology_changed else "config_fingerprint"
            self._invalidate_all(reason)

    def set_alpha_beta(self, alpha: float, beta: float) -> None:
        self.set_config(self.config.with_changes(alpha=float(alpha), beta=float(beta)))

    def set_precision(self, mode: str, fractional_bits: int, accumulator_bits: int) -> None:
        self.set_config(
            self.config.with_changes(
                precision_mode=mode,
                fractional_bits=int(fractional_bits),
                accumulator_bits=int(accumulator_bits),
            )
        )

    def set_order(self, order: str, nnz_chunk_size: int | None = None) -> None:
        changes: dict[str, Any] = {"order": order}
        if nnz_chunk_size is not None:
            changes["nnz_chunk_size"] = int(nnz_chunk_size)
        self.set_config(self.config.with_changes(**changes))

    def set_block_size(self, row_block_size: int, nnz_chunk_size: int | None = None) -> None:
        changes: dict[str, Any] = {"row_block_size": int(row_block_size)}
        if nnz_chunk_size is not None:
            changes["nnz_chunk_size"] = int(nnz_chunk_size)
        self.set_config(self.config.with_changes(**changes))

    def set_b_matrix(self, b_matrix: np.ndarray) -> None:
        b = np.asarray(b_matrix, dtype=np.float64)
        failures = validate_dense(b, (self.csr.cols, self.n_columns), "B")
        if failures:
            raise InputValidationError(
                [
                    Diagnostic(
                        code=E_MATRIX_NONFINITE if "non-finite" in message else E_CONFIG_INVALID,
                        phase="dense_validation",
                        message=message,
                        details={"matrix": "B"},
                    )
                    for message in failures
                ]
            )
        self.b_matrix = b.copy()
        self._refresh_topology()
        self._invalidate_all("replace_b")

    def _block_cost(self, block_id: int) -> int:
        row_start, row_end = self.block_ranges[block_id]
        return sum(self.csr.row_nnz(row) for row in range(row_start, row_end))

    def _prepare_targets(self) -> tuple[list[int], list[int], list[Diagnostic]]:
        failures: list[Diagnostic] = []
        current = set(self.current_block_ids)
        targets = (self.invalid_blocks | self.pending) & current
        for block_id in self.current_block_ids:
            if block_id not in self.output_blocks and block_id not in targets:
                targets.add(block_id)
        for block_id in sorted(targets):
            cached = self.output_blocks.get(block_id)
            dependency = self.dependencies[block_id]
            if cached is not None and cached.dependency_fingerprint != dependency.input_fingerprint:
                targets.add(block_id)
        for block_id in targets:
            self.output_blocks.pop(block_id, None)
        self.invalid_blocks = set(targets)
        self.pending = set(targets)
        return sorted(targets), sorted(current - targets), failures

    def _schedule(
        self, targets: Sequence[int], budget: int
    ) -> tuple[list[int], list[int], list[dict[str, Any]]]:
        scheduled: list[int] = []
        deductions: list[dict[str, Any]] = []
        remaining = int(budget)
        for index, block_id in enumerate(targets):
            cost = self._block_cost(block_id)
            if cost > remaining:
                return scheduled, list(targets[index:]), deductions
            before = remaining
            remaining -= cost
            sequence = len(self.budget_ledger) + 1
            event = {
                "sequence": sequence,
                "run_id": self.run_id + 1,
                "block_id": block_id,
                "cost": cost,
                "budget_before": before,
                "budget_after": remaining,
            }
            self.budget_ledger.append(event)
            deductions.append(dict(event))
            scheduled.append(block_id)
        return scheduled, [], deductions

    def run(self, budget: int | None = None) -> RunReport:
        if budget is None:
            budget = self.config.recompute_budget
        if int(budget) < 0:
            raise ValueError("budget must be non-negative")
        self.run_id += 1
        self._refresh_topology()
        targets, reused, failures = self._prepare_targets()
        scheduled, deferred, deductions = self._schedule(targets, int(budget))
        task_keys = [
            {"block_id": block_id, "stage": "compute", "sort_key": [block_id, "compute"]}
            for block_id in scheduled
        ]
        worker_count = min(self.config.max_workers, len(scheduled))
        successes: list[int] = []
        computed: dict[int, BlockResult] = {}

        if scheduled:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="spmm-block") as pool:
                futures = []
                for block_id in scheduled:
                    row_start, row_end = self.block_ranges[block_id]
                    future = pool.submit(
                        compute_block,
                        self.csr,
                        self.b_matrix,
                        self.c_matrix,
                        self.config,
                        block_id,
                        row_start,
                        row_end,
                    )
                    futures.append((block_id, future))
                for block_id, future in futures:
                    try:
                        result = future.result()
                    except Exception as error:  # Isolate a single block's failure.
                        failures.append(
                            Diagnostic(
                                code=E_BLOCK_COMPUTE_FAILED,
                                phase="block_compute",
                                message=f"block {block_id} failed: {type(error).__name__}: {error}",
                                details={"block_id": block_id},
                            )
                        )
                        continue
                    computed[block_id] = result
                    successes.append(block_id)
                    self.compute_invocations[block_id] += 1

        self.output_blocks.update(computed)
        for block_id, result in computed.items():
            self.dependencies[block_id] = self.dependencies[block_id].with_output(
                result.output_fingerprint
            )
            cost = self._block_cost(block_id)
            self.hot.add_recompute(block_id, cost)
            self.hot_event_log.append(
                {
                    "type": "recompute",
                    "block_id": block_id,
                    "nnz": cost,
                }
            )
            self.recompute_events.append(
                {
                    "run_id": self.run_id,
                    "block_id": block_id,
                    "nnz": cost,
                    "output_fingerprint": result.output_fingerprint,
                }
            )

        failed_ids = set(scheduled) - set(successes)
        pending = sorted(set(deferred) | failed_ids)
        self.pending = set(pending)
        self.invalid_blocks = set(pending)
        for block_id in pending:
            failures.append(
                Diagnostic(
                    code=E_RECOMPUTE_PENDING,
                    phase="budget_schedule",
                    message=f"block {block_id} remains pending for a later recompute budget",
                    details={
                        "block_id": block_id,
                        "cost": self._block_cost(block_id),
                        "batch_budget": int(budget),
                    },
                )
            )
        failures = sort_diagnostics(self.bootstrap_diagnostics + failures)

        block_fingerprints = {
            block_id: block.output_fingerprint
            for block_id, block in sorted(self.output_blocks.items())
        }
        overflow_records = sorted(
            (
                block.first_overflow.to_dict()
                for block in self.output_blocks.values()
                if block.first_overflow is not None
            ),
            key=lambda item: (item["block_id"], item["row"], item["column"]),
        )
        complete = len(self.output_blocks) == len(self.block_ranges)
        output_fingerprint = (
            digest_json(list(block_fingerprints.values())) if complete else None
        )
        return RunReport(
            run_id=self.run_id,
            targets=tuple(targets),
            recomputed=tuple(sorted(successes)),
            reused=tuple(reused),
            pending=tuple(pending),
            budget_deductions=deductions,
            failures=failures,
            block_fingerprints=block_fingerprints,
            overflow_records=overflow_records,
            task_keys=task_keys,
            max_workers_used=worker_count,
            complete=complete,
            output_fingerprint=output_fingerprint,
        )

    def run_until_complete(self, maximum_rounds: int = 10_000) -> list[RunReport]:
        reports: list[RunReport] = []
        for _ in range(maximum_rounds):
            report = self.run(self.config.recompute_budget)
            reports.append(report)
            if not report.pending:
                return reports
            if not report.recomputed:
                raise RuntimeError(
                    "recompute budget cannot fund the next pending block; increase recompute_budget"
                )
        raise RuntimeError("recompute budget did not converge")

    @property
    def row_fingerprints(self) -> tuple[str, ...]:
        rows: list[str] = []
        for block_id in self.current_block_ids:
            block = self.output_blocks.get(block_id)
            if block is None:
                raise RuntimeError("result is incomplete")
            rows.extend(block.row_fingerprints)
        return tuple(rows)

    @property
    def overflow_records(self) -> list[dict[str, Any]]:
        return sorted(
            (
                block.first_overflow.to_dict()
                for block in self.output_blocks.values()
                if block.first_overflow is not None
            ),
            key=lambda item: (item["block_id"], item["row"], item["column"]),
        )

    def materialize_c(self, *, require_complete: bool = True) -> np.ndarray:
        if require_complete and len(self.output_blocks) != len(self.block_ranges):
            raise RuntimeError("result is incomplete; drain the pending budget first")
        result = np.full((self.csr.rows, self.n_columns), np.nan, dtype=np.float32)
        for block_id, (row_start, row_end) in enumerate(self.block_ranges):
            block = self.output_blocks.get(block_id)
            if block is not None:
                result[row_start:row_end] = block.values
        return result

    def output_fingerprint(self) -> str:
        return digest_json(
            [self.output_blocks[block_id].output_fingerprint for block_id in self.current_block_ids]
        )

    def audit_manifest(self) -> dict[str, Any]:
        return {
            "edit_version": self.edit_version,
            "invalidation_events": [dict(item) for item in self.invalidation_events],
            "recompute_events": [dict(item) for item in self.recompute_events],
            "budget_ledger": [dict(item) for item in self.budget_ledger],
            "hot_table": self.hot.to_dict(),
            "hot_event_log": [dict(item) for item in self.hot_event_log],
        }


def _parallel_compute_all(
    csr: CSRMatrix,
    b_matrix: np.ndarray,
    c_matrix: np.ndarray,
    config: SpMMConfig,
    ranges: Sequence[tuple[int, int]],
) -> tuple[dict[int, BlockResult], list[dict[str, Any]], int]:
    keys = [
        {"block_id": block_id, "stage": "compute", "sort_key": [block_id, "compute"]}
        for block_id in range(len(ranges))
    ]
    if not ranges:
        return {}, keys, 0
    workers = min(config.max_workers, len(ranges))
    results: dict[int, BlockResult] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spmm-full") as pool:
        futures = []
        for block_id, (row_start, row_end) in enumerate(ranges):
            futures.append(
                (
                    block_id,
                    pool.submit(
                        compute_block,
                        csr,
                        b_matrix,
                        c_matrix,
                        config,
                        block_id,
                        row_start,
                        row_end,
                    ),
                )
            )
        for block_id, future in futures:
            results[block_id] = future.result()
    return results, keys, workers


def full_rebuild(
    csr: CSRMatrix,
    b_matrix: np.ndarray,
    c_matrix: np.ndarray,
    config: SpMMConfig,
) -> FullRebuildResult:
    failures = validate_bundle(csr, b_matrix, c_matrix, config)
    if failures:
        raise InputValidationError(failures)
    ranges = row_blocks(csr.rows, config.row_block_size)
    blocks, task_keys, _workers = _parallel_compute_all(
        csr, np.asarray(b_matrix, dtype=np.float64), np.asarray(c_matrix, dtype=np.float64), config, ranges
    )
    dependencies = {
        block_id: derive_dependency(
            csr, b_matrix, c_matrix, config, block_id, row_start, row_end
        ).with_output(blocks[block_id].output_fingerprint)
        for block_id, (row_start, row_end) in enumerate(ranges)
    }
    output = np.empty((csr.rows, int(np.asarray(b_matrix).shape[1])), dtype=np.float32)
    row_fingerprints: list[str] = []
    for block_id, (row_start, row_end) in enumerate(ranges):
        block = blocks[block_id]
        output[row_start:row_end] = block.values
        row_fingerprints.extend(block.row_fingerprints)
    overflow_records = sorted(
        (
            block.first_overflow.to_dict()
            for block in blocks.values()
            if block.first_overflow is not None
        ),
        key=lambda item: (item["block_id"], item["row"], item["column"]),
    )
    hot = HotBlockTable(config.hot_capacity)
    ledger: list[dict[str, Any]] = []
    total_cost = sum(block.recompute_nnz for block in blocks.values())
    remaining = total_cost
    for sequence, block_id in enumerate(sorted(blocks), start=1):
        cost = blocks[block_id].recompute_nnz
        hot.add_recompute(block_id, cost)
        before = remaining
        remaining -= cost
        ledger.append(
            {
                "sequence": sequence,
                "run_id": 1,
                "block_id": block_id,
                "cost": cost,
                "budget_before": before,
                "budget_after": remaining,
            }
        )
    output_fingerprint = digest_json(
        [blocks[block_id].output_fingerprint for block_id in sorted(blocks)]
    )
    return FullRebuildResult(
        config=config,
        blocks=blocks,
        dependencies=dependencies,
        c_matrix=output,
        row_fingerprints=tuple(row_fingerprints),
        overflow_records=overflow_records,
        hot_table=hot,
        budget_ledger=ledger,
        output_fingerprint=output_fingerprint,
        task_keys=task_keys,
    )


def compare_incremental_to_full(
    engine: IncrementalEngine,
    reference: FullRebuildResult,
    *,
    compare_history_dependent: bool = False,
) -> dict[str, Any]:
    engine_c = engine.materialize_c()
    c_exact = array_bytes(engine_c, "<f4") == array_bytes(reference.c_matrix, "<f4")
    row_exact = engine.row_fingerprints == reference.row_fingerprints
    overflow_exact = engine.overflow_records == reference.overflow_records
    engine_blocks = {
        block_id: block.output_fingerprint
        for block_id, block in sorted(engine.output_blocks.items())
    }
    reference_blocks = {
        block_id: block.output_fingerprint
        for block_id, block in sorted(reference.blocks.items())
    }
    block_exact = engine_blocks == reference_blocks
    dependency_exact = all(
        engine.dependencies[block_id].input_fingerprint
        == reference.dependencies[block_id].input_fingerprint
        and engine.dependencies[block_id].output_fingerprint
        == reference.dependencies[block_id].output_fingerprint
        for block_id in reference.blocks
    ) and set(engine.dependencies) == set(reference.dependencies)
    difference = np.abs(
        np.asarray(engine_c, dtype=np.float64) - np.asarray(reference.c_matrix, dtype=np.float64)
    )
    result = {
        "numeric_exact": bool(c_exact),
        "row_fingerprints_exact": bool(row_exact),
        "overflow_records_exact": bool(overflow_exact),
        "block_fingerprints_exact": bool(block_exact),
        "dependency_graph_exact": bool(dependency_exact),
        "max_abs_difference": float(np.nanmax(difference)) if difference.size else 0.0,
        "shape_incremental": list(engine_c.shape),
        "shape_full": list(reference.c_matrix.shape),
        "all_exact": bool(
            c_exact and row_exact and overflow_exact and block_exact and dependency_exact
        ),
        "history_dependent_comparison": bool(compare_history_dependent),
    }
    if compare_history_dependent:
        result["hot_table_exact"] = engine.hot.to_dict() == reference.hot_table.to_dict()
        result["budget_ledger_exact"] = engine.budget_ledger == reference.budget_ledger
        result["all_including_history_exact"] = bool(
            result["all_exact"]
            and result["hot_table_exact"]
            and result["budget_ledger_exact"]
        )
    return result


def compare_blocks_exact(left: Sequence[BlockResult], right: Sequence[BlockResult]) -> bool:
    if len(left) != len(right):
        return False
    for first, second in zip(left, right):
        if first.output_fingerprint != second.output_fingerprint:
            return False
    return True


def manifest_digest(value: Any) -> str:
    return hashlib.sha256(digest_json(value).encode("ascii")).hexdigest()
