"""Batch-split vectorized plan compilation.

A plan covers [0, row_count) with contiguous, non-overlapping, gapless row
ranges [start, end). Declared batch lists are validated; any batch exceeding
the total row count is rejected with E_PLAN.
"""

from dataclasses import dataclass

from .errors import PlanError
from .expr import expr_digest, normalize_text, parse


@dataclass(frozen=True)
class Batch:
    index: int
    row_start: int
    row_end: int

    @property
    def rows(self):
        return self.row_end - self.row_start


@dataclass
class Plan:
    expr_text: str
    normalized: str
    digest: str
    row_count: int
    batches: list


def split_batches(row_count, batch_max_rows):
    if batch_max_rows <= 0:
        raise PlanError("batch_max_rows must be positive", batch_max_rows=batch_max_rows)
    if row_count < 0:
        raise PlanError("row_count must be non-negative", row_count=row_count)
    batches = []
    start = 0
    while start < row_count:
        end = min(start + batch_max_rows, row_count)
        batches.append(Batch(len(batches), start, end))
        start = end
    return batches


def validate_declared_batches(row_count, ranges):
    """Validate an explicitly declared list of (start, end) ranges."""
    expect = 0
    for i, (start, end) in enumerate(ranges):
        if start > row_count or end > row_count:
            raise PlanError("batch range exceeds total row count",
                            batch_index=i, row_start=start, row_end=end,
                            row_count=row_count)
        if start != expect:
            raise PlanError("batch ranges must be contiguous and gapless",
                            batch_index=i, expected_start=expect, row_start=start)
        if end <= start:
            raise PlanError("batch range must be non-empty and ordered",
                            batch_index=i, row_start=start, row_end=end)
        expect = end
    if expect != row_count:
        raise PlanError("batch ranges do not cover all rows",
                        covered=expect, row_count=row_count)
    return [Batch(i, s, e) for i, (s, e) in enumerate(ranges)]


def compile_plan(expr_text, row_count, batch_max_rows, declared_ranges=None):
    parse(expr_text)  # syntax check, raises E_PLAN on errors
    normalized = normalize_text(expr_text)
    batches = (validate_declared_batches(row_count, declared_ranges)
               if declared_ranges is not None
               else split_batches(row_count, batch_max_rows))
    return Plan(expr_text=expr_text, normalized=normalized,
                digest=expr_digest(expr_text), row_count=row_count, batches=batches)
