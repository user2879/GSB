from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
import hashlib
import json
import math
import random
import statistics
from typing import Any, Callable

from .model import FilterPredicate, Table, Tuning, Workload


getcontext().prec = 40


@dataclass(frozen=True)
class StatsProfile:
    caliber: str
    version: str
    scan_rows: int
    sample_size: int
    matched_rows: int
    selectivity: Decimal
    confidence_low: Decimal
    confidence_high: Decimal
    confidence_level: float
    seed: str

    def as_dict(self, scale: int) -> dict[str, Any]:
        return {
            "caliber": self.caliber,
            "version": self.version,
            "scan_rows": self.scan_rows,
            "sample_size": self.sample_size,
            "matched_rows": self.matched_rows,
            "selectivity_fx": fixed(self.selectivity, scale),
            "selectivity": float(self.selectivity),
            "confidence_low_fx": fixed(self.confidence_low, scale),
            "confidence_high_fx": fixed(self.confidence_high, scale),
            "confidence_level": self.confidence_level,
            "ci_method": "wilson-score",
            "seed": self.seed,
        }


@dataclass(frozen=True)
class FilterStats:
    table: str
    column: str
    op: str
    value: Any
    is_null_test: bool
    predicate_key: str
    exact: StatsProfile
    sampled: StatsProfile
    selected: StatsProfile

    @property
    def difference(self) -> Decimal:
        return self.exact.selectivity - self.sampled.selectivity

    def as_dict(self, scale: int) -> dict[str, Any]:
        return {
            "table": self.table,
            "column": self.column,
            "op": self.op,
            "value": self.value,
            "is_null_test": self.is_null_test,
            "predicate_key": self.predicate_key,
            "selected_caliber": self.selected.caliber,
            "selected_version": self.selected.version,
            "exact": self.exact.as_dict(scale),
            "sampled": self.sampled.as_dict(scale),
            "difference_fx": fixed_signed(self.difference, scale),
            "difference": float(self.difference),
        }


@dataclass(frozen=True)
class StatisticsBundle:
    seed: int
    exact_scan_threshold_rows: int
    sample_size: int
    confidence_level: float
    filter_stats: tuple[FilterStats, ...]
    hist_versions: tuple[str, ...]
    digest: str

    def selected_map(self) -> dict[str, FilterStats]:
        return {item.predicate_key: item for item in self.filter_stats}

    def as_dict(self, scale: int) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "exact_scan_threshold_rows": self.exact_scan_threshold_rows,
            "sample_size": self.sample_size,
            "confidence_level": self.confidence_level,
            "filter_stats": [item.as_dict(scale) for item in self.filter_stats],
            "hist_versions": list(self.hist_versions),
            "digest": self.digest,
            "caliber_rule": "rows < threshold => exact-v1; otherwise sample-v1",
        }


def collect_statistics(workload: Workload) -> StatisticsBundle:
    tuning = workload.tuning
    table_map = workload.table_map()
    filter_stats: list[FilterStats] = []
    for predicate in workload.query.filters:
        table = table_map[predicate.column.table]
        column_index = table.column_index(predicate.column.column)
        exact = _exact_profile(predicate, table, column_index, tuning)
        sampled = _sampled_profile(predicate, table, column_index, tuning)
        selected = exact if table.row_count < tuning.exact_scan_threshold_rows else sampled
        filter_stats.append(
            FilterStats(
                table=table.name,
                column=predicate.column.column,
                op=predicate.op,
                value=predicate.value,
                is_null_test=predicate.is_null_test,
                predicate_key=predicate.key,
                exact=exact,
                sampled=sampled,
                selected=selected,
            )
        )
    hist_versions: list[str] = []
    for table in workload.tables:
        for column in table.columns:
            histogram = table.histograms[column.name]
            hist_versions.append(
                f"{table.name}.{column.name}@v{histogram.version}:{histogram.digest[:12]}"
            )
    body = {
        "seed": tuning.seed,
        "threshold": tuning.exact_scan_threshold_rows,
        "sample_size": tuning.sample_size,
        "confidence_level": tuning.confidence_level,
        "profiles": [item.as_dict(tuning.fixed_point_scale) for item in filter_stats],
        "hist_versions": hist_versions,
    }
    digest = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    return StatisticsBundle(
        seed=tuning.seed,
        exact_scan_threshold_rows=tuning.exact_scan_threshold_rows,
        sample_size=tuning.sample_size,
        confidence_level=tuning.confidence_level,
        filter_stats=tuple(filter_stats),
        hist_versions=tuple(hist_versions),
        digest=digest,
    )


def evaluate_predicate(predicate: FilterPredicate, row: tuple[Any, ...], index: int) -> bool:
    value = row[index]
    if predicate.is_null_test:
        return (value is None) if predicate.op == "IS NULL" else (value is not None)
    if value is None or predicate.value is None:
        return False
    target = predicate.value
    if predicate.op == "=":
        return value == target
    if predicate.op in {"!=", "<>"}:
        return value != target
    if predicate.op == "<":
        return value < target
    if predicate.op == "<=":
        return value <= target
    if predicate.op == ">":
        return value > target
    if predicate.op == ">=":
        return value >= target
    raise ValueError(f"unsupported predicate op {predicate.op}")


def _exact_profile(
    predicate: FilterPredicate, table: Table, column_index: int, tuning: Tuning
) -> StatsProfile:
    matched = sum(
        1 for row in table.rows if evaluate_predicate(predicate, row, column_index)
    )
    rows = table.row_count
    selectivity = Decimal(matched) / Decimal(rows) if rows else Decimal(0)
    return StatsProfile(
        caliber="exact",
        version="exact-v1",
        scan_rows=rows,
        sample_size=rows,
        matched_rows=matched,
        selectivity=selectivity,
        confidence_low=selectivity,
        confidence_high=selectivity,
        confidence_level=1.0,
        seed="full-scan",
    )


def _sampled_profile(
    predicate: FilterPredicate, table: Table, column_index: int, tuning: Tuning
) -> StatsProfile:
    rows = table.row_count
    size = min(rows, tuning.sample_size)
    seed_material = f"{tuning.seed}|{table.name}|{predicate.key}"
    seed_digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int(seed_digest[:16], 16))
    indices = sorted(rng.sample(range(rows), size)) if size else []
    matched = sum(
        1 for index in indices if evaluate_predicate(predicate, table.rows[index], column_index)
    )
    selectivity = Decimal(matched) / Decimal(size) if size else Decimal(0)
    low, high = _wilson_interval(matched, size, tuning.confidence_level)
    version = f"sample-v1-n{size}-s{seed_digest[:8]}"
    return StatsProfile(
        caliber="sample",
        version=version,
        scan_rows=size,
        sample_size=size,
        matched_rows=matched,
        selectivity=selectivity,
        confidence_low=low,
        confidence_high=high,
        confidence_level=tuning.confidence_level,
        seed=seed_digest,
    )


def _wilson_interval(matched: int, size: int, confidence: float) -> tuple[Decimal, Decimal]:
    if size == 0:
        return Decimal(0), Decimal(0)
    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = matched / size
    denominator = 1.0 + z * z / size
    centre = (p + z * z / (2.0 * size)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * size)) / size) / denominator
    return (
        Decimal(str(max(0.0, centre - margin))),
        Decimal(str(min(1.0, centre + margin))),
    )


def fixed(value: Decimal | int | float, scale: int) -> int:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    return int(value * scale)


def fixed_signed(value: Decimal | int | float, scale: int) -> int:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    return int(value * scale)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
