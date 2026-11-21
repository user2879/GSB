"""Integer fixed-point pipeline stages with explicit rounding rules.

All stage values are signed 64-bit integers. Source integers are multiplied by
10**fixed_scale. Source decimal values are multiplied by 10**fixed_scale and
truncated toward zero. Arithmetic that leaves signed 64-bit range saturates to
INT64_MIN or INT64_MAX. Normalization uses floor division on non-negative
offsets. Statistical clipping computes an integer mean and variance, floors
their divisions, and uses integer square root for standard deviation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from decimal import Decimal, ROUND_DOWN
from typing import Any

from .canonical import shard_digest, table_row_digest
from .errors import E_OUTLIER, E_PARAM, E_STATE, WorkbenchError
from .loader import decimal_text
from .models import (
    INT64_MAX,
    INT64_MIN,
    BatchObservation,
    DatasetSpec,
    ShardingSpec,
    StageComputation,
    StageSpec,
    Table,
    TableColumn,
)


def saturate_int64(value: int | Decimal) -> int:
    if isinstance(value, Decimal):
        if value < INT64_MIN:
            return INT64_MIN
        if value > INT64_MAX:
            return INT64_MAX
        return int(value)
    if value < INT64_MIN:
        return INT64_MIN
    if value > INT64_MAX:
        return INT64_MAX
    return int(value)


def _overflow_or_saturate(
    value: int | Decimal,
    saturate: bool,
    stage: StageSpec,
) -> int:
    if (isinstance(value, Decimal) and (value < INT64_MIN or value > INT64_MAX)) or (
        isinstance(value, int) and (value < INT64_MIN or value > INT64_MAX)
    ):
        if not saturate:
            raise WorkbenchError(
                E_PARAM,
                "fixed-point conversion overflow with saturation disabled",
                stage_id=stage.id,
                param="saturate",
                param_value=0,
                affected_rows=1,
                location=f"pipeline.stages[id={stage.id}].params.saturate",
            )
        return saturate_int64(value)
    return int(value)


def _canonical_value(value: int | Decimal | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    return decimal_text(value)


def table_digest(table: Table) -> str:
    entries = [
        {
            "key": column.key,
            "type": column.type,
            "values": [_canonical_value(value) for value in column.values],
        }
        for column in table.columns
    ]
    return table_row_digest(entries, 0, table.row_count)


def batch_digest(table: Table, start: int, end: int) -> str:
    entries = [
        {
            "key": column.key,
            "type": column.type,
            "values": [_canonical_value(value) for value in column.values],
        }
        for column in table.columns
    ]
    return table_row_digest(entries, start, end)


def table_from_dataset(dataset: DatasetSpec) -> Table:
    columns: list[TableColumn] = []
    for source in dataset.sources:
        for column in source.columns:
            # Sources are concatenated in declaration order. A source-local
            # column is null outside that source's global row interval.
            values: list[int | Decimal | None] = [None] * dataset.row_count
            nulls = [True] * dataset.row_count
            end = source.start_row + source.rows
            values[source.start_row:end] = list(column.values)
            nulls[source.start_row:end] = list(column.nulls)
            columns.append(
                TableColumn(
                    key=column.key,
                    type=column.kind,
                    values=values,
                    nulls=nulls,
                )
            )
    return Table(row_count=dataset.row_count, columns=columns)


def _batch_ranges(row_count: int, batch_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + batch_size, row_count))
        for start in range(0, row_count, batch_size)
    ]


def build_shards(
    table: Table,
    sharding: ShardingSpec,
    stage: StageSpec,
    source_output_digest: str,
) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    index = 0
    for start in range(0, table.row_count, sharding.target_rows):
        end = min(start + sharding.target_rows, table.row_count)
        if end - start < sharding.target_rows and sharding.remainder_policy == "drop_partial":
            continue
        rows_digest = batch_digest(table, start, end)
        fingerprint = shard_digest(stage.id, stage.version, start, end, rows_digest)
        shards.append(
            {
                "shard_index": index,
                "start_row": start,
                "end_row": end,
                "row_count": end - start,
                "rows_digest": rows_digest,
                "shard_fingerprint": fingerprint,
                "source_stage_id": stage.id,
                "source_stage_version": stage.version,
                "source_output_digest": source_output_digest,
            }
        )
        index += 1
    return shards


def apply_stage(
    table: Table,
    stage: StageSpec,
    fixed_scale: int,
    sharding: ShardingSpec,
    batch_size: int,
    on_batch: Callable[[BatchObservation], None] | None = None,
    normalize_scale: int | None = None,
) -> tuple[Table, StageComputation]:
    """Apply one stage and return a possibly-mutated table plus observations.

    Statistics are computed over the whole stage input before row batches are
    mutated, so batch boundaries cannot alter any output. Each batch callback is
    invoked immediately after that batch reaches its final bytes.
    """

    if batch_size <= 0:
        raise WorkbenchError(E_STATE, "batch_size must be positive", param="batch_size")
    input_digest = table_digest(table)
    factor = 10**fixed_scale
    normalize_factor = 10 ** (fixed_scale if normalize_scale is None else normalize_scale)
    batches: list[BatchObservation] = []
    ranges = _batch_ranges(table.row_count, batch_size)

    normalize_stats: tuple[int, int, int] | None = None
    clip_state: tuple[int, int, int, int] | None = None

    if stage.type == "normalize":
        column = table.columns[stage.params["column"]]
        values = [value for value, null in zip(column.values, column.nulls) if not null]
        if not values:
            raise WorkbenchError(
                E_STATE,
                "normalize received an all-null column",
                stage_id=stage.id,
                affected_rows=table.row_count,
            )
        minimum = min(values)
        maximum = max(values)
        normalize_stats = (int(minimum), int(maximum), len(values))
    elif stage.type == "clip_outliers":
        column = table.columns[stage.params["column"]]
        values = [int(value) for value, null in zip(column.values, column.nulls) if not null]
        if not values:
            raise WorkbenchError(
                E_STATE,
                "clip_outliers received an all-null column",
                stage_id=stage.id,
                affected_rows=table.row_count,
            )
        count = len(values)
        mean = sum(values) // count
        variance = sum((value - mean) ** 2 for value in values) // count
        stddev = math.isqrt(variance)
        z_limit = stage.params["z_limit_permille"]
        delta = z_limit * stddev // 1000
        lower = saturate_int64(mean - delta)
        upper = saturate_int64(mean + delta)
        outlier_count = sum(1 for value in values if value < lower or value > upper)
        max_outlier = stage.params["max_outlier_permille"]
        if outlier_count * 1000 > max_outlier * count:
            ratio = outlier_count * 1000 // count
            raise WorkbenchError(
                E_OUTLIER,
                (
                    f"outlier ratio {ratio} permille exceeds declared maximum "
                    f"{max_outlier} permille"
                ),
                stage_id=stage.id,
                param="max_outlier_permille",
                param_value=max_outlier,
                affected_rows=outlier_count,
                location=f"pipeline.stages[id={stage.id}].params.max_outlier_permille",
                details={
                    "outlier_ratio_permille": ratio,
                    "non_null_rows": count,
                    "mean_fixed": mean,
                    "stddev_fixed": stddev,
                    "lower_fixed": lower,
                    "upper_fixed": upper,
                },
            )
        clip_state = (mean, stddev, lower, upper)

    for batch_index, (start, end) in enumerate(ranges):
        before = batch_digest(table, start, end)
        if stage.type == "canonicalize":
            saturate = stage.params["saturate"] == 1
            for column in table.columns:
                for row in range(start, end):
                    if column.nulls[row]:
                        column.values[row] = None
                        continue
                    raw = column.values[row]
                    if not isinstance(raw, (int, Decimal)):
                        raise WorkbenchError(E_STATE, "canonicalize input is not numeric")
                    if isinstance(raw, int):
                        column.values[row] = _overflow_or_saturate(raw * factor, saturate, stage)
                    else:
                        scaled = (raw * factor).to_integral_value(rounding=ROUND_DOWN)
                        column.values[row] = _overflow_or_saturate(scaled, saturate, stage)
                column.type = "fixed"
        elif stage.type == "fill_missing":
            fill = stage.params["fill_fixed"]
            for column in table.columns:
                for row in range(start, end):
                    if column.nulls[row]:
                        column.values[row] = fill
                        column.nulls[row] = False
        elif stage.type == "normalize":
            assert normalize_stats is not None
            minimum, maximum, _ = normalize_stats
            span = maximum - minimum
            column = table.columns[stage.params["column"]]
            denominator = normalize_factor
            for row in range(start, end):
                if column.nulls[row]:
                    raise WorkbenchError(
                        E_STATE,
                        "normalize requires fill_missing first",
                        stage_id=stage.id,
                        affected_rows=1,
                    )
                value = int(column.values[row])
                if span == 0:
                    column.values[row] = 0
                else:
                    column.values[row] = ((value - minimum) * denominator) // span
        elif stage.type == "clip_outliers":
            assert clip_state is not None
            _, _, lower, upper = clip_state
            column = table.columns[stage.params["column"]]
            for row in range(start, end):
                if column.nulls[row]:
                    continue
                value = int(column.values[row])
                column.values[row] = lower if value < lower else upper if value > upper else value
        elif stage.type == "shard_pack":
            pass
        else:
            raise WorkbenchError(E_STATE, f"unsupported stage type {stage.type!r}")

        after = batch_digest(table, start, end)
        observation = BatchObservation(
            batch_index=batch_index,
            start_row=start,
            end_row=end,
            row_count=end - start,
            input_digest=before,
            output_digest=after,
        )
        batches.append(observation)
        if on_batch is not None:
            on_batch(observation)

    output_digest = table_digest(table)
    shards = (
        build_shards(table, sharding, stage, output_digest)
        if stage.type == "shard_pack"
        else []
    )
    computation = StageComputation(
        stage=stage,
        input_digest=input_digest,
        output_digest=output_digest,
        row_count=table.row_count,
        batches=batches,
        shards=shards,
    )
    return table, computation
