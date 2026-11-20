"""Budgeted deterministic imputation-cell generation with durable reuse."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .errors import E_IMPUTATION_FAILED, E_INSUFFICIENT_SAMPLES, diagnostic
from .fixedpoint import combine_directions, deterministic_standard_normal, fixed_to_float, fixed_to_text, quantize_units, to_fixed
from .loader import group_for_row
from .stats import column_params_fingerprint
from .util import sha256_obj


def _job_cache_key(
    row_id: str,
    column: str,
    spec: dict[str, Any],
    source_stat_fingerprint: str,
    source_stat_cache_key: str,
) -> str:
    return sha256_obj(
        {
            "level": "imputation_cell",
            "row_id": row_id,
            "column": column,
            "params_fingerprint": column_params_fingerprint(spec),
            "source_stat_fingerprint": source_stat_fingerprint,
            "source_stat_cache_key": source_stat_cache_key,
            "decimal_places": spec["decimal_places"],
            "quantization_level": spec["quantization_level"],
            "rounding": spec.get("rounding", "half_even"),
        }
    )


def _source_stat(
    row: dict[str, Any],
    column: str,
    spec: dict[str, Any],
    stats: dict[str, Any],
    group_map: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    record = stats.get(column)
    if record is None:
        return None, None
    if spec["imputer"] == "grouped_median":
        group = group_for_row(row, str(spec["group_key"]), group_map)
        if group is None:
            return None, None
        return record.get("groups", {}).get(group), group
    return record, None


def _compute_cell(
    job: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    stats: dict[str, Any],
    group_map: dict[str, Any],
) -> dict[str, Any]:
    row_id = job["row_id"]
    column = job["column"]
    spec = config["columns"][column]
    row = rows_by_id[row_id]
    source, group = _source_stat(row, column, spec, stats, group_map)
    if source is None:
        raise ValueError(f"no statistics available for {column!r}")
    method = spec["imputer"]
    status_codes: list[str] = list(source.get("status_codes", []))
    fallback_used = bool(source.get("fallback_used", False))
    extra: dict[str, Any] = {}
    if method == "mean_noise" and not fallback_used:
        z, noise_source = deterministic_standard_normal(row_id, column)
        raw_std = float(source.get("sample_std_raw", 0.0))
        raw_mean = float(source.get("mean_raw", source.get("value", 0.0)))
        raw_value = raw_mean + z * raw_std * float(spec.get("noise_coefficient", 0.0))
        rounded = to_fixed(raw_value, int(spec["decimal_places"]), spec.get("rounding", "half_even"))
        quantized = quantize_units(
            rounded.units,
            int(spec["quantization_level"]),
            spec.get("rounding", "half_even"),
        )
        fixed = quantized.units
        direction = combine_directions(rounded.direction, quantized.direction)
        value = fixed_to_float(fixed, int(spec["decimal_places"]))
        value_text = fixed_to_text(fixed, int(spec["decimal_places"]))
        extra = {
            "noise_z": z,
            "noise_source": noise_source,
            "noise_coefficient": float(spec.get("noise_coefficient", 0.0)),
        }
    else:
        fixed = int(source["value_fixed"])
        value = source["value"]
        value_text = source["value_text"]
        direction = source.get("rounding_direction", "exact")

    if fallback_used and E_INSUFFICIENT_SAMPLES not in status_codes:
        status_codes.append(E_INSUFFICIENT_SAMPLES)
    cell = {
        "row_id": row_id,
        "column": column,
        "type": spec["type"],
        "value": value,
        "value_text": value_text,
        "value_fixed": fixed,
        "decimal_places": spec["decimal_places"],
        "quantization_level": spec["quantization_level"],
        "rounding_direction": direction,
        "source_stat_fingerprint": source["fingerprint"],
        "source_stat_cache_key": source["cache_key"],
        "source_group": group,
        "cache_key": job["cache_key"],
        "fallback_used": fallback_used,
        "status_codes": sorted(set(status_codes)),
        **extra,
    }
    cell["fingerprint"] = sha256_obj(cell)
    return cell


def build_jobs(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    valid_ids: set[str],
    stats: dict[str, Any],
    group_map: dict[str, Any],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row in rows:
        if row["row_id"] not in valid_ids:
            continue
        for column in sorted(config["columns"]):
            if row["values"].get(column) is not None:
                continue
            spec = config["columns"][column]
            source, group = _source_stat(row, column, spec, stats, group_map)
            if source is None:
                # A row with an unavailable group is already represented by a
                # group-map failure and cannot enter the training set.
                continue
            cache_key = _job_cache_key(
                row["row_id"],
                column,
                spec,
                source["fingerprint"],
                source["cache_key"],
            )
            jobs.append(
                {
                    "column": column,
                    "row_id": row["row_id"],
                    "source_group": group,
                    "source_stat_fingerprint": source["fingerprint"],
                    "source_stat_cache_key": source["cache_key"],
                    "cache_key": cache_key,
                }
            )
    jobs.sort(key=lambda job: (job["column"], job["row_id"]))
    return jobs


def run_imputation_slice(
    previous_cells: dict[str, Any],
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    valid_ids: set[str],
    stats: dict[str, Any],
    group_map: dict[str, Any],
    previous_failures: list[dict[str, Any]],
    budget: int,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compute at most ``budget`` new cells, preserving a stable pending queue."""

    jobs = build_jobs(config, rows, valid_ids, stats, group_map)
    rows_by_id = {row["row_id"]: row for row in rows}
    expected_keys = {(job["row_id"], job["column"]) for job in jobs}

    # Drop cells that are no longer missing or whose row disappeared.
    cells: dict[str, Any] = {}
    for row_id, column_map in previous_cells.items():
        kept: dict[str, Any] = {}
        for column, cell in column_map.items():
            if (row_id, column) in expected_keys:
                kept[column] = cell
        if kept:
            cells[row_id] = kept

    reused = 0
    uncached: list[dict[str, Any]] = []
    for job in jobs:
        old = cells.get(job["row_id"], {}).get(job["column"])
        if old is not None and old.get("cache_key") == job["cache_key"]:
            reused += 1
        else:
            uncached.append(job)

    failed_keys = {
        (failure.get("row_id"), failure.get("column"), failure.get("cache_key"))
        for failure in previous_failures
    }
    terminal_failures: list[dict[str, Any]] = []
    actionable: list[dict[str, Any]] = []
    for job in uncached:
        key = (job["row_id"], job["column"], job["cache_key"])
        if key in failed_keys:
            terminal_failures.append(
                next(
                    failure
                    for failure in previous_failures
                    if (failure.get("row_id"), failure.get("column"), failure.get("cache_key")) == key
                )
            )
        else:
            actionable.append(job)

    selected = actionable[: max(0, int(budget))]
    pending = actionable[len(selected) :]

    def compute(job: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        try:
            return _compute_cell(job, rows_by_id, config, stats, group_map), None
        except Exception as exc:  # Exceptions are isolated per row/column task.
            return None, diagnostic(
                E_IMPUTATION_FAILED,
                f"imputation failed: {type(exc).__name__}: {exc}",
                row_id=job["row_id"],
                column=job["column"],
                stage="imputation",
                cache_key=job["cache_key"],
            )

    outputs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    if workers <= 1:
        for job in selected:
            outputs.append(compute(job))
    elif selected:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="iw-impute") as executor:
            futures = [executor.submit(compute, job) for job in selected]
            for future in futures:
                outputs.append(future.result())

    computed = 0
    new_failures: list[dict[str, Any]] = []
    for cell, failure in outputs:
        if cell is not None:
            cells.setdefault(cell["row_id"], {})[cell["column"]] = cell
            computed += 1
        elif failure is not None:
            new_failures.append(failure)

    # Stable order inside each row makes persisted files easy to inspect.
    cells = {
        row_id: dict(sorted(column_map.items()))
        for row_id, column_map in sorted(cells.items())
    }
    failures = terminal_failures + new_failures
    failures.sort(key=lambda item: (item.get("row_id", ""), item.get("column", ""), item.get("cache_key", "")))
    pending_records = [
        {
            "column": job["column"],
            "row_id": job["row_id"],
            "cache_key": job["cache_key"],
            "source_stat_fingerprint": job["source_stat_fingerprint"],
        }
        for job in pending
    ]
    counts_by_column: dict[str, int] = {}
    for row_column_map in cells.values():
        for column in row_column_map:
            counts_by_column[column] = counts_by_column.get(column, 0) + 1
    metrics = {
        "expected_cells": len(jobs),
        "computed_cells": computed,
        "reused_cells": reused,
        "computed_cells_by_column": _count_selected(selected),
        "imputed_cell_counts_by_column": dict(sorted(counts_by_column.items())),
        "pending_jobs": pending_records,
        "pending_count": len(pending_records),
        "failures": failures,
    }
    return cells, pending_records, failures, metrics


def _count_selected(jobs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job["column"]] = counts.get(job["column"], 0) + 1
    return dict(sorted(counts.items()))


def imputation_data_fingerprint(
    rows: list[dict[str, Any]],
    cells: dict[str, Any],
    excluded_row_ids: set[str],
) -> str:
    material = []
    for row in sorted(rows, key=lambda item: item["row_id"]):
        if row["row_id"] in excluded_row_ids:
            continue
        values = []
        for column in sorted(row["values"]):
            raw = row["values"][column]
            cell = cells.get(row["row_id"], {}).get(column)
            values.append(
                {
                    "column": column,
                    "raw": raw,
                    "imputed_fingerprint": cell.get("fingerprint") if cell is not None else None,
                    "imputed_value": cell.get("value") if cell is not None else None,
                }
            )
        material.append({"row_id": row["row_id"], "values": values})
    return sha256_obj(material)


def model_dependency_fingerprint(
    stats: dict[str, Any],
    rows: list[dict[str, Any]],
    cells: dict[str, Any],
    excluded_row_ids: set[str],
) -> str:
    """Fingerprint both statistics and imputed cells for model invalidation.

    A statistics change is a model dependency change even when a particular column
    currently has no missing cells, so the statistics layer is deliberately part of
    this downstream fingerprint.
    """

    stats_material = {
        column: {
            "fingerprint": record.get("fingerprint"),
            "cache_key": record.get("cache_key"),
            "groups": {
                group: {
                    "fingerprint": group_record.get("fingerprint"),
                    "cache_key": group_record.get("cache_key"),
                }
                for group, group_record in sorted(record.get("groups", {}).items())
            },
        }
        for column, record in sorted(stats.items())
    }
    return sha256_obj(
        {
            "stats": stats_material,
            "imputed": imputation_data_fingerprint(rows, cells, excluded_row_ids),
        }
    )
