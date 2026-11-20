"""Incremental column and grouped statistics aggregation."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import statistics
from typing import Any

from .errors import E_INSUFFICIENT_SAMPLES, WorkbenchError
from .fixedpoint import FixedValue, combine_directions, fixed_to_float, fixed_to_text, quantize_units, to_fixed
from .loader import group_for_row
from .util import sha256_obj


def column_params_fingerprint(spec: dict[str, Any]) -> str:
    material = {
        "type": spec.get("type"),
        "imputer": spec.get("imputer"),
        "min_valid_samples": spec.get("min_valid_samples"),
        "fallback": spec.get("fallback"),
        "group_key": spec.get("group_key"),
        "noise_coefficient": spec.get("noise_coefficient"),
        "constant": spec.get("constant"),
        "domain": spec.get("domain", []),
        "decimal_places": spec.get("decimal_places"),
        "quantization_level": spec.get("quantization_level"),
        "rounding": spec.get("rounding", "half_even"),
    }
    return sha256_obj(material)


def _fixed_numeric(value: float, spec: dict[str, Any]) -> FixedValue:
    rounded = to_fixed(value, int(spec["decimal_places"]), spec.get("rounding", "half_even"))
    quantized = quantize_units(rounded.units, int(spec["quantization_level"]), spec.get("rounding", "half_even"))
    return FixedValue(quantized.units, combine_directions(rounded.direction, quantized.direction))


def _category_index(value: str, spec: dict[str, Any]) -> int:
    try:
        return sorted(spec.get("domain", [])).index(value)
    except ValueError as exc:
        raise WorkbenchError("E_CONFIG_INVALID", f"category value {value!r} is not in domain") from exc


def _numeric_payload(value_fixed: int, spec: dict[str, Any], direction: str) -> dict[str, Any]:
    places = int(spec["decimal_places"])
    return {
        "value_fixed": value_fixed,
        "value": fixed_to_float(value_fixed, places),
        "value_text": fixed_to_text(value_fixed, places),
        "rounding_direction": direction,
    }


def _category_payload(value: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "value_fixed": _category_index(value, spec),
        "value": value,
        "value_text": value,
        "rounding_direction": "exact",
    }


def _fallback_payload(spec: dict[str, Any]) -> dict[str, Any]:
    if spec["type"] == "numeric":
        fixed = _fixed_numeric(float(spec["fallback"]), spec)
        return _numeric_payload(fixed.units, spec, fixed.direction)
    return _category_payload(str(spec["fallback"]), spec)


def _candidate_payload(values: list[Any], spec: dict[str, Any]) -> dict[str, Any]:
    method = spec["imputer"]
    if method in {"median", "grouped_median"}:
        raw = float(statistics.median(values))
        fixed = _fixed_numeric(raw, spec)
        payload = _numeric_payload(fixed.units, spec, fixed.direction)
        payload["raw_value"] = raw
        return payload
    if method == "mean_noise":
        raw_mean = float(statistics.fmean(values))
        raw_std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        fixed_mean = _fixed_numeric(raw_mean, spec)
        fixed_std = _fixed_numeric(raw_std, spec)
        payload = _numeric_payload(fixed_mean.units, spec, fixed_mean.direction)
        payload.update(
            {
                "mean_fixed": fixed_mean.units,
                "mean": fixed_to_float(fixed_mean.units, int(spec["decimal_places"])),
                "mean_text": fixed_to_text(fixed_mean.units, int(spec["decimal_places"])),
                "mean_raw": raw_mean,
                "sample_std_fixed": fixed_std.units,
                "sample_std": fixed_to_float(fixed_std.units, int(spec["decimal_places"])),
                "sample_std_text": fixed_to_text(fixed_std.units, int(spec["decimal_places"])),
                "sample_std_raw": raw_std,
            }
        )
        return payload
    if method == "mode":
        counts = Counter(values)
        winner = min(counts, key=lambda value: (-counts[value], str(value)))
        return _category_payload(str(winner), spec)
    if method == "constant":
        if spec["type"] == "numeric":
            fixed = _fixed_numeric(float(spec["constant"]), spec)
            return _numeric_payload(fixed.units, spec, fixed.direction)
        return _category_payload(str(spec["constant"]), spec)
    raise WorkbenchError("E_CONFIG_INVALID", f"unsupported imputer {method!r}")


def _evidence_for_ungrouped(
    rows: list[dict[str, Any]], valid_ids: set[str], column: str
) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    referenced: list[str] = []
    for row in rows:
        if row["row_id"] not in valid_ids:
            continue
        value = row["values"].get(column)
        if value is None:
            continue
        evidence.append({"row_id": row["row_id"], "value": value})
        referenced.append(row["row_id"])
    return evidence, referenced


def _evidence_for_grouped(
    rows: list[dict[str, Any]],
    valid_ids: set[str],
    column: str,
    group_key: str,
    group_map: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["row_id"] not in valid_ids:
            continue
        group = group_for_row(row, group_key, group_map)
        if group is None:
            # Group coverage failures remove these rows before stats normally run.
            continue
        # Retain group membership even when every target cell is missing.  This is
        # what makes a brand-new empty group visible as an explicit insufficient-
        # sample fallback instead of silently dropping the row from imputation.
        groups.setdefault(group, []).append(
            {
                "row_id": row["row_id"],
                "value": row["values"].get(column),
                "group_key_value": row["values"].get(group_key),
            }
        )
    return {group: entries for group, entries in sorted(groups.items())}


def _compute_group_record(
    column: str,
    group: str,
    entries: list[dict[str, Any]],
    spec: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    evidence_fp = sha256_obj(entries)
    params_fp = column_params_fingerprint(spec)
    cache_key = sha256_obj(
        {
            "level": "group_stat",
            "column": column,
            "group": group,
            "params_fingerprint": params_fp,
            "evidence_fingerprint": evidence_fp,
            "decimal_places": spec["decimal_places"],
            "quantization_level": spec["quantization_level"],
        }
    )
    values = [entry["value"] for entry in entries if entry["value"] is not None]
    sample_count = len(values)
    sufficient = sample_count >= int(spec["min_valid_samples"])
    status_codes: list[str] = []
    if sufficient:
        payload = _candidate_payload(values, spec)
    else:
        status_codes.append(E_INSUFFICIENT_SAMPLES)
        candidate = _candidate_payload(values, spec) if values else None
        payload = _fallback_payload(spec)
        if candidate is not None:
            payload["candidate"] = candidate
    fingerprint = sha256_obj(
        {
            "cache_key": cache_key,
            "column": column,
            "group": group,
            "method": spec["imputer"],
            "sample_count": sample_count,
            "sufficient": sufficient,
            "fallback_used": not sufficient,
            "payload": payload,
            "referenced_row_ids": [entry["row_id"] for entry in entries],
        }
    )
    return (
        {
            "group": group,
            "sample_count": sample_count,
            "sufficient": sufficient,
            "fallback_used": not sufficient,
            "status_codes": status_codes,
            "group_key_values": sorted(
                {str(entry["group_key_value"]) for entry in entries if entry.get("group_key_value") is not None}
            ),
            "evidence_fingerprint": evidence_fp,
            "cache_key": cache_key,
            "fingerprint": fingerprint,
            "referenced_row_ids": [entry["row_id"] for entry in entries],
            **payload,
        },
        False,
    )


def _build_grouped_record(
    column: str,
    spec: dict[str, Any],
    previous: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    valid_ids: set[str],
    group_map: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    group_key = str(spec["group_key"])
    evidence = _evidence_for_grouped(rows, valid_ids, column, group_key, group_map)
    previous_groups = previous.get("groups", {}) if previous else {}
    params_fp = column_params_fingerprint(spec)
    groups: dict[str, Any] = {}
    recomputed_groups: list[str] = []
    reused_groups: list[str] = []
    for group, entries in evidence.items():
        evidence_fp = sha256_obj(entries)
        old = previous_groups.get(group)
        if (
            old is not None
            and old.get("evidence_fingerprint") == evidence_fp
            and old.get("params_fingerprint", params_fp) == params_fp
            and previous is not None
            and previous.get("params_fingerprint") == params_fp
        ):
            groups[group] = deepcopy(old)
            reused_groups.append(group)
        else:
            record, _ = _compute_group_record(column, group, entries, spec)
            record["params_fingerprint"] = params_fp
            groups[group] = record
            recomputed_groups.append(group)

    group_fps = {group: record["fingerprint"] for group, record in sorted(groups.items())}
    sample_count = sum(record["sample_count"] for record in groups.values())
    insufficient_groups = sorted(
        group for group, record in groups.items() if not record["sufficient"]
    )
    status_codes = [E_INSUFFICIENT_SAMPLES] if insufficient_groups else []
    referenced = sorted(
        {row_id for record in groups.values() for row_id in record["referenced_row_ids"]}
    )
    cache_key = sha256_obj(
        {
            "level": "column_stat",
            "column": column,
            "imputer": "grouped_median",
            "group_key": group_key,
            "params_fingerprint": params_fp,
            "group_fingerprints": group_fps,
            "decimal_places": spec["decimal_places"],
            "quantization_level": spec["quantization_level"],
        }
    )
    record = {
        "column": column,
        "type": spec["type"],
        "imputer": spec["imputer"],
        "group_key": group_key,
        "sample_count": sample_count,
        "sufficient": not insufficient_groups,
        "fallback_used": bool(insufficient_groups),
        "insufficient_groups": insufficient_groups,
        "status_codes": status_codes,
        "decimal_places": spec["decimal_places"],
        "quantization_level": spec["quantization_level"],
        "rounding": spec.get("rounding", "half_even"),
        "params_fingerprint": params_fp,
        "input_fingerprint": sha256_obj(
            {group: sha256_obj(entries) for group, entries in evidence.items()}
        ),
        "cache_key": cache_key,
        "fingerprint": sha256_obj(
            {
                "cache_key": cache_key,
                "group_fingerprints": group_fps,
                "insufficient_groups": insufficient_groups,
            }
        ),
        "referenced_row_ids": referenced,
        "groups": groups,
    }
    return record, {"recomputed_groups": recomputed_groups, "reused_groups": reused_groups}


def _build_ungrouped_record(
    column: str,
    spec: dict[str, Any],
    previous: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    valid_ids: set[str],
) -> tuple[dict[str, Any], bool]:
    evidence, referenced = _evidence_for_ungrouped(rows, valid_ids, column)
    params_fp = column_params_fingerprint(spec)
    evidence_fp = sha256_obj(evidence)
    sample_count = len(evidence)
    if spec["imputer"] == "constant":
        reuse_value = previous is not None and previous.get("params_fingerprint") == params_fp
    else:
        reuse_value = (
            previous is not None
            and previous.get("params_fingerprint") == params_fp
            and previous.get("evidence_fingerprint") == evidence_fp
        )
    values = [entry["value"] for entry in evidence]
    sufficient = spec["imputer"] == "constant" or sample_count >= int(spec["min_valid_samples"])
    status_codes: list[str] = []
    if not sufficient:
        status_codes.append(E_INSUFFICIENT_SAMPLES)
    cache_key = sha256_obj(
        {
            "level": "column_stat",
            "column": column,
            "imputer": spec["imputer"],
            "params_fingerprint": params_fp,
            "evidence_fingerprint": evidence_fp if spec["imputer"] != "constant" else "constant",
            "decimal_places": spec["decimal_places"],
            "quantization_level": spec["quantization_level"],
        }
    )
    if reuse_value:
        record = deepcopy(previous or {})
        record["sample_count"] = sample_count
        record["referenced_row_ids"] = referenced
        record["status_codes"] = status_codes
        record["sufficient"] = sufficient
        record["fallback_used"] = not sufficient
        return record, True

    if sufficient:
        payload = _candidate_payload(values, spec) if values else _fallback_payload(spec)
    else:
        candidate = _candidate_payload(values, spec) if values else None
        payload = _fallback_payload(spec)
        if candidate is not None:
            payload["candidate"] = candidate
    fingerprint = sha256_obj(
        {
            "cache_key": cache_key,
            "column": column,
            "method": spec["imputer"],
            "sample_count": sample_count,
            "sufficient": sufficient,
            "fallback_used": not sufficient,
            "payload": payload,
            "referenced_row_ids": referenced,
        }
    )
    record = {
        "column": column,
        "type": spec["type"],
        "imputer": spec["imputer"],
        "sample_count": sample_count,
        "sufficient": sufficient,
        "fallback_used": not sufficient,
        "status_codes": status_codes,
        "decimal_places": spec["decimal_places"],
        "quantization_level": spec["quantization_level"],
        "rounding": spec.get("rounding", "half_even"),
        "params_fingerprint": params_fp,
        "evidence_fingerprint": evidence_fp,
        "input_fingerprint": evidence_fp,
        "cache_key": cache_key,
        "fingerprint": fingerprint,
        "referenced_row_ids": referenced,
        **payload,
    }
    return record, False


def aggregate_stats_once(
    column: str,
    config: dict[str, Any],
    previous: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    valid_ids: set[str],
    group_map: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = config["columns"][column]
    if spec["imputer"] == "grouped_median":
        record, reuse_info = _build_grouped_record(
            column, spec, previous, rows, valid_ids, group_map
        )
        recomputed = len(reuse_info["recomputed_groups"]) > 0
        metrics = {
            "column": column,
            "computed": recomputed,
            "reused": not recomputed,
            **reuse_info,
        }
        return record, metrics
    record, reused = _build_ungrouped_record(column, spec, previous, rows, valid_ids)
    return record, {"column": column, "computed": not reused, "reused": reused}


def aggregate_stats(
    config: dict[str, Any],
    previous_stats: dict[str, Any],
    rows: list[dict[str, Any]],
    valid_ids: set[str],
    group_map: dict[str, Any],
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    columns = sorted(config["columns"])
    tasks = [(column, f"stats:{column}") for column in columns]
    results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def run_task(column: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        record, metrics = aggregate_stats_once(
            column,
            config,
            previous_stats.get(column),
            rows,
            valid_ids,
            group_map,
        )
        return column, record, metrics

    if workers <= 1:
        for column, _sort_key in tasks:
            name, record, metrics = run_task(column)
            results[name] = (record, metrics)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="iw-stats") as executor:
            futures = [executor.submit(run_task, column) for column, _sort_key in tasks]
            for future in futures:
                name, record, metrics = future.result()
                results[name] = (record, metrics)

    ordered_stats = {column: results[column][0] for column, _ in tasks}
    recomputed_columns: list[str] = []
    reused_columns: list[str] = []
    recomputed_groups: list[dict[str, str]] = []
    reused_groups: list[dict[str, str]] = []
    for column, _ in tasks:
        metrics = results[column][1]
        if metrics["computed"]:
            recomputed_columns.append(column)
        else:
            reused_columns.append(column)
        for group in metrics.get("recomputed_groups", []):
            recomputed_groups.append({"column": column, "group": group})
        for group in metrics.get("reused_groups", []):
            reused_groups.append({"column": column, "group": group})
    return ordered_stats, {
        "recomputed_columns": recomputed_columns,
        "reused_columns": reused_columns,
        "recomputed_groups": recomputed_groups,
        "reused_groups": reused_groups,
    }
