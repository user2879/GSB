"""Single source of truth for stage cache keys and group content fingerprints."""

from __future__ import annotations

from typing import Iterable

from .config import CalibrationConfig
from .hashing import fingerprint_rows, sha256_value, stable_cache_key
from .table import FeatureTable


def column_content_fingerprint(table: FeatureTable, group_id: str, column: str) -> str:
    return fingerprint_rows(
        (
            (row.row_id, row.group_id, {column: row.values.get(column)})
            for row in table.rows_for_group(group_id)
        ),
        [column],
        include_group=False,
    )


def group_content_fingerprint(
    table: FeatureTable,
    group_id: str,
    columns: Iterable[str],
) -> str:
    typed_columns = tuple(sorted(columns))
    return fingerprint_rows(
        (
            (row.row_id, row.group_id, {column: row.values.get(column) for column in typed_columns})
            for row in table.rows_for_group(group_id)
        ),
        typed_columns,
        include_group=False,
    )


def calibration_parameter_fingerprint(
    config: CalibrationConfig,
    table: FeatureTable,
    group_id: str,
    columns: Iterable[str],
    bit_width: int,
) -> str:
    typed_columns = tuple(sorted(columns))
    column_fingerprints = {
        column: column_content_fingerprint(table, group_id, column) for column in typed_columns
    }
    return sha256_value(
        {
            "group_id": group_id,
            "columns": list(typed_columns),
            "calibration": config.calibration_parameters(),
            "bit_width": int(bit_width),
            "fractional_bits": config.quantization.fractional_bits,
            "column_content_fingerprints": column_fingerprints,
        }
    )


def calibration_keys(
    config: CalibrationConfig,
    table: FeatureTable,
    group_id: str,
    columns: Iterable[str],
    bit_width: int,
) -> dict[str, str]:
    typed_columns = tuple(sorted(columns))
    content_fingerprint = group_content_fingerprint(table, group_id, typed_columns)
    parameter_fingerprint = calibration_parameter_fingerprint(
        config, table, group_id, typed_columns, bit_width
    )
    cache_key = stable_cache_key(
        "quantile_calibration",
        content_fingerprint=content_fingerprint,
        parameter_fingerprint=parameter_fingerprint,
        bit_width=bit_width,
        fractional_bits=config.quantization.fractional_bits,
    )
    return {
        "content_fingerprint": content_fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
        "cache_key": cache_key,
    }


def fixed_rescale_parameter_fingerprint(
    config: CalibrationConfig,
    calibration_parameter_fp: str,
    calibration_cache_key: str,
) -> str:
    return sha256_value(
        {
            "calibration_parameter_fingerprint": calibration_parameter_fp,
            "calibration_cache_key": calibration_cache_key,
            "clipping_strategy": config.clipping_strategy,
        }
    )


def fixed_rescale_keys(
    config: CalibrationConfig,
    *,
    content_fingerprint: str,
    calibration_parameter_fingerprint: str,
    calibration_cache_key: str,
    bit_width: int,
) -> dict[str, str]:
    parameter_fingerprint = fixed_rescale_parameter_fingerprint(
        config, calibration_parameter_fingerprint, calibration_cache_key
    )
    cache_key = stable_cache_key(
        "fixed_rescale",
        content_fingerprint=content_fingerprint,
        parameter_fingerprint=parameter_fingerprint,
        bit_width=bit_width,
        fractional_bits=config.quantization.fractional_bits,
        upstream_key=calibration_cache_key,
        stage_parameters={"clipping_strategy": config.clipping_strategy},
    )
    return {"parameter_fingerprint": parameter_fingerprint, "cache_key": cache_key}


def attribution_parameter_fingerprint(
    config: CalibrationConfig,
    rescale_cache_key: str,
) -> str:
    return sha256_value(
        {
            "rescale_cache_key": rescale_cache_key,
            "tolerance": {
                "absolute": config.tolerance.absolute,
                "relative": config.tolerance.relative,
                "comparison": config.tolerance.comparison,
                "zero_relative_branch": config.tolerance.zero_relative_branch,
                "tie_epsilon": config.tolerance.tie_epsilon,
            },
        }
    )


def attribution_keys(
    config: CalibrationConfig,
    *,
    content_fingerprint: str,
    rescale_cache_key: str,
    bit_width: int,
) -> dict[str, str]:
    parameter_fingerprint = attribution_parameter_fingerprint(config, rescale_cache_key)
    cache_key = stable_cache_key(
        "error_attribution",
        content_fingerprint=content_fingerprint,
        parameter_fingerprint=parameter_fingerprint,
        bit_width=bit_width,
        fractional_bits=config.quantization.fractional_bits,
        upstream_key=rescale_cache_key,
        stage_parameters={
            "absolute": config.tolerance.absolute,
            "relative": config.tolerance.relative,
            "comparison": config.tolerance.comparison,
            "tie_epsilon": config.tolerance.tie_epsilon,
        },
    )
    return {"parameter_fingerprint": parameter_fingerprint, "cache_key": cache_key}


def sensitivity_keys(
    config: CalibrationConfig,
    *,
    content_fingerprint: str,
    attribution_cache_key: str,
    bit_width: int,
) -> dict[str, str]:
    parameter_fingerprint = sha256_value(
        {
            "attribution_cache_key": attribution_cache_key,
            "outlier_ratio_limit": config.tolerance.outlier_ratio_limit,
        }
    )
    cache_key = stable_cache_key(
        "sensitivity",
        content_fingerprint=content_fingerprint,
        parameter_fingerprint=parameter_fingerprint,
        bit_width=bit_width,
        fractional_bits=config.quantization.fractional_bits,
        upstream_key=attribution_cache_key,
        stage_parameters={"outlier_ratio_limit": config.tolerance.outlier_ratio_limit},
    )
    return {"parameter_fingerprint": parameter_fingerprint, "cache_key": cache_key}


def expected_stage_keys(
    config: CalibrationConfig,
    table: FeatureTable,
    group_id: str,
    columns: Iterable[str],
    bit_width: int,
) -> dict[str, str]:
    calibration = calibration_keys(config, table, group_id, columns, bit_width)
    fixed = fixed_rescale_keys(
        config,
        content_fingerprint=calibration["content_fingerprint"],
        calibration_parameter_fingerprint=calibration["parameter_fingerprint"],
        calibration_cache_key=calibration["cache_key"],
        bit_width=bit_width,
    )
    attribution = attribution_keys(
        config,
        content_fingerprint=calibration["content_fingerprint"],
        rescale_cache_key=fixed["cache_key"],
        bit_width=bit_width,
    )
    sensitivity = sensitivity_keys(
        config,
        content_fingerprint=calibration["content_fingerprint"],
        attribution_cache_key=attribution["cache_key"],
        bit_width=bit_width,
    )
    return {
        "quantile_calibration": calibration["cache_key"],
        "fixed_rescale": fixed["cache_key"],
        "error_attribution": attribution["cache_key"],
        "sensitivity": sensitivity["cache_key"],
    }
