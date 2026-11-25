"""Deterministic in-memory tables generated from an integer seed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .config import ColumnSpec, LoadedBundle
from .hashing import canonical_json, sha256_hex


@dataclass
class TableData:
    table_name: str
    seed: int
    row_count: int
    columns: dict[str, ColumnSpec]
    values: dict[str, list[Any]]
    missing_bitmap: dict[str, list[bool]]
    column_fingerprints: dict[str, str]
    table_fingerprint: str
    replacements: set[str] = field(default_factory=set)

    def column_names(self) -> list[str]:
        return list(self.columns.keys())

    def categorical_columns(self) -> list[str]:
        return [name for name, spec in self.columns.items() if spec.dtype == "categorical"]

    def replace_column(
        self,
        name: str,
        values: Iterable[Any],
        missing_bitmap: Iterable[bool] | None = None,
    ) -> None:
        if name not in self.columns:
            raise KeyError(f"unknown column: {name}")
        materialized = [None if value is None else _python_value(value) for value in values]
        if len(materialized) != self.row_count:
            raise ValueError("replacement column length must equal row_count")
        if missing_bitmap is None:
            bitmap = [value is None for value in materialized]
        else:
            bitmap = [bool(value) for value in missing_bitmap]
            if len(bitmap) != self.row_count:
                raise ValueError("missing bitmap length must equal row_count")
            for index, missing in enumerate(bitmap):
                if missing:
                    materialized[index] = None
        self.values[name] = materialized
        self.missing_bitmap[name] = bitmap
        self.column_fingerprints[name] = _fingerprint_column(
            self.columns[name], materialized, bitmap
        )
        self.replacements.add(name)
        self.table_fingerprint = _table_fingerprint(
            self.row_count, self.column_fingerprints
        )


def generate_table(bundle: LoadedBundle | Mapping[str, Any]) -> TableData:
    if isinstance(bundle, LoadedBundle):
        raw = bundle.raw_features
        column_specs = list(bundle.columns)
    else:
        raw = dict(bundle)
        from .config import load_bundle_from_dicts

        parsed = load_bundle_from_dicts(raw, raw.get("strategy") or {})
        column_specs = list(parsed.columns)

    seed = raw.get("seed", 280280)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    row_count = raw.get("row_count", 240)
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count <= 0:
        raise ValueError("row_count must be a positive integer")
    table_name = str(raw.get("table_name", "deterministic_sample"))

    values: dict[str, list[Any]] = {}
    bitmaps: dict[str, list[bool]] = {}
    fingerprints: dict[str, str] = {}
    specs = {spec.name: spec for spec in column_specs}

    for spec in column_specs:
        rng = np.random.default_rng(_column_seed(seed, spec.name))
        if spec.dtype == "categorical":
            raw_values = _generate_categorical(rng, spec, row_count)
        else:
            raw_values = _generate_numeric(rng, spec, row_count)

        missing_draw = rng.random(row_count)
        bitmap = [bool(draw < spec.missing_rate) for draw in missing_draw]
        column_values: list[Any] = []
        for index, raw_value in enumerate(raw_values):
            if bitmap[index]:
                column_values.append(None)
            else:
                column_values.append(_python_value(raw_value))
        values[spec.name] = column_values
        bitmaps[spec.name] = bitmap
        fingerprints[spec.name] = _fingerprint_column(spec, column_values, bitmap)

    return TableData(
        table_name=table_name,
        seed=seed,
        row_count=row_count,
        columns=specs,
        values=values,
        missing_bitmap=bitmaps,
        column_fingerprints=fingerprints,
        table_fingerprint=_table_fingerprint(row_count, fingerprints),
    )


def _column_seed(master_seed: int, name: str) -> int:
    return int(sha256_hex("column-seed-v1", master_seed, name)[:8], 16)


def _generate_categorical(
    rng: np.random.Generator, spec: ColumnSpec, row_count: int
) -> list[str]:
    if spec.categories:
        categories = list(spec.categories)
    else:
        width = max(4, len(str(spec.cardinality)))
        categories = [
            f"{spec.name}__c{index:0{width}d}" for index in range(int(spec.cardinality))
        ]
    cardinality = len(categories)
    if spec.distribution == "zipf":
        draws = rng.zipf(spec.zipf_a, size=row_count)
        indexes = (draws.astype(np.int64) - 1) % cardinality
    else:
        indexes = rng.integers(0, cardinality, size=row_count)
    return [categories[int(index)] for index in indexes]


def _generate_numeric(
    rng: np.random.Generator, spec: ColumnSpec, row_count: int
) -> list[float]:
    if spec.distribution == "normal":
        values = rng.normal(spec.mean, spec.std, size=row_count)
    else:
        values = rng.uniform(spec.low, spec.high, size=row_count)
    return [float(value) for value in values]


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def _fingerprint_column(
    spec: ColumnSpec, values: list[Any], bitmap: list[bool]
) -> str:
    return sha256_hex(
        "column-content-v1",
        {
            "name": spec.name,
            "dtype": spec.dtype,
            "values": values,
            "missing_bitmap": bitmap,
        },
    )


def _table_fingerprint(
    row_count: int, column_fingerprints: Mapping[str, str]
) -> str:
    return sha256_hex(
        "table-content-v1",
        row_count,
        [
            [name, column_fingerprints[name]]
            for name in sorted(column_fingerprints)
        ],
    )


def quantize_numeric(
    values: list[Any],
    spec: ColumnSpec,
    quantization_levels: int,
    fixed_point_width: int,
) -> list[dict[str, int] | None]:
    """Quantize a numeric column deterministically.

    The fixed-point width contributes a stable scale mask while
    quantization_levels controls the diagnostic bin count.
    """

    span = spec.high - spec.low
    fraction_bits = max(0, min(16, fixed_point_width // 2))
    scale = 2**fraction_bits
    fixed_mask = (1 << fixed_point_width) - 1
    quantized: list[dict[str, int] | None] = []
    for value in values:
        if value is None:
            quantized.append(None)
            continue
        ratio = (float(value) - spec.low) / span if span else 0.0
        ratio = min(1.0, max(0.0, ratio))
        bin_index = min(quantization_levels - 1, int(ratio * quantization_levels))
        fixed_point_code = int(round(ratio * quantization_levels * scale)) & fixed_mask
        quantized.append(
            {
                "bin": bin_index,
                "fixed_point_code": fixed_point_code,
                "fraction_bits": fraction_bits,
            }
        )
    return quantized


def numeric_quantization_summary(table: TableData, levels: int, width: int) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name, spec in table.columns.items():
        if spec.dtype != "numeric":
            continue
        quantized = quantize_numeric(table.values[name], spec, levels, width)
        non_null = [value for value in quantized if value is not None]
        summaries[name] = {
            "quantization_levels": levels,
            "fixed_point_width": width,
            "fraction_bits": non_null[0]["fraction_bits"] if non_null else 0,
            "unique_bin_count": len({value["bin"] for value in non_null}),
            "unique_fixed_point_code_count": len(
                {value["fixed_point_code"] for value in non_null}
            ),
            "null_count": sum(value is None for value in quantized),
            "fingerprint": sha256_hex(
                "numeric-quantization-v2", name, levels, width, quantized
            ),
        }
    return summaries
