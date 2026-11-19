"""Deterministic local sample generation, loading and row-level validation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import digest
from .errors import E_BUCKET, E_SPEC, fail
from .spec import ColumnSpec, SourceSpec, Spec

DATA_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SourceData:
    source_id: str
    raw_rows: tuple[Mapping[str, int | float | None], ...]
    null_flags: Mapping[str, tuple[bool, ...]]
    row_keys: tuple[str, ...]
    primary_key_name: str
    digest: str

    @property
    def rows(self) -> int:
        return len(self.raw_rows)

    def value(self, row: int, column: str) -> int | float | None:
        return self.raw_rows[row][column]

    def is_null(self, row: int, column: str) -> bool:
        return self.null_flags[column][row]


def _column_rng(source: SourceSpec, column: ColumnSpec) -> random.Random:
    material = f"bucketpack|{source.id}|{source.seed}|{column.name}".encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(seed)


def generate_rows(source: SourceSpec) -> list[dict[str, int | float]]:
    """Generate each column from an independently derived integer seed.

    Deriving a column seed from ``(source seed, source id, column name)`` keeps
    a column stable when unrelated columns are added or reordered.
    """

    columns = list(source.columns)
    rngs = {column.name: _column_rng(source, column) for column in columns}
    rows: list[dict[str, int | float]] = []
    for _ in range(source.rows):
        row: dict[str, int | float] = {}
        for column in columns:
            rng = rngs[column.name]
            if column.type == "int":
                value: int | float = rng.randint(int(column.minimum), int(column.maximum))
            else:
                value = round(rng.uniform(float(column.minimum), float(column.maximum)), column.precision)
                if value == -0.0:
                    value = 0.0
            row[column.name] = value
        rows.append(row)
    return rows


def _resolve_data_file(spec: Spec, source: SourceSpec) -> Path:
    assert source.data_file is not None
    path = Path(source.data_file)
    if not path.is_absolute():
        path = Path(spec.base_dir) / path
    return path.resolve()


def _validate_value(
    source: SourceSpec,
    column: ColumnSpec,
    value: Any,
    *,
    row: int,
    origin: str,
) -> int | float | None:
    path = f"{origin}[{row}].{column.name}"
    if value is None:
        return None
    if column.type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            fail(
                E_BUCKET,
                path,
                "sample value conflicts with int column type",
                source_id=source.id,
                column=column.name,
                expected="int",
                actual=type(value).__name__,
            )
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(
            E_BUCKET,
            path,
            "sample value conflicts with float column type",
            source_id=source.id,
            column=column.name,
            expected="float",
            actual=type(value).__name__,
        )
    try:
        as_float = float(value)
    except (OverflowError, ValueError):
        fail(
            E_BUCKET,
            path,
            "sample value conflicts with float column type",
            source_id=source.id,
            column=column.name,
            expected="finite float",
            actual=type(value).__name__,
        )
    if not math.isfinite(as_float):
        fail(
            E_BUCKET,
            path,
            "sample float value must be finite",
            source_id=source.id,
            column=column.name,
        )
    return as_float


def _load_file_rows(spec: Spec, source: SourceSpec) -> list[dict[str, Any]]:
    path = _resolve_data_file(spec, source)
    try:
        def reject_constant(value: str) -> Any:
            fail(
                E_BUCKET,
                "data",
                "non-finite JSON numbers are not allowed in sample data",
                value=value,
                file=str(path),
            )

        raw = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except OSError as exc:
        fail(
            E_SPEC,
            f"$.sources[id={source.id}].data_file",
            "cannot read local sample data file",
            file=str(path),
            error=str(exc),
        )
    except json.JSONDecodeError as exc:
        fail(
            E_SPEC,
            f"$.sources[id={source.id}].data_file",
            "sample data file is not valid JSON",
            file=str(path),
            line=exc.lineno,
            column=exc.colno,
        )
    if not isinstance(raw, dict):
        fail(E_BUCKET, "$", "sample data file root must be an object", file=str(path))
    if raw.get("format_version") != DATA_FORMAT_VERSION:
        fail(
            E_SPEC,
            "data.format_version",
            f"sample data format version must be {DATA_FORMAT_VERSION}",
            file=str(path),
        )
    if raw.get("source_id") != source.id:
        fail(
            E_SPEC,
            "data.source_id",
            "sample data source_id does not match the source declaration",
            file=str(path),
        )
    rows = raw.get("rows")
    if not isinstance(rows, list):
        fail(E_BUCKET, "data.rows", "sample data rows must be an array", file=str(path))
    if len(rows) != source.rows:
        fail(
            E_BUCKET,
            "data.rows",
            "sample data row count does not match the source declaration",
            expected=source.rows,
            actual=len(rows),
        )
    return rows


def load_source(
    spec: Spec,
    source: SourceSpec,
    *,
    value_overrides: Mapping[tuple[int, str], Any] | None = None,
    null_overrides: Mapping[tuple[int, str], bool] | None = None,
) -> SourceData:
    """Load one source and apply explicit incremental change overrides."""

    value_overrides = dict(value_overrides or {})
    null_overrides = dict(null_overrides or {})
    if source.data_file is None:
        rows: list[dict[str, Any]] = generate_rows(source)
        origin = f"sources[id={source.id}].generated_rows"
    else:
        rows = _load_file_rows(spec, source)
        origin = "data.rows"

    columns = {column.name: column for column in source.columns}
    normalized_rows: list[dict[str, int | float | None]] = []
    null_flags: dict[str, list[bool]] = {name: [] for name in columns}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            fail(
                E_BUCKET,
                f"{origin}[{row_index}]",
                "sample row must be an object",
                source_id=source.id,
            )
        extra = sorted(set(row) - set(columns))
        if extra:
            fail(
                E_BUCKET,
                f"{origin}[{row_index}].{extra[0]}",
                "sample row contains an undeclared column",
                source_id=source.id,
            )
        normalized: dict[str, int | float | None] = {}
        for name in sorted(columns):
            if name not in row:
                fail(
                    E_BUCKET,
                    f"{origin}[{row_index}].{name}",
                    "sample row is missing a declared column",
                    source_id=source.id,
                )
            bit = bool(source.null_bitmaps[name][row_index])
            if (row_index, name) in null_overrides:
                bit = bool(null_overrides[(row_index, name)])
            raw_value = value_overrides.get((row_index, name), row[name])
            value = _validate_value(
                source,
                columns[name],
                raw_value,
                row=row_index,
                origin=origin,
            )
            if bit:
                if value is not None and (row_index, name) not in value_overrides:
                    # A bitmap may deliberately hide a stored value.  Treat the
                    # bitmap as authoritative.
                    value = None
            elif value is None:
                fail(
                    E_BUCKET,
                    f"{origin}[{row_index}].{name}",
                    "null value requires a matching null bitmap bit",
                    source_id=source.id,
                    column=name,
                )
            normalized[name] = value
            null_flags[name].append(bit)
        normalized_rows.append(normalized)

    for (row_index, column_name) in sorted(value_overrides):
        if row_index < 0 or row_index >= source.rows:
            fail(
                E_BUCKET,
                f"changes.values[{row_index}].row",
                "change row index is outside the source",
                source_id=source.id,
            )
        if column_name not in columns:
            fail(
                E_BUCKET,
                f"changes.values[{row_index}].column",
                "change column does not exist",
                source_id=source.id,
            )
    for (row_index, column_name) in sorted(null_overrides):
        if row_index < 0 or row_index >= source.rows or column_name not in columns:
            fail(E_BUCKET, "changes.nulls", "null change target is outside the source schema")

    key_name = source.primary_key
    row_keys: list[str] = []
    seen_keys: dict[str, int] = {}
    for row_index, row in enumerate(normalized_rows):
        key_value = row[key_name]
        if key_value is None:
            fail(
                E_SPEC,
                f"sources[id={source.id}].primary_key",
                "primary key value cannot be null",
                row=row_index,
            )
        key = str(key_value)
        if key in seen_keys:
            fail(
                E_SPEC,
                f"sources[id={source.id}].primary_key",
                "primary key values must be unique",
                first_row=seen_keys[key],
                row=row_index,
                key=key,
            )
        seen_keys[key] = row_index
        row_keys.append(key)

    digest_value = {
        "source_id": source.id,
        "rows": normalized_rows,
        "nulls": {
            name: "".join("1" if bit else "0" for bit in null_flags[name])
            for name in sorted(null_flags)
        },
    }
    return SourceData(
        source_id=source.id,
        raw_rows=tuple(normalized_rows),
        null_flags={name: tuple(values) for name, values in sorted(null_flags.items())},
        row_keys=tuple(row_keys),
        primary_key_name=key_name,
        digest=digest(digest_value),
    )


def materialize_source_data(spec: Spec, source: SourceSpec) -> dict[str, Any]:
    """Return the deterministic local sample file payload for one source."""

    data = load_source(spec, source)
    return {
        "format_version": DATA_FORMAT_VERSION,
        "source_id": source.id,
        "seed": source.seed,
        "rows": [dict(row) for row in data.raw_rows],
    }
