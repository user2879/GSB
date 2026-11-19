"""YAML column-description loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigLoadError

NUMERIC_KINDS = {"float", "int"}
CATEGORY_KINDS = {"category_low", "category_high"}
OTHER_KINDS = {"id", "redundant", "target"}
VALID_KINDS = NUMERIC_KINDS | CATEGORY_KINDS | OTHER_KINDS

TYPE_ALIASES = {
    "numeric": "float",
    "integer": "int",
    "categorical_low": "category_low",
    "categorical_high": "category_high",
    "low_cardinality": "category_low",
    "high_cardinality": "category_high",
    "duplicate": "redundant",
}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSpec:
    name: str
    seed: int
    target: str | None
    columns: list[ColumnSpec]
    source_path: str
    content_hash: str
    column_map: dict[str, ColumnSpec]

    @property
    def target_column(self) -> str | None:
        if self.target:
            return self.target
        for column in self.columns:
            if column.kind == "target":
                return column.name
        return None

    def category_columns(self) -> list[ColumnSpec]:
        return [column for column in self.columns if column.kind in CATEGORY_KINDS]

    def numeric_columns(self) -> list[ColumnSpec]:
        return [column for column in self.columns if column.kind in NUMERIC_KINDS | {"redundant"}]


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigLoadError(f"{field_name} must be a mapping", field=field_name)
    return value


def load_data_spec(path: str | Path) -> DataSpec:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigLoadError(f"cannot read data spec {source}: {exc}", field="data_spec") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"invalid data spec YAML: {exc}", field="data_spec") from exc
    if not isinstance(raw, dict):
        raise ConfigLoadError("data spec root must be a mapping", field="data_spec")

    raw_columns = raw.get("columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise ConfigLoadError("data spec columns must be a non-empty list", field="columns")

    columns: list[ColumnSpec] = []
    seen: set[str] = set()
    for index, raw_column in enumerate(raw_columns):
        if not isinstance(raw_column, dict):
            raise ConfigLoadError(f"columns[{index}] must be a mapping", field="columns")
        name = raw_column.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigLoadError(f"columns[{index}].name must be a non-empty string", field="columns")
        if name in seen:
            raise ConfigLoadError(f"duplicate column name: {name}", field="columns")
        seen.add(name)
        kind = raw_column.get("type")
        if not isinstance(kind, str):
            raise ConfigLoadError(f"column {name} is missing a string type", field="columns")
        kind = TYPE_ALIASES.get(kind, kind)
        if kind not in VALID_KINDS:
            raise ConfigLoadError(f"column {name} has unsupported type {kind!r}", field="columns")
        params = {key: value for key, value in raw_column.items() if key not in {"name", "type"}}
        columns.append(ColumnSpec(name=name, kind=kind, params=params))

    column_map = {column.name: column for column in columns}
    for column in columns:
        if column.kind == "redundant":
            source_name = column.params.get("source")
            if not isinstance(source_name, str) or source_name not in column_map:
                raise ConfigLoadError(
                    f"redundant column {column.name} references unknown source {source_name!r}",
                    field="columns",
                )
            if column_map[source_name].kind not in NUMERIC_KINDS | {"redundant"}:
                raise ConfigLoadError(
                    f"redundant column {column.name} source must be numeric",
                    field="columns",
                )

    target = raw.get("target")
    if target is not None and not isinstance(target, str):
        raise ConfigLoadError("data spec target must be a string", field="target")
    seed = raw.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigLoadError("data spec seed must be an integer", field="seed")
    name = raw.get("name", source.stem)
    if not isinstance(name, str) or not name:
        raise ConfigLoadError("data spec name must be a non-empty string", field="name")

    return DataSpec(
        name=name,
        seed=seed,
        target=target,
        columns=columns,
        source_path=str(source),
        content_hash=__import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
        column_map=column_map,
    )
