"""Strongly validated, order-independent pipeline specifications."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import digest
from .errors import E_SPEC, BusinessError, fail

SPEC_FORMAT_VERSION = 1
SOURCE_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
COLUMN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
RULE_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
SUPPORTED_TYPES = ("int", "float")
INPUT_KINDS = ("column", "bucket")
STAT_METRICS = ("sum_counts",)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        fail(E_SPEC, path, "must be an object")
    return value


def _require_list(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        fail(E_SPEC, path, "must be an array")
    return value


def _check_keys(
    obj: Mapping[str, Any],
    path: str,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    allowed = set(required) | set(optional)
    for key in sorted(obj, key=lambda item: str(item)):
        if not isinstance(key, str) or key not in allowed:
            fail(E_SPEC, f"{path}.{key}" if path else key, "unknown field")
    for key in required:
        if key not in obj:
            fail(E_SPEC, f"{path}.{key}" if path else key, "required field is missing")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: str
    minimum: int | float
    maximum: int | float
    precision: int
    null_fill: int | float

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "min": self.minimum,
            "max": self.maximum,
            "null_fill": self.null_fill,
        }
        if self.type == "float":
            out["precision"] = self.precision
        return out


@dataclass(frozen=True)
class SourceSpec:
    id: str
    seed: int
    rows: int
    primary_key: str
    columns: tuple[ColumnSpec, ...]
    null_bitmaps: Mapping[str, tuple[bool, ...]]
    data_file: str | None = None

    def column(self, name: str) -> ColumnSpec | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "seed": self.seed,
            "rows": self.rows,
            "primary_key": self.primary_key,
            "columns": [column.to_dict() for column in self.columns],
            "null_bitmaps": {
                name: "".join("1" if bit else "0" for bit in self.null_bitmaps[name])
                for name in sorted(self.null_bitmaps)
            },
        }
        if self.data_file is not None:
            out["data_file"] = self.data_file
        return out


@dataclass(frozen=True)
class RuleSpec:
    id: str
    source: str
    input_kind: str
    input_name: str
    boundaries: tuple[int, ...]
    scale: int
    width_bits: int | None = None

    @property
    def bucket_count(self) -> int:
        return len(self.boundaries) + 1

    @property
    def effective_width_bits(self) -> int:
        if self.width_bits is not None:
            return self.width_bits
        return max(1, (self.bucket_count - 1).bit_length())

    @property
    def cell_width(self) -> int:
        # One presence/null bit plus the fixed-width bucket-id slot.
        return self.effective_width_bits + 1

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "input": {"kind": self.input_kind, "name": self.input_name},
            "boundaries": list(self.boundaries),
            "scale": self.scale,
        }
        if self.width_bits is not None:
            out["width_bits"] = self.width_bits
        return out


@dataclass(frozen=True)
class StatSpec:
    id: str
    source: str
    refs: tuple[str, ...]
    metric: str = "sum_counts"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "refs": list(self.refs),
            "metric": self.metric,
        }


@dataclass(frozen=True)
class PackingSpec:
    rows_per_pack: int
    pack_width_budget: int
    global_width_budget: int
    pack_quota_per_source: int
    workers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_per_pack": self.rows_per_pack,
            "pack_width_budget": self.pack_width_budget,
            "global_width_budget": self.global_width_budget,
            "pack_quota_per_source": self.pack_quota_per_source,
            "workers": self.workers,
        }


@dataclass(frozen=True)
class AccountingSpec:
    unit_cost_micros: int
    budget_micros: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_cost_micros": self.unit_cost_micros,
            "budget_micros": self.budget_micros,
        }


@dataclass(frozen=True)
class Spec:
    sources: tuple[SourceSpec, ...]
    rules: tuple[RuleSpec, ...]
    statistics: tuple[StatSpec, ...]
    packing: PackingSpec
    accounting: AccountingSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format_version: int = SPEC_FORMAT_VERSION
    base_dir: str = ""

    def source(self, source_id: str) -> SourceSpec | None:
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def rule(self, rule_id: str) -> RuleSpec | None:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def rules_for_source(self, source_id: str) -> tuple[RuleSpec, ...]:
        return tuple(rule for rule in self.rules if rule.source == source_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "sources": [source.to_dict() for source in self.sources],
            "rules": [rule.to_dict() for rule in self.rules],
            "statistics": [stat.to_dict() for stat in self.statistics],
            "packing": self.packing.to_dict(),
            "accounting": self.accounting.to_dict(),
            "metadata": dict(self.metadata),
        }

    @property
    def digest(self) -> str:
        return digest(self.to_dict())

    def replace(self, **changes: Any) -> Spec:
        values = {
            "sources": self.sources,
            "rules": self.rules,
            "statistics": self.statistics,
            "packing": self.packing,
            "accounting": self.accounting,
            "metadata": self.metadata,
            "format_version": self.format_version,
            "base_dir": self.base_dir,
        }
        values.update(changes)
        return Spec(**values)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise BusinessError(
                code=E_SPEC,
                path="$",
                message=f"duplicate object key: {key}",
                details={"key": key},
            )
        out[key] = value
    return out


def _reject_json_constant(value: str) -> Any:
    raise BusinessError(
        code=E_SPEC,
        path="$",
        message="non-finite JSON numbers are not allowed",
        details={"value": value},
    )


def _validate_bitmap(value: Any, path: str, rows: int) -> tuple[bool, ...]:
    if isinstance(value, str):
        if any(ch not in "01" for ch in value):
            fail(E_SPEC, path, "null bitmap may contain only '0' and '1'")
        if len(value) != rows:
            fail(
                E_SPEC,
                path,
                "null bitmap length must equal source rows",
                expected_rows=rows,
                actual_length=len(value),
            )
        return tuple(ch == "1" for ch in value)
    if isinstance(value, list):
        if len(value) != rows:
            fail(
                E_SPEC,
                path,
                "null bitmap length must equal source rows",
                expected_rows=rows,
                actual_length=len(value),
            )
        bits: list[bool] = []
        for idx, bit in enumerate(value):
            if bit not in (0, 1) or isinstance(bit, bool):
                fail(E_SPEC, f"{path}[{idx}]", "null bitmap element must be integer 0 or 1")
            bits.append(bool(bit))
        return tuple(bits)
    fail(E_SPEC, path, "null bitmap must be a string or an array of 0/1")


def _validate_column(value: Any, path: str) -> ColumnSpec:
    obj = _require_mapping(value, path)
    _check_keys(obj, path, required=("name", "type", "min", "max"), optional=("precision", "null_fill"))
    name = obj["name"]
    if not isinstance(name, str) or not COLUMN_NAME_RE.fullmatch(name):
        fail(E_SPEC, f"{path}.name", "column name must match ^[A-Za-z_][A-Za-z0-9_]*$")
    type_name = obj["type"]
    if type_name not in SUPPORTED_TYPES:
        fail(E_SPEC, f"{path}.type", "supported column types are int and float")
    minimum = obj["min"]
    maximum = obj["max"]
    if type_name == "int":
        if not _is_int(minimum):
            fail(E_SPEC, f"{path}.min", "int column min must be an integer")
        if not _is_int(maximum):
            fail(E_SPEC, f"{path}.max", "int column max must be an integer")
        if "precision" in obj:
            fail(E_SPEC, f"{path}.precision", "precision is only valid for float columns")
        if minimum > maximum:
            fail(E_SPEC, f"{path}.min", "min must be less than or equal to max")
        null_fill = obj.get("null_fill", minimum)
        if not _is_int(null_fill):
            fail(E_SPEC, f"{path}.null_fill", "int column null_fill must be an integer")
        return ColumnSpec(
            name=name,
            type=type_name,
            minimum=minimum,
            maximum=maximum,
            precision=0,
            null_fill=null_fill,
        )

    if not _is_number(minimum):
        fail(E_SPEC, f"{path}.min", "float column min must be a finite number")
    if not _is_number(maximum):
        fail(E_SPEC, f"{path}.max", "float column max must be a finite number")
    if float(minimum) > float(maximum):
        fail(E_SPEC, f"{path}.min", "min must be less than or equal to max")
    precision = obj.get("precision", 6)
    if not _is_int(precision) or precision < 0 or precision > 9:
        fail(E_SPEC, f"{path}.precision", "precision must be an integer in 0..9")
    null_fill = obj.get("null_fill", float(minimum))
    if not _is_number(null_fill):
        fail(E_SPEC, f"{path}.null_fill", "float column null_fill must be a finite number")
    return ColumnSpec(
        name=name,
        type=type_name,
        minimum=float(minimum),
        maximum=float(maximum),
        precision=precision,
        null_fill=float(null_fill),
    )


def _validate_source(value: Any, path: str) -> SourceSpec:
    obj = _require_mapping(value, path)
    _check_keys(
        obj,
        path,
        required=("id", "seed", "rows", "primary_key", "columns", "null_bitmaps"),
        optional=("data_file",),
    )
    source_id = obj["id"]
    if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
        fail(E_SPEC, f"{path}.id", "source id must match ^[A-Za-z_][A-Za-z0-9_.-]*$")
    seed = obj["seed"]
    if not _is_int(seed):
        fail(E_SPEC, f"{path}.seed", "seed must be an integer")
    rows = obj["rows"]
    if not _is_int(rows) or rows <= 0:
        fail(E_SPEC, f"{path}.rows", "rows must be a positive integer")
    primary_key = obj["primary_key"]
    if not isinstance(primary_key, str) or not primary_key:
        fail(E_SPEC, f"{path}.primary_key", "primary_key must be a non-empty string")
    columns_raw = _require_list(obj["columns"], f"{path}.columns")
    if not columns_raw:
        fail(E_SPEC, f"{path}.columns", "at least one column is required")
    columns: list[ColumnSpec] = []
    seen_columns: dict[str, int] = {}
    for idx, item in enumerate(columns_raw):
        column = _validate_column(item, f"{path}.columns[{idx}]")
        if column.name in seen_columns:
            fail(
                E_SPEC,
                f"{path}.columns[{idx}].name",
                "duplicate column name",
                first_index=seen_columns[column.name],
                name=column.name,
            )
        seen_columns[column.name] = idx
        columns.append(column)
    primary = next((col for col in columns if col.name == primary_key), None)
    if primary is None:
        fail(E_SPEC, f"{path}.primary_key", "primary_key must name a declared column")
    if primary.type != "int":
        fail(E_SPEC, f"{path}.primary_key", "primary_key column must have type int")

    bitmaps_raw = _require_mapping(obj["null_bitmaps"], f"{path}.null_bitmaps")
    bitmaps: dict[str, tuple[bool, ...]] = {}
    column_names = {col.name for col in columns}
    for column_name in sorted(bitmaps_raw):
        if column_name not in column_names:
            fail(E_SPEC, f"{path}.null_bitmaps.{column_name}", "unknown column")
        bitmaps[column_name] = _validate_bitmap(
            bitmaps_raw[column_name], f"{path}.null_bitmaps.{column_name}", rows
        )
    for column in columns:
        bitmaps.setdefault(column.name, (False,) * rows)

    data_file = obj.get("data_file")
    if data_file is not None:
        if not isinstance(data_file, str) or not data_file:
            fail(E_SPEC, f"{path}.data_file", "data_file must be a non-empty local path string")
        lowered = data_file.lower()
        if "://" in lowered or lowered.startswith(("http:", "https:")):
            fail(E_SPEC, f"{path}.data_file", "remote URLs are not allowed")

    return SourceSpec(
        id=source_id,
        seed=seed,
        rows=rows,
        primary_key=primary_key,
        columns=tuple(sorted(columns, key=lambda c: c.name)),
        null_bitmaps=bitmaps,
        data_file=data_file,
    )


def _validate_input(
    value: Any,
    path: str,
    source: SourceSpec,
    all_rules: Mapping[str, Mapping[str, Any]],
    current_rule_id: str,
) -> tuple[str, str]:
    if isinstance(value, str):
        name = value
        if not name:
            fail(E_SPEC, path, "input name must be non-empty")
        in_columns = source.column(name) is not None
        in_buckets = name in all_rules
        if in_columns and in_buckets:
            fail(E_SPEC, path, "ambiguous input name; use an explicit kind")
        if in_columns:
            return "column", name
        if in_buckets:
            return "bucket", name
        fail(E_SPEC, path, "input name does not name a source column or bucket column")
    obj = _require_mapping(value, path)
    _check_keys(obj, path, required=("kind", "name"))
    kind = obj["kind"]
    name = obj["name"]
    if kind not in INPUT_KINDS:
        fail(E_SPEC, f"{path}.kind", "input kind must be column or bucket")
    if not isinstance(name, str) or not name:
        fail(E_SPEC, f"{path}.name", "input name must be a non-empty string")
    if name == current_rule_id:
        # This is a valid reference shape; graph validation reports the cycle.
        return kind, name
    if kind == "column" and source.column(name) is None:
        fail(E_SPEC, f"{path}.name", "input column does not exist in the source")
    return kind, name


def _validate_rule(
    value: Any,
    path: str,
    sources: Mapping[str, SourceSpec],
    all_rules: Mapping[str, Mapping[str, Any]],
) -> RuleSpec:
    obj = _require_mapping(value, path)
    _check_keys(
        obj,
        path,
        required=("id", "source", "input", "boundaries"),
        optional=("scale", "width_bits"),
    )
    rule_id = obj["id"]
    if not isinstance(rule_id, str) or not RULE_ID_RE.fullmatch(rule_id):
        fail(E_SPEC, f"{path}.id", "bucket column id must match ^[A-Za-z_][A-Za-z0-9_.-]*$")
    source_id = obj["source"]
    if not isinstance(source_id, str) or source_id not in sources:
        fail(E_SPEC, f"{path}.source", "rule source does not exist")
    source = sources[source_id]
    input_kind, input_name = _validate_input(
        obj["input"], f"{path}.input", source, all_rules, rule_id
    )
    boundaries_raw = _require_list(obj["boundaries"], f"{path}.boundaries")
    if not boundaries_raw:
        fail(E_SPEC, f"{path}.boundaries", "at least one integer boundary is required")
    boundaries: list[int] = []
    previous: int | None = None
    for idx, boundary in enumerate(boundaries_raw):
        if not _is_int(boundary):
            fail(E_SPEC, f"{path}.boundaries[{idx}]", "bucket boundary must be an integer")
        if previous is not None and boundary <= previous:
            fail(
                E_SPEC,
                f"{path}.boundaries[{idx}]",
                "boundaries must be strictly increasing and unique",
                previous=previous,
                value=boundary,
            )
        boundaries.append(boundary)
        previous = boundary
    scale = obj.get("scale", 0)
    if not _is_int(scale) or scale < 0 or scale > 9:
        fail(E_SPEC, f"{path}.scale", "scale must be an integer in 0..9")
    width_bits = obj.get("width_bits")
    if width_bits is not None and (not _is_int(width_bits) or width_bits <= 0 or width_bits > 64):
        fail(E_SPEC, f"{path}.width_bits", "width_bits must be an integer in 1..64")
    minimum_width = len(boundaries).bit_length()
    if width_bits is not None and width_bits < minimum_width:
        fail(
            E_SPEC,
            f"{path}.width_bits",
            "width_bits cannot represent every bucket id",
            bucket_count=len(boundaries) + 1,
            minimum_width_bits=minimum_width,
        )
    return RuleSpec(
        id=rule_id,
        source=source_id,
        input_kind=input_kind,
        input_name=input_name,
        boundaries=tuple(boundaries),
        scale=scale,
        width_bits=width_bits,
    )


def _validate_stat(
    value: Any,
    path: str,
    sources: Mapping[str, SourceSpec],
    rule_sources: Mapping[str, str],
) -> StatSpec:
    obj = _require_mapping(value, path)
    _check_keys(obj, path, required=("id", "source", "refs"), optional=("metric",))
    stat_id = obj["id"]
    if not isinstance(stat_id, str) or not RULE_ID_RE.fullmatch(stat_id):
        fail(E_SPEC, f"{path}.id", "statistic id must match ^[A-Za-z_][A-Za-z0-9_.-]*$")
    source_id = obj["source"]
    if not isinstance(source_id, str) or source_id not in sources:
        fail(E_SPEC, f"{path}.source", "statistic source does not exist")
    refs_raw = _require_list(obj["refs"], f"{path}.refs")
    if not refs_raw:
        fail(E_SPEC, f"{path}.refs", "at least one bucket reference is required")
    refs: list[str] = []
    seen: set[str] = set()
    for idx, ref in enumerate(refs_raw):
        if not isinstance(ref, str):
            fail(E_SPEC, f"{path}.refs[{idx}]", "bucket reference must be a string")
        if ref in seen:
            fail(E_SPEC, f"{path}.refs[{idx}]", "duplicate bucket reference")
        if ref in rule_sources and rule_sources[ref] != source_id:
            fail(E_SPEC, f"{path}.refs[{idx}]", "bucket reference belongs to another source")
        seen.add(ref)
        refs.append(ref)
    metric = obj.get("metric", "sum_counts")
    if metric not in STAT_METRICS:
        fail(E_SPEC, f"{path}.metric", "unsupported statistic metric")
    return StatSpec(id=stat_id, source=source_id, refs=tuple(refs), metric=metric)


def _validate_packing(value: Any, path: str) -> PackingSpec:
    obj = _require_mapping(value, path)
    _check_keys(
        obj,
        path,
        required=(
            "rows_per_pack",
            "pack_width_budget",
            "global_width_budget",
            "pack_quota_per_source",
            "workers",
        ),
    )
    fields: dict[str, int] = {}
    for key in (
        "rows_per_pack",
        "pack_width_budget",
        "global_width_budget",
        "pack_quota_per_source",
        "workers",
    ):
        value_int = obj[key]
        if not _is_int(value_int) or value_int <= 0:
            fail(E_SPEC, f"{path}.{key}", "must be a positive integer")
        fields[key] = value_int
    return PackingSpec(**fields)


def _validate_accounting(value: Any, path: str) -> AccountingSpec:
    obj = _require_mapping(value, path)
    _check_keys(obj, path, required=("unit_cost_micros", "budget_micros"))
    unit = obj["unit_cost_micros"]
    budget = obj["budget_micros"]
    if not _is_int(unit) or unit < 0:
        fail(E_SPEC, f"{path}.unit_cost_micros", "must be a non-negative integer")
    if not _is_int(budget) or budget <= 0:
        fail(E_SPEC, f"{path}.budget_micros", "must be a positive integer")
    return AccountingSpec(unit_cost_micros=unit, budget_micros=budget)


def validate_spec(raw: Any, *, base_dir: str = "") -> Spec:
    obj = _require_mapping(raw, "$")
    _check_keys(
        obj,
        "$",
        required=("format_version", "sources", "rules", "packing", "accounting"),
        optional=("statistics", "metadata"),
    )
    version = obj["format_version"]
    if not _is_int(version) or version != SPEC_FORMAT_VERSION:
        fail(
            E_SPEC,
            "$.format_version",
            f"unsupported format version; expected {SPEC_FORMAT_VERSION}",
            actual=version,
        )

    sources_raw = _require_list(obj["sources"], "$.sources")
    if not sources_raw:
        fail(E_SPEC, "$.sources", "at least one data source is required")
    sources: list[SourceSpec] = []
    seen_sources: dict[str, int] = {}
    for idx, item in enumerate(sources_raw):
        source = _validate_source(item, f"$.sources[{idx}]")
        if source.id in seen_sources:
            fail(
                E_SPEC,
                f"$.sources[{idx}].id",
                "duplicate data source id",
                first_index=seen_sources[source.id],
                source_id=source.id,
            )
        seen_sources[source.id] = idx
        sources.append(source)
    source_map = {source.id: source for source in sources}

    rules_raw = _require_list(obj["rules"], "$.rules")
    if not rules_raw:
        fail(E_SPEC, "$.rules", "at least one bucket rule is required")
    rule_objects: dict[str, Mapping[str, Any]] = {}
    for idx, item in enumerate(rules_raw):
        rule_obj = _require_mapping(item, f"$.rules[{idx}]")
        rule_id = rule_obj.get("id")
        if isinstance(rule_id, str):
            if rule_id in rule_objects:
                fail(E_SPEC, f"$.rules[{idx}].id", "duplicate bucket column id", rule_id=rule_id)
            rule_objects[rule_id] = rule_obj
    rules: list[RuleSpec] = []
    for idx, item in enumerate(rules_raw):
        rules.append(_validate_rule(item, f"$.rules[{idx}]", source_map, rule_objects))
    rule_sources = {rule.id: rule.source for rule in rules}

    stats_raw = obj.get("statistics", [])
    stats_list = _require_list(stats_raw, "$.statistics")
    statistics: list[StatSpec] = []
    seen_stats: dict[str, int] = {}
    for idx, item in enumerate(stats_list):
        stat = _validate_stat(item, f"$.statistics[{idx}]", source_map, rule_sources)
        if stat.id in seen_stats:
            fail(
                E_SPEC,
                f"$.statistics[{idx}].id",
                "duplicate statistic id",
                first_index=seen_stats[stat.id],
            )
        seen_stats[stat.id] = idx
        statistics.append(stat)

    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        fail(E_SPEC, "$.metadata", "metadata must be an object")

    return Spec(
        sources=tuple(sorted(sources, key=lambda source: source.id)),
        rules=tuple(sorted(rules, key=lambda rule: rule.id)),
        statistics=tuple(sorted(statistics, key=lambda stat: stat.id)),
        packing=_validate_packing(obj["packing"], "$.packing"),
        accounting=_validate_accounting(obj["accounting"], "$.accounting"),
        metadata=dict(metadata),
        format_version=version,
        base_dir=base_dir,
    )


def load_spec(path: str | Path) -> Spec:
    spec_path = Path(path)
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(E_SPEC, "$", "cannot read specification file", error=str(exc), file=str(spec_path))
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except BusinessError:
        raise
    except json.JSONDecodeError as exc:
        fail(
            E_SPEC,
            "$",
            "invalid JSON",
            line=exc.lineno,
            column=exc.colno,
            reason=exc.msg,
        )
    return validate_spec(raw, base_dir=str(spec_path.resolve().parent))


def dump_spec(spec: Spec) -> dict[str, Any]:
    return spec.to_dict()
