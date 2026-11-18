"""Pipeline specification loading, strong validation and fingerprinting."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from .canonical import canonical_bytes, combined_hash, domained_hash
from .errors import WorkbenchError
from .fixedpoint import ROUNDING_MODES

SUPPORTED_COLUMN_TYPES = ("int", "float")
STAGE_TYPES = ("read", "normalize_types", "fill_missing",
               "normalize", "clip", "shard")
REMAINDER_RULES = ("keep_partial", "drop_partial")

# Inclusive integer bounds for every integer parameter, keyed by stage type.
PARAM_BOUNDS: dict[str, dict[str, tuple[int, int]]] = {
    "read": {"chunk_rows": (1, 100_000)},
    # fixed_scale is the absolute integer scaling factor (e.g. 1000), so a
    # raw integer cell n becomes n * fixed_scale in fixed-point form.
    "normalize_types": {"fixed_scale": (1, 1_000_000_000)},
    "fill_missing": {"fill_value": (-10_000_000_000, 10_000_000_000)},
    "normalize": {"scale": (0, 9), "divisor": (1, 1_000_000_000)},
    "clip": {"lower": (-10_000_000_000, 10_000_000_000),
             "upper": (-10_000_000_000, 10_000_000_000),
             "max_outlier_bp": (0, 10_000)},
    "shard": {"target_rows": (1, 1_000_000)},
}
STRING_PARAMS = {
    "normalize_types": ("rounding",),
    "normalize": ("rounding",),
    "shard": ("remainder_rule",),
}


@dataclass(frozen=True)
class SourceSpec:
    id: str
    path: str
    columns: dict[str, str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    content_digest: str = ""


@dataclass(frozen=True)
class StageSpec:
    id: str
    type: str
    version: int
    upstream: tuple[str, ...]
    params: dict[str, Any]
    ordinal: int = 0


@dataclass(frozen=True)
class PipelineSpec:
    schema_version: int
    sources: tuple[SourceSpec, ...]
    stages: tuple[StageSpec, ...]
    raw: dict[str, Any]
    samples_digest: str
    fingerprint: str
    stage_fingerprints: dict[str, str]


def _spec_error(message: str, stage_id: Optional[str] = None,
                field_path: str = "$", details: Optional[dict] = None
                ) -> WorkbenchError:
    return WorkbenchError("E_SPEC", message, stage_id, field_path, details)


def _param_error(message: str, stage_id: str, field_path: str,
                 details: dict) -> WorkbenchError:
    # Out-of-bounds integer params are runtime parameter faults (E_PARAM),
    # distinct from structural specification faults (E_SPEC).
    return WorkbenchError("E_PARAM", message, stage_id, field_path, details)


def _read_json_file(path: str, stage_id: Optional[str],
                    field_path: str, corrupt_code: str = "E_SPEC"
                    ) -> Any:
    if not os.path.isfile(path):
        raise WorkbenchError(corrupt_code, f"file not found: {path}",
                             stage_id, field_path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # A broken *sample* file is an E_DATA runtime failure, while a broken
        # pipeline document remains an E_SPEC loading failure.
        code = "E_DATA" if corrupt_code == "E_DATA" else "E_SPEC"
        raise WorkbenchError(code, f"invalid JSON: {exc}", stage_id,
                             field_path, {"path": path})
    except OSError as exc:
        raise WorkbenchError(corrupt_code, f"unreadable file: {exc}",
                             stage_id, field_path, {"path": path})


def _validate_cell(column: str, declared: str, value: Any,
                   source_id: str, row_index: int) -> None:
    path = f"$.sources[{source_id}].rows[{row_index}].{column}"
    if value is None:
        return
    if declared == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise _spec_error(
                f"column {column!r} declared int but got {type(value).__name__}",
                field_path=path)
        return
    if declared == "float":
        if isinstance(value, bool):
            raise _spec_error(f"boolean cell in float column {column!r}",
                              field_path=path)
        if isinstance(value, int):
            return
        if isinstance(value, float):
            try:
                decimal_value = Decimal(str(value))
            except Exception:
                raise _spec_error(f"invalid float in column {column!r}",
                                  field_path=path)
            if not decimal_value.is_finite():
                raise _spec_error(f"non-finite float in column {column!r}",
                                  field_path=path)
            return
        raise _spec_error(
            f"column {column!r} declared float but got {type(value).__name__}",
            field_path=path)


def _load_source(raw_source: dict[str, Any], base_dir: str,
                 index: int) -> SourceSpec:
    path_prefix = f"$.sources[{index}]"
    if not isinstance(raw_source, dict):
        raise _spec_error("source entry must be an object",
                          field_path=path_prefix)
    source_id = raw_source.get("id")
    if not isinstance(source_id, str) or not source_id:
        raise _spec_error("source id must be a non-empty string",
                          field_path=path_prefix + ".id")
    columns_raw = raw_source.get("columns")
    if not isinstance(columns_raw, dict) or not columns_raw:
        raise _spec_error("source columns must be a non-empty object",
                          source_id, path_prefix + ".columns")
    columns: dict[str, str] = {}
    for column, meta in columns_raw.items():
        if not isinstance(meta, dict) or "type" not in meta:
            raise _spec_error("column definition requires type", source_id,
                              f"{path_prefix}.columns.{column}")
        col_type = meta["type"]
        if col_type not in SUPPORTED_COLUMN_TYPES:
            raise _spec_error(
                f"unsupported column type {col_type!r}; allowed: "
                f"{', '.join(SUPPORTED_COLUMN_TYPES)}", source_id,
                f"{path_prefix}.columns.{column}.type")
        columns[column] = col_type
    rel_path = raw_source.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        raise _spec_error("source path must be a non-empty string", source_id,
                          path_prefix + ".path")
    sample_path = rel_path if os.path.isabs(rel_path) else os.path.normpath(
        os.path.join(base_dir, rel_path))
    document = _read_json_file(sample_path, source_id,
                               path_prefix + ".path", corrupt_code="E_DATA")
    if not isinstance(document, dict) or not isinstance(document.get("rows"),
                                                        list):
        raise WorkbenchError("E_DATA",
                             "sample document must contain a rows array",
                             source_id, path_prefix + ".rows",
                             {"path": sample_path})
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(document["rows"]):
        if not isinstance(row, dict):
            raise WorkbenchError("E_DATA", "each row must be an object",
                                 source_id,
                                 f"{path_prefix}.rows[{row_index}]",
                                 {"path": sample_path})
        unknown = set(row) - set(columns)
        if unknown:
            raise _spec_error(
                f"undeclared columns in row: {sorted(unknown)}", source_id,
                f"{path_prefix}.rows[{row_index}]")
        clean_row: dict[str, Any] = {}
        for column, col_type in columns.items():
            value = row.get(column)
            _validate_cell(column, col_type, value, source_id, row_index)
            clean_row[column] = value
        rows.append(clean_row)
    content_digest = _sample_content_digest(columns, rows)
    return SourceSpec(source_id, sample_path, dict(sorted(columns.items())),
                      rows, len(rows), content_digest)


def _sample_content_digest(columns: dict[str, str],
                           rows: list[dict[str, Any]]) -> str:
    """Digest over declared columns and exact cell contents (order-sensitive)."""
    payload = {
        "columns": columns,
        "rows": [{col: _decimal_text(row.get(col)) for col in columns}
                 for row in rows],
    }
    return domained_hash("sample-content-v1", payload)


def _decimal_text(value: Any) -> Any:
    if isinstance(value, float):
        # Canonical textual form removes host float formatting differences.
        return format(Decimal(str(value)), "f")
    return value


def _validate_stage(raw_stage: dict[str, Any], index: int,
                    source_ids: set[str]) -> StageSpec:
    prefix = f"$.stages[{index}]"
    if not isinstance(raw_stage, dict):
        raise _spec_error("stage entry must be an object", field_path=prefix)
    stage_id = raw_stage.get("id")
    if not isinstance(stage_id, str) or not stage_id:
        raise _spec_error("stage id must be a non-empty string",
                          field_path=prefix + ".id")
    stage_type = raw_stage.get("type")
    if stage_type not in STAGE_TYPES:
        raise _spec_error(
            f"unsupported stage type {stage_type!r}; allowed: "
            f"{', '.join(STAGE_TYPES)}", stage_id, prefix + ".type")
    version = raw_stage.get("version")
    if not isinstance(version, int) or isinstance(version, bool) \
            or not 1 <= version <= 1_000_000:
        raise _spec_error("stage version must be an integer within [1, 1e6]",
                          stage_id, prefix + ".version")
    upstream_raw = raw_stage.get("upstream")
    if not isinstance(upstream_raw, list) or not all(
            isinstance(item, str) for item in upstream_raw):
        raise _spec_error("upstream must be a list of stage id strings",
                          stage_id, prefix + ".upstream")
    params_raw = raw_stage.get("params")
    if not isinstance(params_raw, dict):
        raise _spec_error("params must be an object", stage_id,
                          prefix + ".params")
    params = _validate_params(stage_type, params_raw, stage_id, prefix,
                              source_ids)
    return StageSpec(stage_id, stage_type, version, tuple(upstream_raw),
                     dict(sorted(params.items())), ordinal=index)


def _validate_params(stage_type: str, params_raw: dict[str, Any],
                     stage_id: str, prefix: str,
                     source_ids: set[str]) -> dict[str, Any]:
    bounds = PARAM_BOUNDS[stage_type]
    string_names = STRING_PARAMS.get(stage_type, ())
    allowed = set(bounds) | set(string_names)
    if stage_type == "read":
        allowed.add("source_ids")
    unknown = set(params_raw) - allowed
    if unknown:
        raise _spec_error(f"unknown params {sorted(unknown)}", stage_id,
                          prefix + ".params")
    params: dict[str, Any] = {}
    for name, (low, high) in bounds.items():
        if name not in params_raw:
            raise _spec_error(f"missing integer param {name!r}", stage_id,
                              f"{prefix}.params.{name}")
        value = params_raw[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise _spec_error(f"param {name!r} must be an integer", stage_id,
                              f"{prefix}.params.{name}")
        if not low <= value <= high:
            raise _param_error(
                f"param {name!r}={value} outside inclusive bounds [{low},{high}]",
                stage_id, f"{prefix}.params.{name}",
                {"param": name, "value": value, "min": low, "max": high})
        params[name] = value
    for name in string_names:
        if name not in params_raw:
            raise _spec_error(f"missing string param {name!r}", stage_id,
                              f"{prefix}.params.{name}")
        value = params_raw[name]
        if not isinstance(value, str):
            raise _spec_error(f"param {name!r} must be a string", stage_id,
                              f"{prefix}.params.{name}")
        if name == "rounding" and value not in ROUNDING_MODES:
            raise _spec_error(f"unknown rounding mode {value!r}", stage_id,
                              f"{prefix}.params.{name}")
        if name == "remainder_rule" and value not in REMAINDER_RULES:
            raise _spec_error(f"unknown remainder rule {value!r}", stage_id,
                              f"{prefix}.params.{name}")
        params[name] = value
    if stage_type == "read":
        source_list = params_raw.get("source_ids")
        if not isinstance(source_list, list) or not source_list or \
                not all(isinstance(item, str) for item in source_list):
            raise _spec_error("source_ids must be a non-empty string list",
                              stage_id, f"{prefix}.params.source_ids")
        missing = sorted(set(source_list) - source_ids)
        if missing:
            raise _spec_error(f"unknown source ids {missing}", stage_id,
                              f"{prefix}.params.source_ids")
        if len(set(source_list)) != len(source_list):
            raise _spec_error("source_ids must not contain duplicates",
                              stage_id, f"{prefix}.params.source_ids")
        params["source_ids"] = list(source_list)
    if stage_type == "clip" and params["lower"] > params["upper"]:
        raise _spec_error("clip lower bound must not exceed upper bound",
                          stage_id, f"{prefix}.params.lower")
    return params


def _validate_graph(stages: tuple[StageSpec, ...]) -> None:
    ids = [stage.id for stage in stages]
    if len(set(ids)) != len(ids):
        duplicates = sorted({stage_id for stage_id in ids
                             if ids.count(stage_id) > 1})
        raise _spec_error(f"duplicate stage ids: {duplicates}",
                          field_path="$.stages[*].id")
    id_set = set(ids)
    adjacency: dict[str, list[str]] = {}
    for ordinal, stage in enumerate(stages):
        prefix = f"$.stages[{ordinal}].upstream"
        unknown = sorted(set(stage.upstream) - id_set)
        if unknown:
            raise _spec_error(f"upstream references unknown stages {unknown}",
                              stage.id, prefix)
        if stage.type == "read":
            if stage.upstream:
                raise _spec_error("read stage must have empty upstream",
                                  stage.id, prefix)
        elif not stage.upstream:
            raise _spec_error("non-read stage requires at least one upstream",
                              stage.id, prefix)
        adjacency[stage.id] = list(stage.upstream)
    # Deterministic iterative DFS cycle detection (white/gray/black coloring).
    color = {stage_id: "white" for stage_id in id_set}
    for root in ids:
        if color[root] != "white":
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            node, edge = stack[-1]
            if edge == 0:
                color[node] = "gray"
            neighbors = adjacency[node]
            if edge < len(neighbors):
                nxt = neighbors[edge]
                stack[-1] = (node, edge + 1)
                if color[nxt] == "gray":
                    raise _spec_error(
                        f"cycle detected through stage {nxt!r}", node,
                        "$.stages[*].upstream")
                if color[nxt] == "white":
                    stack.append((nxt, 0))
            else:
                color[node] = "black"
                stack.pop()


def _stage_fingerprint(stage: StageSpec) -> str:
    return combined_hash("stage-fingerprint-v1", (
        ("id", stage.id),
        ("type", stage.type),
        ("version", stage.version),
        ("upstream", list(stage.upstream)),
        ("params", stage.params),
    ))


def _pipeline_fingerprint(sources: tuple[SourceSpec, ...],
                          stages: tuple[StageSpec, ...],
                          stage_fingerprints: dict[str, str]) -> tuple[
                                  str, str]:
    sample_parts = [(src.id, {"columns": src.columns,
                              "digest": src.content_digest,
                              "row_count": src.row_count})
                    for src in sorted(sources, key=lambda item: item.id)]
    samples_digest = combined_hash("samples-digest-v1", sample_parts)
    stage_parts = [(f"{stage.ordinal:04d}:{stage.id}",
                    stage_fingerprints[stage.id])
                   for stage in stages]
    fingerprint = combined_hash("pipeline-fingerprint-v1",
                                [("samples", samples_digest)] + stage_parts)
    return fingerprint, samples_digest


def load_pipeline(spec_path: str) -> PipelineSpec:
    """Load and strongly validate a pipeline document and its sample files."""
    spec_path = os.path.abspath(spec_path)
    base_dir = os.path.dirname(spec_path)
    document = _read_json_file(spec_path, None, "$", corrupt_code="E_SPEC")
    if not isinstance(document, dict):
        raise _spec_error("pipeline document must be a JSON object")
    schema_version = document.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise _spec_error("schema_version must be an integer",
                          field_path="$.schema_version")
    raw_sources = document.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise _spec_error("sources must be a non-empty array",
                          field_path="$.sources")
    sources = tuple(_load_source(raw, base_dir, index)
                    for index, raw in enumerate(raw_sources))
    source_ids = {src.id for src in sources}
    raw_stages = document.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise _spec_error("stages must be a non-empty array",
                          field_path="$.stages")
    stages = tuple(_validate_stage(raw, index, source_ids)
                   for index, raw in enumerate(raw_stages))
    _validate_graph(stages)
    if stages[0].type != "read" or any(
            stage.type == "read" for stage in stages[1:]):
        raise _spec_error("the chain must contain exactly one read stage first",
                          field_path="$.stages[*].type")
    if not any(stage.type == "shard" for stage in stages):
        raise _spec_error("the chain must contain a shard stage",
                          field_path="$.stages[*].type")
    stage_fingerprints = {stage.id: _stage_fingerprint(stage)
                          for stage in stages}
    fingerprint, samples_digest = _pipeline_fingerprint(
        sources, stages, stage_fingerprints)
    return PipelineSpec(
        schema_version=schema_version,
        sources=sources,
        stages=stages,
        raw=document,
        samples_digest=samples_digest,
        fingerprint=fingerprint,
        stage_fingerprints=stage_fingerprints,
    )


def canonical_spec_snapshot(spec: PipelineSpec) -> bytes:
    """Stable on-disk snapshot of the validated pipeline document."""
    return canonical_bytes(spec.raw)
