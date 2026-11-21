"""Strong, all-or-nothing loading of dataset and pipeline JSON files."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .canonical import (
    canonical_json,
    dataset_digest,
    pipeline_digest,
    spec_fingerprint,
    stage_config_digest,
)
from .errors import E_SPEC, WorkbenchError
from .models import (
    ColumnData,
    DatasetSpec,
    PipelineSpec,
    PreparedSpec,
    ShardingSpec,
    SourceData,
    StageSpec,
)

DATASET_FORMAT = "mlpipe.dataset.v1"
PIPELINE_FORMAT = "mlpipe.pipeline.v1"
STAGE_TYPES = (
    "canonicalize",
    "fill_missing",
    "normalize",
    "clip_outliers",
    "shard_pack",
)
STAGE_RANK = {name: index for index, name in enumerate(STAGE_TYPES)}
STAGE_PARAM_KEYS = {
    "canonicalize": ("saturate",),
    "fill_missing": ("fill_fixed",),
    "normalize": ("column",),
    "clip_outliers": ("column", "z_limit_permille", "max_outlier_permille"),
    "shard_pack": ("remainder_policy",),
}
REMAINDER_TO_INT = {"drop_partial": 0, "keep_partial": 1}
INT_TO_REMAINDER = {value: key for key, value in REMAINDER_TO_INT.items()}
ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
SOURCE_ID_RE = ID_RE
COLUMN_NAME_RE = ID_RE
DECIMAL_RE = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)


def _fail(
    message: str,
    location: str,
    *,
    stage_id: str | None = None,
    param: str | None = None,
    param_value: Any = None,
    affected_rows: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    raise WorkbenchError(
        code=E_SPEC,
        message=message,
        stage_id=stage_id,
        param=param,
        param_value=param_value,
        affected_rows=affected_rows,
        location=location,
        details=details or {},
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_json_file(path: str | Path, label: str) -> Any:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkbenchError(
            E_SPEC,
            f"cannot read {label} file: {exc}",
            location=f"{label}.file",
        ) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkbenchError(
            E_SPEC,
            f"invalid JSON: {exc.msg}",
            location=f"{label}:line {exc.lineno}, column {exc.colno}",
        ) from exc


def decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _decimal_from_json(value: Any, location: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        _fail("float sample must be a finite number or decimal string", location)
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, int):
        parsed = Decimal(value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            _fail("float sample must be finite", location)
        parsed = Decimal(str(value))
    elif isinstance(value, str):
        if len(value) > 128 or not DECIMAL_RE.match(value.strip()):
            _fail("float sample string is not a canonical decimal", location)
        try:
            parsed = Decimal(value.strip())
        except InvalidOperation:
            _fail("float sample string is not a decimal", location)
    else:
        _fail("sample value has unsupported JSON type", location)
    if not parsed.is_finite():
        _fail("float sample must be finite", location)
    if abs(parsed.adjusted()) > 18:
        _fail("float sample magnitude exceeds the supported range", location)
    return parsed


def _validate_dataset(obj: Any) -> DatasetSpec:
    if not isinstance(obj, dict):
        _fail("dataset root must be an object", "data")
    if obj.get("format") != DATASET_FORMAT:
        _fail(
            f"dataset format must be {DATASET_FORMAT!r}",
            "data.format",
        )
    sources_obj = obj.get("sources")
    if not isinstance(sources_obj, list) or not sources_obj:
        _fail("dataset.sources must be a non-empty array", "data.sources")

    source_ids: set[str] = set()
    global_keys: set[str] = set()
    sources: list[SourceData] = []
    global_columns: list[ColumnData] = []
    cursor = 0

    for source_index, source_obj in enumerate(sources_obj):
        base = f"data.sources[{source_index}]"
        if not isinstance(source_obj, dict):
            _fail("source must be an object", base)
        source_id = source_obj.get("id")
        if not isinstance(source_id, str) or not SOURCE_ID_RE.match(source_id):
            _fail("source.id must match ^[A-Za-z][A-Za-z0-9_.-]{0,63}$", f"{base}.id")
        if source_id in source_ids:
            _fail(f"duplicate source id {source_id!r}", f"{base}.id")
        source_ids.add(source_id)

        rows = source_obj.get("rows")
        if not _is_int(rows) or rows <= 0 or rows > 1_000_000:
            _fail("source.rows must be an integer in [1, 1000000]", f"{base}.rows")
        columns_obj = source_obj.get("columns")
        if not isinstance(columns_obj, list) or not columns_obj:
            _fail("source.columns must be a non-empty array", f"{base}.columns")

        source_names: set[str] = set()
        source_columns: list[ColumnData] = []
        for column_index, column_obj in enumerate(columns_obj):
            cbase = f"{base}.columns[{column_index}]"
            if not isinstance(column_obj, dict):
                _fail("column must be an object", cbase)
            name = column_obj.get("name")
            if not isinstance(name, str) or not COLUMN_NAME_RE.match(name):
                _fail(
                    "column.name must match ^[A-Za-z][A-Za-z0-9_.-]{0,63}$",
                    f"{cbase}.name",
                )
            if name in source_names:
                _fail(f"duplicate column name {name!r} in source", f"{cbase}.name")
            source_names.add(name)
            key = f"{source_id}.{name}"
            if key in global_keys:
                _fail(f"duplicate globally qualified column {key!r}", f"{cbase}.name")
            global_keys.add(key)

            kind = column_obj.get("type")
            if kind not in ("int", "float"):
                _fail("column.type must be 'int' or 'float'", f"{cbase}.type")
            values_obj = column_obj.get("values")
            nulls_obj = column_obj.get("nulls")
            if not isinstance(values_obj, list):
                _fail("column.values must be an array", f"{cbase}.values")
            if len(values_obj) != rows:
                _fail(
                    f"column.values length {len(values_obj)} does not match rows {rows}",
                    f"{cbase}.values",
                    affected_rows=rows,
                )
            if not isinstance(nulls_obj, list):
                _fail("column.nulls must be an array", f"{cbase}.nulls")
            if len(nulls_obj) != rows:
                _fail(
                    f"null bitmap length {len(nulls_obj)} does not match rows {rows}",
                    f"{cbase}.nulls",
                    affected_rows=rows,
                )

            canonical_values: list[int | str | None] = []
            runtime_values: list[int | Decimal | None] = []
            runtime_nulls: list[bool] = []
            for row_index, (raw_value, null_flag) in enumerate(zip(values_obj, nulls_obj)):
                vbase = f"{cbase}.values[{row_index}]"
                if not _is_int(null_flag) or null_flag not in (0, 1):
                    _fail("null bitmap entries must be 0 or 1", f"{cbase}.nulls[{row_index}]")
                is_null = bool(null_flag)
                runtime_nulls.append(is_null)
                if raw_value is None:
                    if not is_null:
                        _fail("non-null sample value must not be null", vbase)
                    runtime_values.append(None)
                    canonical_values.append(None)
                    continue
                if kind == "int":
                    if not _is_int(raw_value):
                        _fail("integer column sample must be an integer", vbase)
                    if not -(2**63) <= raw_value <= 2**63 - 1:
                        _fail("integer sample is outside signed 64-bit range", vbase)
                    parsed_value: int | Decimal = raw_value
                    canonical_values.append(raw_value)
                else:
                    parsed_decimal = _decimal_from_json(raw_value, vbase)
                    parsed_value = parsed_decimal
                    canonical_values.append(decimal_text(parsed_decimal))
                runtime_values.append(None if is_null else parsed_value)

            column = ColumnData(
                source_id=source_id,
                name=name,
                kind=kind,
                values=runtime_values,
                nulls=runtime_nulls,
            )
            # Attach the canonical serialisation without changing the runtime model.
            setattr(column, "_canonical_values", canonical_values)
            source_columns.append(column)
            global_columns.append(column)

        sources.append(
            SourceData(
                id=source_id,
                rows=rows,
                columns=source_columns,
                start_row=cursor,
            )
        )
        cursor += rows

    canonical_sources: list[dict[str, Any]] = []
    for source in sources:
        canonical_columns = []
        for column in source.columns:
            canonical_columns.append(
                {
                    "name": column.name,
                    "type": column.kind,
                    "values": list(getattr(column, "_canonical_values")),
                    "nulls": [1 if flag else 0 for flag in column.nulls],
                }
            )
        canonical_sources.append(
            {"id": source.id, "rows": source.rows, "columns": canonical_columns}
        )
    canonical = {"format": DATASET_FORMAT, "sources": canonical_sources}
    digest = dataset_digest(canonical)
    seed = None
    generator = obj.get("generator")
    if isinstance(generator, dict) and _is_int(generator.get("seed")):
        seed = generator["seed"]
    return DatasetSpec(
        sources=tuple(sources),
        columns=tuple(global_columns),
        row_count=cursor,
        canonical=canonical,
        digest=digest,
        seed=seed,
    )


def _expect_params_dict(stage_obj: dict[str, Any], base: str) -> dict[str, int]:
    stage_type = stage_obj["type"]
    params_obj = stage_obj.get("params")
    if not isinstance(params_obj, dict):
        _fail("stage.params must be an object", f"{base}.params")
    expected = STAGE_PARAM_KEYS[stage_type]
    expected_set = set(expected)
    for key in expected:
        if key not in params_obj:
            _fail(f"missing integer parameter {key!r}", f"{base}.params.{key}")
        value = params_obj[key]
        if not _is_int(value):
            _fail("stage parameter must be an integer", f"{base}.params.{key}")
    extra = sorted(set(params_obj) - expected_set)
    if extra:
        _fail(f"unsupported stage parameter {extra[0]!r}", f"{base}.params.{extra[0]}")
    return {key: int(params_obj[key]) for key in expected}


def _validate_stage_params(
    stage: StageSpec,
    column_count: int,
    sharding: ShardingSpec,
) -> None:
    base = f"pipeline.stages[id={stage.id}]"
    params = stage.params
    if stage.type == "canonicalize":
        if params["saturate"] not in (0, 1):
            _fail("saturate must be 0 or 1", f"{base}.params.saturate", stage_id=stage.id)
    elif stage.type == "fill_missing":
        value = params["fill_fixed"]
        if not -(10**9) <= value <= 10**9:
            _fail(
                "fill_fixed must be in [-1000000000, 1000000000]",
                f"{base}.params.fill_fixed",
                stage_id=stage.id,
                param="fill_fixed",
                param_value=value,
            )
    elif stage.type == "normalize":
        column = params["column"]
        if not 0 <= column < column_count:
            _fail(
                f"column index {column} is outside [0, {column_count - 1}]",
                f"{base}.params.column",
                stage_id=stage.id,
                param="column",
                param_value=column,
            )
    elif stage.type == "clip_outliers":
        column = params["column"]
        if not 0 <= column < column_count:
            _fail(
                f"column index {column} is outside [0, {column_count - 1}]",
                f"{base}.params.column",
                stage_id=stage.id,
                param="column",
                param_value=column,
            )
        z_limit = params["z_limit_permille"]
        if not 1 <= z_limit <= 100_000:
            _fail(
                "z_limit_permille must be in [1, 100000]",
                f"{base}.params.z_limit_permille",
                stage_id=stage.id,
                param="z_limit_permille",
                param_value=z_limit,
            )
        max_outlier = params["max_outlier_permille"]
        if not 0 <= max_outlier <= 1000:
            _fail(
                "max_outlier_permille must be in [0, 1000]",
                f"{base}.params.max_outlier_permille",
                stage_id=stage.id,
                param="max_outlier_permille",
                param_value=max_outlier,
            )
    elif stage.type == "shard_pack":
        policy = params["remainder_policy"]
        if policy not in INT_TO_REMAINDER:
            _fail(
                "remainder_policy must be 0 (drop_partial) or 1 (keep_partial)",
                f"{base}.params.remainder_policy",
                stage_id=stage.id,
                param="remainder_policy",
                param_value=policy,
            )
        if INT_TO_REMAINDER[policy] != sharding.remainder_policy:
            _fail(
                "shard_pack remainder_policy must match pipeline.sharding.remainder_policy",
                f"{base}.params.remainder_policy",
                stage_id=stage.id,
                param="remainder_policy",
                param_value=policy,
            )


def _compile_pipeline(obj: Any, column_count: int) -> PipelineSpec:
    if not isinstance(obj, dict):
        _fail("pipeline root must be an object", "pipeline")
    if obj.get("format") != PIPELINE_FORMAT:
        _fail(
            f"pipeline format must be {PIPELINE_FORMAT!r}",
            "pipeline.format",
        )
    fixed_scale = obj.get("fixed_scale")
    if not _is_int(fixed_scale) or not 0 <= fixed_scale <= 6:
        _fail(
            "fixed_scale must be an integer in [0, 6]",
            "pipeline.fixed_scale",
            param="fixed_scale",
            param_value=fixed_scale,
        )

    sharding_obj = obj.get("sharding")
    if not isinstance(sharding_obj, dict):
        _fail("pipeline.sharding must be an object", "pipeline.sharding")
    target_rows = sharding_obj.get("target_rows")
    if not _is_int(target_rows) or not 1 <= target_rows <= 1_000_000:
        _fail(
            "target_rows must be an integer in [1, 1000000]",
            "pipeline.sharding.target_rows",
            param="target_rows",
            param_value=target_rows,
        )
    remainder_policy = sharding_obj.get("remainder_policy")
    if remainder_policy not in REMAINDER_TO_INT:
        _fail(
            "remainder_policy must be 'keep_partial' or 'drop_partial'",
            "pipeline.sharding.remainder_policy",
            param="remainder_policy",
            param_value=remainder_policy,
        )
    sharding = ShardingSpec(target_rows=target_rows, remainder_policy=remainder_policy)

    stages_obj = obj.get("stages")
    if not isinstance(stages_obj, list) or not stages_obj:
        _fail("pipeline.stages must be a non-empty array", "pipeline.stages")

    raw_stages: dict[str, StageSpec] = {}
    file_order: list[str] = []
    for index, stage_obj in enumerate(stages_obj):
        base = f"pipeline.stages[{index}]"
        if not isinstance(stage_obj, dict):
            _fail("stage must be an object", base)
        stage_id = stage_obj.get("id")
        if not isinstance(stage_id, str) or not ID_RE.match(stage_id):
            _fail("stage.id must match ^[A-Za-z][A-Za-z0-9_.-]{0,63}$", f"{base}.id")
        if stage_id in raw_stages:
            _fail(f"duplicate stage id {stage_id!r}", f"{base}.id", stage_id=stage_id)
        stage_type = stage_obj.get("type")
        if stage_type not in STAGE_TYPES:
            _fail(
                f"stage.type must be one of {', '.join(STAGE_TYPES)}",
                f"{base}.type",
                stage_id=stage_id,
            )
        version = stage_obj.get("version")
        if not _is_int(version) or not 1 <= version <= 1_000_000_000:
            _fail(
                "stage.version must be an integer in [1, 1000000000]",
                f"{base}.version",
                stage_id=stage_id,
                param="version",
                param_value=version,
            )
        upstream = stage_obj.get("upstream")
        if upstream is not None and not isinstance(upstream, str):
            _fail("stage.upstream must be a stage id or null", f"{base}.upstream", stage_id=stage_id)
        params = _expect_params_dict(stage_obj, base)
        stage = StageSpec(
            id=stage_id,
            type=stage_type,
            version=version,
            upstream=upstream,
            params=params,
        )
        raw_stages[stage_id] = stage
        file_order.append(stage_id)

    for stage_id in file_order:
        stage = raw_stages[stage_id]
        if stage.upstream is not None and stage.upstream not in raw_stages:
            _fail(
                f"upstream stage {stage.upstream!r} does not exist",
                f"pipeline.stages[id={stage.id}].upstream",
                stage_id=stage.id,
                param="upstream",
                param_value=stage.upstream,
            )

    # Deterministic cycle detection. Stage ids, not JSON object order, select
    # the first traversal path and therefore the first reported violation.
    state: dict[str, int] = {stage_id: 0 for stage_id in raw_stages}

    def visit(stage_id: str) -> None:
        state[stage_id] = 1
        upstream = raw_stages[stage_id].upstream
        if upstream is not None:
            if state[upstream] == 1:
                _fail(
                    "stage graph contains a cycle",
                    f"pipeline.stages[id={stage_id}].upstream",
                    stage_id=stage_id,
                    param="upstream",
                    param_value=upstream,
                )
            if state[upstream] == 0:
                visit(upstream)
        state[stage_id] = 2

    for stage_id in sorted(raw_stages):
        if state[stage_id] == 0:
            visit(stage_id)

    roots = sorted(stage.id for stage in raw_stages.values() if stage.upstream is None)
    if len(roots) != 1:
        _fail(
            "pipeline must be one connected chain with exactly one root",
            "pipeline.stages",
            details={"roots": roots},
        )
    children: dict[str, list[str]] = {stage_id: [] for stage_id in raw_stages}
    for stage in raw_stages.values():
        if stage.upstream is not None:
            children[stage.upstream].append(stage.id)
    for parent in sorted(children):
        children[parent].sort()
        if len(children[parent]) > 1:
            _fail(
                "pipeline branches are not supported; each stage has at most one successor",
                f"pipeline.stages[id={parent}]",
                stage_id=parent,
                details={"successors": children[parent]},
            )

    ordered_ids: list[str] = []
    cursor: str | None = roots[0]
    while cursor is not None:
        ordered_ids.append(cursor)
        cursor = children[cursor][0] if children[cursor] else None
    if len(ordered_ids) != len(raw_stages):
        missing = sorted(set(raw_stages) - set(ordered_ids))
        _fail(
            "pipeline stages are disconnected from the root chain",
            "pipeline.stages",
            details={"unreachable_stages": missing},
        )

    ordered: list[StageSpec] = []
    previous_rank = -1
    for ordinal, stage_id in enumerate(ordered_ids):
        stage = raw_stages[stage_id]
        rank = STAGE_RANK[stage.type]
        if rank < previous_rank:
            _fail(
                "stage types must follow canonicalize -> fill_missing -> normalize -> clip_outliers -> shard_pack",
                f"pipeline.stages[id={stage.id}].type",
                stage_id=stage.id,
                param="type",
                param_value=stage.type,
            )
        previous_rank = rank
        _validate_stage_params(stage, column_count, sharding)
        config_digest = stage_config_digest(stage.config_object())
        ordered.append(stage.with_ordinal(ordinal, config_digest))

    if ordered[0].type != "canonicalize":
        _fail("the chain root must be a canonicalize stage", f"pipeline.stages[id={ordered[0].id}]")
    if ordered[-1].type != "shard_pack":
        _fail("the chain tail must be a shard_pack stage", f"pipeline.stages[id={ordered[-1].id}]")
    if sum(stage.type == "canonicalize" for stage in ordered) != 1:
        _fail("the chain must contain exactly one canonicalize stage", "pipeline.stages")
    if sum(stage.type == "shard_pack" for stage in ordered) != 1:
        _fail("the chain must contain exactly one shard_pack stage", "pipeline.stages")
    for required_type in ("fill_missing", "normalize", "clip_outliers"):
        if not any(stage.type == required_type for stage in ordered):
            _fail(f"the chain must contain at least one {required_type} stage", "pipeline.stages")

    canonical = {
        "format": PIPELINE_FORMAT,
        "fixed_scale": fixed_scale,
        "sharding": sharding.as_dict(),
        "stages": [stage.config_object() for stage in ordered],
    }
    digest = pipeline_digest(canonical)
    return PipelineSpec(
        fixed_scale=fixed_scale,
        sharding=sharding,
        stages=tuple(ordered),
        canonical=canonical,
        digest=digest,
    )


def prepare_from_objects(dataset_obj: Any, pipeline_obj: Any) -> PreparedSpec:
    dataset = _validate_dataset(dataset_obj)
    pipeline = _compile_pipeline(pipeline_obj, len(dataset.columns))
    fingerprint = spec_fingerprint(dataset.digest, pipeline.digest)
    return PreparedSpec(dataset=dataset, pipeline=pipeline, fingerprint=fingerprint)


def load_workspace(data_path: str | Path, pipeline_path: str | Path) -> PreparedSpec:
    dataset_obj = _parse_json_file(data_path, "data")
    pipeline_obj = _parse_json_file(pipeline_path, "pipeline")
    # Every check above and below completes before the caller can allocate a
    # spec version or a run, so an invalid workspace is rejected wholesale.
    return prepare_from_objects(dataset_obj, pipeline_obj)


def snapshot_json(value: Any) -> str:
    return canonical_json(value)
