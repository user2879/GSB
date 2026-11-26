"""Sample JSONL validation, in-memory table, and edit command application."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from .errors import E_INPUT_INVALID
from .fixedpoint import FixedParams, as_decimal
from .hashing import canonical_sha256, short_key


@dataclass
class SampleRecord:
    id: str
    seed: int
    features: list[Decimal]
    target: Decimal
    weight: Decimal
    line: int | None = None

    def content_material(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seed": self.seed,
            "features": [format(v, "f") for v in self.features],
            "target": format(self.target, "f"),
            "weight": format(self.weight, "f"),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.content_material())


@dataclass
class Dataset:
    columns: list[str]
    rows: dict[str, SampleRecord] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return sorted(self.rows)

    def require_aligned(self) -> None:
        width = len(self.columns)
        for sample_id in self.ids:
            if len(self.rows[sample_id].features) != width:
                raise ValueError(f"sample {sample_id} has width {len(self.rows[sample_id].features)} != {width}")

    def clone(self) -> "Dataset":
        return deepcopy(self)

    def column_fingerprint(self, index: int) -> str:
        name = self.columns[index]
        material = {
            "name": name,
            "values": [[sample_id, format(self.rows[sample_id].features[index], "f")] for sample_id in self.ids],
        }
        return canonical_sha256(material)

    def column_fingerprints(self) -> list[str]:
        return [self.column_fingerprint(i) for i in range(len(self.columns))]

    def fingerprint_material(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "column_fingerprints": self.column_fingerprints(),
            "samples": [[sample_id, self.rows[sample_id].fingerprint] for sample_id in self.ids],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.fingerprint_material())


def _number_list(value: Any, field_name: str, width: int | None, location: str) -> tuple[list[Decimal], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return [], [{"code": E_INPUT_INVALID, "field": f"{location}.{field_name}", "message": "must be an array"}]
    if width is not None and len(value) != width:
        errors.append({
            "code": E_INPUT_INVALID,
            "field": f"{location}.{field_name}",
            "message": f"feature dimension {len(value)} != declared column count {width}",
        })
    result: list[Decimal] = []
    for index, item in enumerate(value):
        try:
            if isinstance(item, bool):
                raise ValueError("boolean is not a number")
            result.append(as_decimal(item, f"{field_name}[{index}]"))
        except ValueError as exc:
            errors.append({
                "code": E_INPUT_INVALID,
                "field": f"{location}.{field_name}[{index}]",
                "message": str(exc),
            })
    return result, errors


def validate_sample_object(
    obj: Any,
    *,
    location: str,
    width: int | None,
    seen_ids: set[str],
) -> tuple[SampleRecord | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(obj, dict):
        return None, [{"code": E_INPUT_INVALID, "field": location, "message": "sample must be a JSON object"}]

    sample_id = obj.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        errors.append({"code": E_INPUT_INVALID, "field": f"{location}.id", "message": "id must be a non-empty string"})
        sample_id = None
    elif sample_id in seen_ids:
        errors.append({"code": E_INPUT_INVALID, "field": f"{location}.id", "message": f"duplicate sample id: {sample_id}"})

    seed = obj.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append({"code": E_INPUT_INVALID, "field": f"{location}.seed", "message": "seed must be an integer"})
        seed = -1

    features, feature_errors = _number_list(obj.get("features"), "features", width, location)
    errors.extend(feature_errors)

    target = None
    try:
        target = as_decimal(obj.get("target"), "target")
    except ValueError as exc:
        errors.append({"code": E_INPUT_INVALID, "field": f"{location}.target", "message": str(exc)})

    weight = None
    try:
        weight = as_decimal(obj.get("weight"), "weight")
    except ValueError as exc:
        errors.append({"code": E_INPUT_INVALID, "field": f"{location}.weight", "message": str(exc)})

    if errors:
        return None, errors
    assert sample_id is not None and target is not None and weight is not None
    seen_ids.add(sample_id)
    return SampleRecord(sample_id, seed, features, target, weight), []


def load_samples_jsonl(path: str | Path, columns: list[str]) -> tuple[list[SampleRecord], list[dict[str, Any]]]:
    path = Path(path)
    records: list[SampleRecord] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    width = len(columns)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], [{"code": E_INPUT_INVALID, "field": "samples", "message": f"sample file not found: {path}"}]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        location = f"samples.line_{line_number}"
        try:
            obj = json.loads(line, parse_float=Decimal, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        except json.JSONDecodeError as exc:
            errors.append({"code": E_INPUT_INVALID, "field": f"{location}.json", "message": f"invalid JSON: {exc.msg}"})
            continue
        except ValueError as exc:
            errors.append({"code": E_INPUT_INVALID, "field": f"{location}.json", "message": f"non-finite number: {exc}"})
            continue
        record, record_errors = validate_sample_object(obj, location=location, width=width, seen_ids=seen)
        errors.extend(record_errors)
        if record is not None:
            record.line = line_number
            records.append(record)
    return records, errors


def dataset_from_records(columns: list[str], records: Iterable[SampleRecord]) -> Dataset:
    dataset = Dataset(columns=list(columns))
    for record in records:
        dataset.rows[record.id] = record
    dataset.require_aligned()
    return dataset


def load_command_stream(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    errors: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    try:
        parsed = json.loads(text, parse_float=Decimal, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if isinstance(parsed, list):
            commands = parsed
        else:
            errors.append({"code": E_INPUT_INVALID, "field": "edits", "message": "command file must be JSONL or a JSON array"})
    except json.JSONDecodeError:
        for index, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                commands.append(json.loads(line, parse_float=Decimal, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))))
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append({"code": E_INPUT_INVALID, "field": f"edits.command_{index}.json", "message": str(exc)})
    if not isinstance(commands, list):
        commands = []
    for index, command in enumerate(commands):
        if not isinstance(command, dict) or not isinstance(command.get("op"), str):
            errors.append({"code": E_INPUT_INVALID, "field": f"edits[{index}].op", "message": "command must contain string op"})
    return commands, errors


def _finite_number(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite number")
    return as_decimal(value, field)


def apply_commands(
    dataset: Dataset,
    config: dict[str, Any],
    commands: list[dict[str, Any]],
) -> tuple[Dataset, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply commands in order. Returns new dataset, mutable config, results, errors."""

    dataset = dataset.clone()
    config = deepcopy(config)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, command in enumerate(commands):
        loc = f"edits[{index}]"
        if not isinstance(command, dict) or not isinstance(command.get("op"), str):
            errors.append({"code": E_INPUT_INVALID, "field": f"{loc}.op", "message": "invalid command"})
            continue
        op = command["op"]
        try:
            if op == "append_sample":
                sample = command.get("sample")
                seen = set(dataset.ids)
                record, validation_errors = validate_sample_object(
                    sample, location=loc, width=len(dataset.columns), seen_ids=seen
                )
                if validation_errors:
                    errors.extend(validation_errors)
                    continue
                assert record is not None
                dataset.rows[record.id] = record
                results.append({"index": index, "op": op, "status": "applied", "id": record.id})

            elif op == "replace_sample":
                sample_id = command.get("id")
                if not isinstance(sample_id, str) or sample_id not in dataset.rows:
                    raise ValueError(f"{loc}.id sample does not exist: {sample_id}")
                sample = command.get("sample")
                seen = {sid for sid in dataset.ids if sid != sample_id}
                record, validation_errors = validate_sample_object(
                    sample, location=loc, width=len(dataset.columns), seen_ids=seen
                )
                if validation_errors:
                    errors.extend(validation_errors)
                    continue
                assert record is not None
                if record.id != sample_id:
                    raise ValueError(f"{loc}.sample.id must match command id")
                record.line = dataset.rows[sample_id].line
                dataset.rows[sample_id] = record
                results.append({"index": index, "op": op, "status": "applied", "id": sample_id})

            elif op == "delete_sample":
                sample_id = command.get("id")
                if not isinstance(sample_id, str) or sample_id not in dataset.rows:
                    raise ValueError(f"{loc}.id sample does not exist: {sample_id}")
                del dataset.rows[sample_id]
                results.append({"index": index, "op": op, "status": "applied", "id": sample_id})

            elif op == "add_column":
                name = command.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{loc}.name must be non-empty string")
                if name in dataset.columns:
                    raise ValueError(f"{loc}.name column already exists: {name}")
                values_by_id: dict[str, Decimal] = {}
                if "value" in command:
                    scalar = _finite_number(command["value"], f"{loc}.value")
                    for sample_id in dataset.ids:
                        values_by_id[sample_id] = scalar
                else:
                    raw_values = command.get("values")
                    if not isinstance(raw_values, dict):
                        raise ValueError(f"{loc}.values must be an object or provide value")
                    raw_map = {str(k): v for k, v in raw_values.items()}
                    if set(raw_map) != set(dataset.ids):
                        missing = sorted(set(dataset.ids) - set(raw_map))
                        unknown = sorted(set(raw_map) - set(dataset.ids))
                        raise ValueError(f"{loc}.values id mismatch missing={missing} unknown={unknown}")
                    for sample_id, raw_value in raw_map.items():
                        values_by_id[sample_id] = _finite_number(raw_value, f"{loc}.values.{sample_id}")
                dataset.columns.append(name)
                for sample_id in dataset.ids:
                    dataset.rows[sample_id].features.append(values_by_id[sample_id])
                results.append({"index": index, "op": op, "status": "applied", "column": name})

            elif op == "delete_column":
                name = command.get("name")
                if name not in dataset.columns:
                    raise ValueError(f"{loc}.name column does not exist: {name}")
                if len(dataset.columns) <= 1:
                    raise ValueError("at least one feature column is required")
                col_index = dataset.columns.index(name)
                dataset.columns.pop(col_index)
                for sample_id in dataset.ids:
                    dataset.rows[sample_id].features.pop(col_index)
                results.append({"index": index, "op": op, "status": "applied", "column": name})

            elif op == "update_column":
                name = command.get("name")
                if not isinstance(name, str) or name not in dataset.columns:
                    raise ValueError(f"{loc}.name column does not exist: {name}")
                col_index = dataset.columns.index(name)
                updates: dict[str, Decimal] = {}
                if "value" in command:
                    scalar = _finite_number(command["value"], f"{loc}.value")
                    updates = {sample_id: scalar for sample_id in dataset.ids}
                else:
                    raw_values = command.get("values")
                    if not isinstance(raw_values, dict) or not raw_values:
                        raise ValueError(f"{loc}.values must be a non-empty object or provide value")
                    for raw_id, raw_value in raw_values.items():
                        sample_id = str(raw_id)
                        if sample_id not in dataset.rows:
                            raise ValueError(f"{loc}.values unknown sample id: {sample_id}")
                        updates[sample_id] = _finite_number(raw_value, f"{loc}.values.{sample_id}")
                for sample_id, value in updates.items():
                    dataset.rows[sample_id].features[col_index] = value
                results.append({
                    "index": index,
                    "op": op,
                    "status": "applied",
                    "column": name,
                    "updated_samples": sorted(updates),
                })

            elif op == "set_alphas":
                target_name = command.get("path", command.get("target"))
                if target_name is None and len(config["paths"]) == 1:
                    target_name = next(iter(config["paths"]))
                if target_name not in config["paths"]:
                    raise ValueError(f"{loc}.path unknown path: {target_name}")
                raw_alphas = command.get("alphas")
                if not isinstance(raw_alphas, list) or not raw_alphas:
                    raise ValueError(f"{loc}.alphas must be non-empty array")
                alphas: list[Decimal] = []
                for a_index, raw_alpha in enumerate(raw_alphas):
                    alpha = _finite_number(raw_alpha, f"{loc}.alphas[{a_index}]")
                    if alpha < 0:
                        raise ValueError("alphas must be non-negative")
                    alphas.append(alpha)
                for a_index in range(1, len(alphas)):
                    if alphas[a_index] <= alphas[a_index - 1]:
                        raise ValueError("alphas must be strictly increasing")
                path = config["paths"][target_name]
                path["alphas"] = alphas
                results.append({"index": index, "op": op, "status": "applied", "path": target_name, "count": len(alphas)})

            elif op in {"set_fixed_point", "set_quantization"}:
                raw_patch = command.get("fixed_point", command)
                current = config["fixed"]
                decimal_places = int(raw_patch.get("decimal_places", current.decimal_places))
                quant_level = int(raw_patch.get("quant_level", current.quant_level))
                margin = int(raw_patch.get("accum_rounding_margin", current.accum_rounding_margin))
                config["fixed"] = FixedParams(decimal_places, quant_level, margin)
                results.append({"index": index, "op": op, "status": "applied", "fixed": config["fixed"].fingerprint_material()})

            else:
                raise ValueError(f"unknown op: {op}")
        except (ValueError, TypeError) as exc:
            errors.append({"code": E_INPUT_INVALID, "field": loc, "message": str(exc)})

    dataset.require_aligned()
    config["columns"] = list(dataset.columns)
    config["config_fingerprint"] = canonical_sha256({
        "columns": config["columns"],
        "fixed": config["fixed"].fingerprint_material(),
        "paths": {
            name: {"penalty": p["penalty"], "alphas": [str(a) for a in p["alphas"]]}
            for name, p in config["paths"].items()
        },
    })
    return dataset, config, results, errors


def dataset_cache_material(dataset: Dataset, fixed: FixedParams) -> dict[str, Any]:
    return {
        "dataset": dataset.fingerprint_material(),
        "fixed": fixed.fingerprint_material(),
    }


def dataset_cache_key(dataset: Dataset, fixed: FixedParams, stage: str) -> str:
    return short_key(stage, dataset_cache_material(dataset, fixed))
