"""Deterministic sample generation, integrity checks, and column replacement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import mini_yaml
from .config import PipelineConfig, failure, normalize_failures
from .util import (
    atomic_write_json,
    fingerprint,
    read_json,
    sha256_bytes,
)

SAMPLE_SCHEMA_VERSION = 1
MISSING_CATEGORY = "__MISSING__"


def _derive_seed(base_seed: int, *parts: str) -> int:
    material = "|".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(bytes.fromhex(sha256_bytes(material))[:8], "big") % (2**63 - 1)


def _scalar_key(value: Any) -> str:
    return f"{type(value).__name__}:{json.dumps(value, ensure_ascii=False, allow_nan=False)}"


def _unit_effect(column: str, value: Any) -> float:
    raw = sha256_bytes(f"{column}|{_scalar_key(value)}".encode("utf-8"))
    return int.from_bytes(bytes.fromhex(raw)[:8], "big") / float(2**64) - 0.5


def _generate_category(
    descriptor: dict[str, Any], rows: int, seed: int
) -> tuple[list[Any], list[int], dict[str, Any]]:
    rng = np.random.default_rng(_derive_seed(seed, "category", descriptor["name"]))
    choices = list(descriptor["values"])
    generated = [choices[int(index)] for index in rng.integers(0, len(choices), size=rows)]
    missing = rng.random(rows) < float(descriptor["missing_rate"])
    filled_count = int(missing.sum())
    filled = [MISSING_CATEGORY if is_missing else value for value, is_missing in zip(generated, missing)]
    map_entries = []
    for code, value in enumerate(sorted({_scalar_key(item): item for item in filled}.items(), key=lambda pair: pair[0])):
        map_entries.append({"value": value[1], "code": code})
    key_to_code = {_scalar_key(entry["value"]): entry["code"] for entry in map_entries}
    encoded = [key_to_code[_scalar_key(value)] for value in filled]
    return filled, encoded, {
        "fill_value": MISSING_CATEGORY,
        "filled_count": filled_count,
        "category_map": map_entries,
    }


def _generate_numeric(
    descriptor: dict[str, Any], rows: int, seed: int
) -> tuple[list[float], list[int], dict[str, Any]]:
    rng = np.random.default_rng(_derive_seed(seed, "numeric", descriptor["name"]))
    minimum = float(descriptor["min"])
    maximum = float(descriptor["max"])
    generated = rng.uniform(minimum, maximum, size=rows)
    missing = rng.random(rows) < float(descriptor["missing_rate"])
    filled_count = int(missing.sum())
    observed = generated[~missing]
    fill_value = float(observed.mean()) if observed.size else (minimum + maximum) / 2.0
    filled = [fill_value if is_missing else float(value) for value, is_missing in zip(generated, missing)]
    return filled, [0 for _ in filled], {
        "fill_value": fill_value,
        "filled_count": filled_count,
        "category_map": [],
    }


def _build_target(
    values: dict[str, list[Any]], descriptors: list[dict[str, Any]], rows: int, seed: int, target_name: str
) -> list[int]:
    signal = np.zeros(rows, dtype=np.float64)
    for descriptor in descriptors:
        name = descriptor["name"]
        column = values[name]
        if descriptor["type"] == "category":
            signal += np.array(
                [_unit_effect(name, value) * (0.0 if value == MISSING_CATEGORY else 0.9) for value in column],
                dtype=np.float64,
            )
        else:
            minimum = float(descriptor["min"])
            maximum = float(descriptor["max"])
            width = maximum - minimum or 1.0
            signal += np.array([((float(value) - minimum) / width - 0.5) * 1.1 for value in column])
    rng = np.random.default_rng(_derive_seed(seed, "target", target_name))
    signal += rng.normal(0.0, 0.12, size=rows)
    threshold = float(np.median(signal))
    return [int(value > threshold) for value in signal]


def generate_sample(config: PipelineConfig, *, target_column: str = "target") -> dict[str, Any]:
    if config.rows <= 0:
        raise ValueError("cannot generate a sample with zero rows")
    if not config.descriptors:
        raise ValueError("cannot generate a sample without valid column descriptors")
    values: dict[str, list[Any]] = {}
    encoded: dict[str, list[int]] = {}
    columns: dict[str, dict[str, Any]] = {}
    fill_report: dict[str, dict[str, Any]] = {}
    category_maps: dict[str, list[dict[str, Any]]] = {}
    for descriptor in config.descriptors:
        name = descriptor["name"]
        if descriptor["type"] == "category":
            column, codes, details = _generate_category(descriptor, config.rows, config.seed)
        else:
            column, codes, details = _generate_numeric(descriptor, config.rows, config.seed)
        values[name] = column
        encoded[name] = codes
        columns[name] = {key: value for key, value in descriptor.items() if key != "values"}
        if descriptor["type"] == "category":
            columns[name]["values"] = list(descriptor["values"])
        fill_report[name] = {
            "fill_value": details["fill_value"],
            "filled_count": details["filled_count"],
        }
        category_maps[name] = details["category_map"]
    target = _build_target(values, config.descriptors, config.rows, config.seed, target_column)
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "generation_config_hash": config.generation_config_hash,
        "seed": config.seed,
        "rows": config.rows,
        "column_order": [descriptor["name"] for descriptor in config.descriptors],
        "columns": columns,
        "values": values,
        "encoded": encoded,
        "category_maps": category_maps,
        "fill_report": fill_report,
        "target_column": target_column,
        "target": target,
    }


def sample_checksum(sample: dict[str, Any]) -> str:
    return fingerprint(sample)


def prepare_sample(config: PipelineConfig, *, force: bool = False) -> dict[str, Any]:
    """Generate once, then reuse the persisted sample and its offline digest."""

    manifest_path = config.paths["sample_manifest"]
    sample_path = config.paths["sample"]
    if not force and manifest_path.exists() and sample_path.exists():
        try:
            manifest = read_json(manifest_path)
            stored = read_json(sample_path)
            if (
                manifest.get("generation_config_hash") == config.generation_config_hash
                and manifest.get("checksum") == sample_checksum(stored)
            ):
                return {**manifest, "prepared": False, "reused": True}
        except Exception:
            pass
    sample = generate_sample(config, target_column=str(config.raw["metrics"]["target_column"]))
    checksum = sample_checksum(sample)
    manifest = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "generation_config_hash": config.generation_config_hash,
        "checksum": checksum,
        "checksum_algorithm": "sha256-canonical-json",
        "seed": config.seed,
        "rows": config.rows,
        "column_order": sample["column_order"],
        "target_column": sample["target_column"],
        "generated_by": "feature-pipeline prepare",
    }
    atomic_write_json(sample_path, sample)
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "prepared": True, "reused": False}


@dataclass
class LoadedSample:
    values: dict[str, list[Any]]
    encoded: dict[str, list[int]]
    target: list[int]
    target_column: str
    column_fingerprints: dict[str, str]
    base_checksum: str
    effective_checksum: str
    override_provenance: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    rows: int


def _read_override_file(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return mini_yaml.load(path)


def _validate_override_values(
    descriptor: dict[str, Any], values: Any, source: str
) -> tuple[list[Any] | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    name = descriptor["name"]
    if not isinstance(values, list):
        failures.append(failure("E_DATA", source, "replacement values must be a list"))
        return None, failures
    if descriptor["type"] == "numeric":
        for index, value in enumerate(values):
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
                failures.append(
                    failure("E_TYPE_MISMATCH", f"{name}[{index}]", "numeric replacement contains a non-numeric value")
                )
        return values, normalize_failures(failures)
    expected_types = {type(value) for value in descriptor.get("values", [])}
    for index, value in enumerate(values):
        if value is None:
            continue
        if type(value) not in expected_types or isinstance(value, (dict, list)):
            failures.append(
                failure("E_TYPE_MISMATCH", f"{name}[{index}]", "category replacement value type does not match column")
            )
    return values, normalize_failures(failures)


def _fill_and_encode_column(
    descriptor: dict[str, Any], source_values: list[Any], override_values: list[Any] | None, rows: int
) -> tuple[list[Any], list[int], dict[str, Any]]:
    supplied = list(override_values) if override_values is not None else list(source_values)
    if len(supplied) != rows:
        raise ValueError(f"replacement length {len(supplied)} does not match sample rows {rows}")
    if descriptor["type"] == "category":
        observed = [value for value in supplied if value is not None]
        all_values = list(descriptor.get("values", [])) + observed
        missing = [value is None for value in supplied]
        filled = [MISSING_CATEGORY if is_missing else value for value, is_missing in zip(supplied, missing)]
        category_values = sorted({_scalar_key(value): value for value in all_values + filled}.items())
        map_entries = [{"value": value, "code": code} for code, (_, value) in enumerate(category_values)]
        key_to_code = {_scalar_key(entry["value"]): entry["code"] for entry in map_entries}
        encoded = [key_to_code[_scalar_key(value)] for value in filled]
        return filled, encoded, {
            "fill_value": MISSING_CATEGORY,
            "filled_count": int(sum(missing)),
            "category_map": map_entries,
        }
    observed = [float(value) for value in supplied if value is not None]
    fill_value = float(np.mean(observed)) if observed else (float(descriptor["min"]) + float(descriptor["max"])) / 2.0
    missing = [value is None for value in supplied]
    filled = [fill_value if is_missing else float(value) for value, is_missing in zip(supplied, missing)]
    return filled, [0 for _ in filled], {
        "fill_value": fill_value,
        "filled_count": int(sum(missing)),
        "category_map": [],
    }


def load_sample(config: PipelineConfig) -> LoadedSample:
    failures: list[dict[str, Any]] = []
    manifest_path = config.paths["sample_manifest"]
    sample_path = config.paths["sample"]
    try:
        manifest = read_json(manifest_path)
        sample = read_json(sample_path)
    except Exception as exc:
        failures.append(failure("E_SAMPLE", "sample", f"cannot load sample or manifest: {exc}"))
        return LoadedSample({}, {}, [], "", {}, "", "", [], normalize_failures(failures), 0)
    base_checksum = str(manifest.get("checksum", ""))
    if base_checksum != sample_checksum(sample):
        failures.append(failure("E_SAMPLE", "sample", "offline sample checksum does not match the persisted data"))
    if manifest.get("generation_config_hash") != config.generation_config_hash:
        failures.append(
            failure("E_SAMPLE", "sample", "sample was generated from a different column specification; rerun prepare")
        )
    if sample.get("rows") != config.rows:
        failures.append(failure("E_SAMPLE", "sample", "sample row count differs from configuration"))
    values = {name: list(column) for name, column in sample.get("values", {}).items()}
    target_column = str(config.raw["metrics"]["target_column"])
    target: list[int] = []
    if target_column == sample.get("target_column", "target"):
        target = [int(value) for value in sample.get("target", [])]
    elif target_column in values:
        raw_target = values[target_column]
        numeric = np.array([float(value) for value in raw_target], dtype=np.float64)
        threshold = float(np.median(numeric))
        target = [int(value > threshold) for value in numeric]
    else:
        failures.append(failure("E_TARGET", target_column, "target column is not present in the persisted sample"))

    column_fingerprints: dict[str, str] = {}
    encoded: dict[str, list[int]] = {}
    override_provenance: list[dict[str, Any]] = []
    overrides_dir = config.paths["overrides"]
    override_files: list[Path] = []
    if overrides_dir.exists():
        override_files = sorted(
            [path for path in overrides_dir.iterdir() if path.suffix.lower() in {".json", ".yaml", ".yml"}],
            key=lambda path: path.name,
        )
    overridden: dict[str, list[Any]] = {}
    for path in override_files:
        try:
            payload = _read_override_file(path)
            if isinstance(payload, dict):
                name = payload.get("column", path.stem)
                replacement = payload.get("values")
            else:
                name = path.stem
                replacement = payload
        except Exception as exc:
            failures.append(failure("E_DATA", path.name, f"cannot parse replacement: {exc}"))
            continue
        if not isinstance(name, str) or name not in config.column_by_name:
            failures.append(failure("E_UNKNOWN_COLUMN", str(name), f"unknown replacement column in {path.name}"))
            continue
        descriptor = config.column_by_name[name]
        replacement, replacement_failures = _validate_override_values(descriptor, replacement, path.name)
        failures.extend(replacement_failures)
        if replacement is None or replacement_failures:
            continue
        if len(replacement) != config.rows:
            failures.append(
                failure(
                    "E_DATA",
                    name,
                    f"replacement length {len(replacement)} does not match sample rows {config.rows}",
                )
            )
            continue
        overridden[name] = replacement
        override_provenance.append(
            {
                "column": name,
                "file": path.name,
                "sha256": sha256_bytes(path.read_bytes()),
            }
        )

    for descriptor in config.descriptors:
        name = descriptor["name"]
        if name not in values:
            failures.append(failure("E_SAMPLE", name, "sample does not contain a configured column"))
            continue
        replacement = overridden.get(name)
        if replacement is None:
            encoded[name] = [int(code) for code in sample.get("encoded", {}).get(name, [])]
            if not encoded[name] or len(encoded[name]) != len(values[name]):
                encoded[name] = [0 for _ in values[name]]
        else:
            try:
                values[name], encoded[name], _ = _fill_and_encode_column(
                    descriptor, values[name], replacement, config.rows
                )
            except Exception as exc:
                failures.append(failure("E_DATA", name, str(exc)))
        column_fingerprints[name] = fingerprint(values[name])

    effective_checksum = fingerprint(
        {
            "base_checksum": base_checksum,
            "values": values,
            "encoded": encoded,
            "target": target,
            "overrides": override_provenance,
        }
    )
    return LoadedSample(
        values=values,
        encoded=encoded,
        target=target,
        target_column=target_column,
        column_fingerprints=column_fingerprints,
        base_checksum=base_checksum,
        effective_checksum=effective_checksum,
        override_provenance=override_provenance,
        failures=normalize_failures(failures),
        rows=len(target),
    )


def read_replacement_file(path: str | Path) -> list[Any]:
    payload = _read_override_file(Path(path))
    if isinstance(payload, dict):
        if "values" not in payload:
            raise ValueError("replacement mapping must contain a values list")
        payload = payload["values"]
    if not isinstance(payload, list):
        raise ValueError("replacement file must contain a list of values")
    return payload


def write_column_override(
    config: PipelineConfig, column: str, values: list[Any]
) -> dict[str, Any]:
    descriptor = config.column_by_name.get(column)
    if descriptor is None:
        raise KeyError(f"unknown column: {column}")
    validated, failures = _validate_override_values(descriptor, values, f"replacement:{column}")
    if failures:
        raise ValueError("; ".join(item["message"] for item in failures))
    if validated is None or len(validated) != config.rows:
        raise ValueError(f"replacement must contain exactly {config.rows} values")
    path = config.paths["overrides"] / f"{column}.json"
    payload = {
        "schema_version": 1,
        "column": column,
        "values": validated,
    }
    atomic_write_json(path, payload)
    return {
        "path": str(path.relative_to(config.root)) if path.is_relative_to(config.root) else str(path),
        "sha256": sha256_bytes(path.read_bytes()),
        "column": column,
        "rows": len(validated),
    }
