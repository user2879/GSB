"""Deterministic edit operations for samples, configuration and level maps."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .audio import load_audio
from .config import DIMENSIONS, FILE_ID_PATTERN, Config
from .errors import FormatError, InterruptedRun
from .levels import LevelMap
from .pipeline import EvaluationPipeline
from .util import atomic_write_json


CONFIG_MUTATING_OPS = {
    "set_weight",
    "set_threshold",
    "set_quantization_levels",
    "set_fixed_bits",
    "set_concurrency",
    "set_snapshot_interval",
    "set_verify_serial",
}
LEVEL_MUTATING_OPS = {"add_level", "remove_level", "set_level_threshold"}


def _read_state(root: Path) -> dict[str, Any]:
    path = root / "state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {
            "schema": "aqeval.state.v1",
            "active_version": 2,
            "config_revision": 0,
            "global_run": 0,
            "last_edits": [],
        }


def _write_config(root: Path, config: Config) -> None:
    raw = {"schema": "aqeval.config.v1", **config.to_dict()}
    atomic_write_json(root / "config.json", raw)


def apply_edit(root: str | Path, operation: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    op = str(operation.get("op", ""))
    if not op:
        raise FormatError("edit operation requires an op field")
    state = _read_state(root)
    config = Config.load(root / "config.json")
    level_map = LevelMap.load(root / "level_map.json")
    changed_config = False
    changed_levels = False
    changed_version = False
    changed_audio: list[str] = []
    if op == "replace_audio":
        file_id = str(operation.get("file_id", ""))
        if not FILE_ID_PATTERN.fullmatch(file_id):
            raise FormatError(f"invalid file_id for audio replacement: {file_id!r}")
        source = Path(str(operation.get("source", "")))
        if not source.is_absolute():
            source = root / source
        load_audio(source, file_id, config)
        destination = root / "samples" / f"{file_id}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        changed_audio.append(file_id)
    elif op == "set_weight":
        dimension = str(operation.get("dimension", ""))
        if dimension not in DIMENSIONS:
            raise FormatError(f"unknown weight dimension: {dimension}")
        weights = dict(config.weights)
        weights[dimension] = float(operation["value"])
        config = config.with_updates(weights=weights)
        changed_config = True
    elif op == "set_threshold":
        key = str(operation.get("key", ""))
        allowed = {
            "silence_threshold_dbfs",
            "noise_floor_percentile",
            "clipping_amplitude_threshold",
            "consecutive_zero_samples",
            "frame_length_ms",
            "hop_ms",
        }
        if key not in allowed:
            raise FormatError(f"unsupported threshold key: {key}")
        config = config.with_updates(**{key: operation["value"]})
        changed_config = True
    elif op == "set_quantization_levels":
        config = config.with_updates(quantization_levels=list(operation["value"]))
        changed_config = True
    elif op == "set_fixed_bits":
        config = config.with_updates(fixed_decimal_bits=int(operation["value"]))
        changed_config = True
    elif op == "set_concurrency":
        config = config.with_updates(max_concurrency=int(operation["value"]))
        changed_config = True
    elif op == "set_snapshot_interval":
        config = config.with_updates(snapshot_commit_interval=int(operation["value"]))
        changed_config = True
    elif op == "set_verify_serial":
        config = config.with_updates(verify_serial_consistency=bool(operation["value"]))
        changed_config = True
    elif op == "add_level":
        level_map = level_map.add(
            str(operation["code"]),
            str(operation["name"]),
            int(operation["min_total_permille"]),
        )
        changed_levels = True
    elif op == "remove_level":
        level_map = level_map.remove(
            code=str(operation["code"]) if "code" in operation else None,
            name=str(operation["name"]) if "name" in operation else None,
        )
        changed_levels = True
    elif op == "set_level_threshold":
        level_map = level_map.set_threshold(
            code=str(operation["code"]) if "code" in operation else None,
            name=str(operation["name"]) if "name" in operation else None,
            value=int(operation["value"]),
        )
        changed_levels = True
    elif op == "switch_version":
        version = int(operation.get("value", operation.get("version", 0)))
        if version not in (1, 2, 3):
            raise FormatError("switch_version value must be 1, 2 or 3")
        state["active_version"] = version
        changed_version = True
    else:
        raise FormatError(f"unsupported edit operation: {op}")
    if changed_config:
        _write_config(root, config)
    if changed_levels:
        atomic_write_json(root / "level_map.json", level_map.to_dict())
    if changed_config or changed_levels or changed_version:
        state["config_revision"] = int(state.get("config_revision", 0)) + 1
    edits = list(state.get("last_edits", []))
    edits.append(operation)
    state["last_edits"] = edits[-200:]
    state["schema"] = "aqeval.state.v1"
    atomic_write_json(root / "state.json", state)
    return {
        "op": op,
        "changed_config": changed_config,
        "changed_levels": changed_levels,
        "changed_version": changed_version,
        "changed_audio": changed_audio,
        "config_revision": state["config_revision"],
        "active_version": state.get("active_version", 2),
    }


def apply_edit_script(
    root: str | Path,
    script_path: str | Path,
    *,
    run_after_each: bool = True,
    run_final: bool = True,
    rebuild: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    operations: list[dict[str, Any]] = []
    with open(script_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise FormatError(f"invalid edit JSON at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise FormatError(f"edit at line {line_number} must be an object")
            operations.append(value)
    applied: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    first_run = True
    for operation in operations:
        applied.append(apply_edit(root, operation))
        if run_after_each:
            pipeline = EvaluationPipeline(
                root,
                rebuild_cache=rebuild and first_run,
                rebuild_snapshot=rebuild and first_run,
            )
            try:
                report = pipeline.run()
                runs.append(
                    {
                        "after_op": operation,
                        "summary": report["summary"],
                        "output_sha256": report["outputs"]["sha256"],
                        "stale": report["snapshot"]["stale"],
                        "reuse_count": report["cache_ledger"]["reuse_count"],
                        "recompute_count": report["cache_ledger"]["recompute_count"],
                    }
                )
            except InterruptedRun:
                raise
            first_run = False
    if run_final and not run_after_each:
        pipeline = EvaluationPipeline(
            root,
            rebuild_cache=rebuild and first_run,
            rebuild_snapshot=rebuild and first_run,
        )
        report = pipeline.run()
        runs.append(
            {
                "after_op": None,
                "summary": report["summary"],
                "output_sha256": report["outputs"]["sha256"],
                "stale": report["snapshot"]["stale"],
                "reuse_count": report["cache_ledger"]["reuse_count"],
                "recompute_count": report["cache_ledger"]["recompute_count"],
            }
        )
    return {"applied": applied, "runs": runs, "operation_count": len(operations)}

