"""Apply the JSON edit-command stream to local sample/config files."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from .util import atomic_write_json


class EditError(ValueError):
    pass


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise EditError(f"target page must be a relative path without '..': {value}")
    return path


def _set_nested(target: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = target
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def apply_edit_commands(config_path: Path, commands_path: Path) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    commands_path = Path(commands_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commands = json.loads(commands_path.read_text(encoding="utf-8"))
    if not isinstance(commands, dict) or commands.get("schema_version") != 1:
        raise EditError("command stream schema_version must be 1")
    command_list = commands.get("commands")
    if not isinstance(command_list, list):
        raise EditError("command stream must contain a commands array")
    sample_dir = (config_path.parent / config["sample_dir"]).resolve()
    sample_dir.mkdir(parents=True, exist_ok=True)

    applied: List[Dict[str, Any]] = []
    for index, command in enumerate(command_list):
        if not isinstance(command, dict):
            raise EditError(f"command {index} must be an object")
        operation = command.get("op")
        if operation in {"add_page", "replace_page"}:
            page = _safe_relative(str(command.get("page", "")))
            source_png = (commands_path.parent / str(command.get("source_png", ""))).resolve()
            source_truth = (commands_path.parent / str(command.get("source_truth", ""))).resolve()
            if not source_png.is_file() or not source_truth.is_file():
                raise EditError(f"command {index}: source PNG or truth file is missing")
            target_png = sample_dir / page
            target_truth = target_png.with_suffix(".json")
            _atomic_copy(source_png, target_png)
            _atomic_copy(source_truth, target_truth)
            applied.append(
                {
                    "index": index,
                    "op": operation,
                    "page": page.as_posix(),
                    "source_png": str(source_png),
                    "target_png": str(target_png),
                }
            )
        elif operation == "remove_page":
            page = _safe_relative(str(command.get("page", "")))
            target_png = sample_dir / page
            target_truth = target_png.with_suffix(".json")
            removed = []
            for path in (target_png, target_truth):
                if path.exists():
                    path.unlink()
                    removed.append(str(path))
            applied.append({"index": index, "op": operation, "page": page.as_posix(), "removed": removed})
        elif operation == "set_config":
            dotted = str(command.get("path", ""))
            if not dotted:
                raise EditError(f"command {index}: set_config.path is required")
            _set_nested(config, dotted, command.get("value"))
            applied.append({"index": index, "op": operation, "path": dotted, "value": command.get("value")})
        else:
            raise EditError(f"command {index}: unsupported op {operation!r}")

    atomic_write_json(config_path, config)
    return {
        "schema_version": 1,
        "config_path": str(config_path),
        "commands_path": str(commands_path),
        "sample_dir": str(sample_dir),
        "applied": applied,
    }
