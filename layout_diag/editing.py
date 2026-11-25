"""Replayable edit-command application for local dataset workspaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import FailureLog
from .utils import atomic_write_json, copy_atomic, read_json, stable_hash, write_json


def load_edit_commands(path: str | Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if isinstance(value, dict):
        value = value.get("commands", [])
    if not isinstance(value, list):
        raise ValueError("edit stream must be an array or an object with a commands array")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError("every edit command must be an object")
    return list(value)


def apply_edit_stream(
    dataset_root: str | Path,
    config_path: str | Path,
    commands: list[dict[str, Any]],
    *,
    base_dir: str | Path | None = None,
) -> FailureLog:
    failures = FailureLog()
    dataset_root = Path(dataset_root).resolve()
    config_path = Path(config_path).resolve()
    base_dir = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    manifest_path = dataset_root / "manifest.json"
    try:
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("manifest root is not an object")
    except Exception as exc:
        failures.add("E_EDIT_COMMAND", "edit", str(manifest_path), f"cannot load manifest: {exc}")
        return failures
    try:
        config = read_json(config_path)
        if not isinstance(config, dict):
            raise ValueError("config root is not an object")
    except Exception as exc:
        failures.add("E_EDIT_COMMAND", "edit", str(config_path), f"cannot load config: {exc}")
        return failures

    images = manifest.get("images")
    if not isinstance(images, list):
        failures.add("E_EDIT_COMMAND", "edit", "manifest.images", "manifest images must be an array")
        return failures

    applied: list[dict[str, Any]] = []
    for index, command in enumerate(commands, start=1):
        operation = command.get("op")
        before = len(failures)
        try:
            if operation in {"replace_image", "add_image"}:
                _upsert_image(dataset_root, manifest, command, failures, require_existing=(operation == "replace_image"), base_dir=base_dir)
            elif operation == "remove_image":
                _remove_image(dataset_root, manifest, command, failures, base_dir=base_dir)
            elif operation == "replace_keypoints":
                _replace_keypoints(dataset_root, manifest, command, failures, base_dir=base_dir)
            elif operation == "set_window":
                _set_window(config, command, failures)
            elif operation == "set_step":
                _set_step(config, command, failures)
            elif operation == "set_tier":
                _set_tier(config, command, failures)
            elif operation == "set_merge_threshold":
                _set_merge_threshold(config, command, failures)
            elif operation == "set_quantization":
                _set_quantization(config, command, failures)
            elif operation == "set_cc":
                _set_cc(config, command, failures)
            elif operation == "set_point_budget":
                _set_nonnegative_int(config, command, "point_budget", failures, allow_zero=True)
            elif operation == "set_max_workers":
                _set_nonnegative_int(config, command, "max_workers", failures, allow_zero=False)
            elif operation == "set_keypoint_version":
                version = command.get("version")
                if not isinstance(version, str) or not version:
                    failures.add("E_EDIT_COMMAND", "edit", f"commands[{index}]", "set_keypoint_version needs a version")
                else:
                    config["keypoint_version"] = version
            else:
                failures.add(
                    "E_EDIT_COMMAND",
                    "edit",
                    f"commands[{index}]",
                    "unknown edit operation",
                    operation=operation,
                )
        except Exception as exc:
            failures.add(
                "E_EDIT_COMMAND",
                "edit",
                f"commands[{index}]",
                f"operation failed: {exc}",
                operation=operation,
            )
        if len(failures) == before:
            applied.append({"index": index, "op": operation})

    manifest["last_edit"] = {
        "command_count": len(commands),
        "applied_count": len(applied),
        "content_hint": stable_hash(commands, length=16),
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(config_path, config)
    write_json(
        dataset_root / "edit_history.json",
        {
            "schema_version": 1,
            "commands": commands,
            "applied": applied,
            "failure_codes": [failure.code for failure in failures.items],
        },
    )
    return failures


def _entry_by_name(manifest: dict[str, Any], name: str) -> dict[str, Any] | None:
    for entry in manifest.get("images", []):
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    return None


def _resolve_source(command: dict[str, Any], keys: tuple[str, ...], base_dir: Path) -> Path | None:
    for key in keys:
        value = command.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            return (path if path.is_absolute() else base_dir / path).resolve()
    return None


def _upsert_image(
    root: Path,
    manifest: dict[str, Any],
    command: dict[str, Any],
    failures: FailureLog,
    *,
    require_existing: bool,
    base_dir: Path,
) -> None:
    name = command.get("image", command.get("name"))
    if not isinstance(name, str) or not name or Path(name).name != name or not name.lower().endswith(".png"):
        failures.add("E_EDIT_COMMAND", "edit", str(name), "image name must be a plain .png file name")
        return
    source = _resolve_source(command, ("source", "image_path", "image_source"), base_dir)
    if source is None or not source.is_file():
        failures.add("E_EDIT_COMMAND", "edit", name, "image source is missing", source=str(source) if source else None)
        return
    truth_source = _resolve_source(command, ("truth_source", "truth_path"), base_dir)
    if truth_source is not None and not truth_source.is_file():
        failures.add("E_EDIT_COMMAND", "edit", name, "truth source is missing", source=str(truth_source))
        return
    version_source = _resolve_source(command, ("version_source", "keypoint_source"), base_dir)
    if version_source is not None and not version_source.is_file():
        failures.add("E_EDIT_COMMAND", "edit", name, "keypoint version source is missing")
        return
    existing = _entry_by_name(manifest, name)
    if require_existing and existing is None:
        failures.add("E_EDIT_COMMAND", "edit", name, "replace_image target does not exist")
        return
    if not require_existing and existing is not None:
        failures.add("E_EDIT_COMMAND", "edit", name, "add_image target already exists")
        return

    image_relative = existing.get("path", f"images/{name}") if existing else f"images/{name}"
    truth_relative = (
        existing.get("truth_path", f"truth/{Path(name).stem}.json")
        if existing
        else f"truth/{Path(name).stem}.json"
    )
    if truth_source is not None:
        try:
            truth_doc = read_json(truth_source)
        except Exception as exc:
            failures.add("E_EDIT_COMMAND", "edit", name, f"cannot parse truth source: {exc}")
            return
    else:
        try:
            truth_doc = read_json(root / truth_relative)
        except Exception as exc:
            failures.add("E_EDIT_COMMAND", "edit", name, f"truth document is unavailable: {exc}")
            return

    copy_atomic(source, root / image_relative)
    if truth_source is not None:
        copy_atomic(truth_source, root / truth_relative)

    if existing is None:
        existing = {
            "name": name,
            "path": image_relative,
            "truth_path": truth_relative,
            "keypoint_versions": {},
        }
        manifest["images"].append(existing)
    else:
        existing["path"] = image_relative
        existing["truth_path"] = truth_relative
    existing["keypoint_versions"] = existing.get("keypoint_versions", {}) if isinstance(existing.get("keypoint_versions"), dict) else {}

    version = command.get("version", "v1")
    if not isinstance(version, str) or not version:
        failures.add("E_EDIT_COMMAND", "edit", name, "version label must be a non-empty string")
        return
    version_relative = f"keypoints/{version}/{Path(name).stem}.json"
    if version_source is not None:
        copy_atomic(version_source, root / version_relative)
    else:
        payload = {
            "schema_version": 1,
            "version": version,
            "image": name,
            "keypoints": truth_doc.get("keypoints", []) if isinstance(truth_doc, dict) else [],
        }
        atomic_write_json(root / version_relative, payload)
    existing["keypoint_versions"][version] = version_relative


def _remove_image(
    root: Path,
    manifest: dict[str, Any],
    command: dict[str, Any],
    failures: FailureLog,
    *,
    base_dir: Path,
) -> None:
    name = command.get("image", command.get("name"))
    if not isinstance(name, str):
        failures.add("E_EDIT_COMMAND", "edit", str(name), "remove_image needs an image name")
        return
    entry = _entry_by_name(manifest, name)
    if entry is None:
        failures.add("E_EDIT_COMMAND", "edit", name, "remove_image target does not exist")
        return
    manifest["images"] = [item for item in manifest["images"] if item is not entry]
    paths = [entry.get("path"), entry.get("truth_path")]
    versions = entry.get("keypoint_versions", {})
    if isinstance(versions, dict):
        paths.extend(versions.values())
    for relative in paths:
        if not isinstance(relative, str):
            continue
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            continue
        if target.is_file():
            target.unlink()


def _replace_keypoints(
    root: Path,
    manifest: dict[str, Any],
    command: dict[str, Any],
    failures: FailureLog,
    *,
    base_dir: Path,
) -> None:
    name = command.get("image", command.get("name"))
    version = command.get("version")
    if not isinstance(name, str) or not isinstance(version, str) or not version:
        failures.add("E_EDIT_COMMAND", "edit", str(name), "replace_keypoints needs image and version")
        return
    entry = _entry_by_name(manifest, name)
    if entry is None:
        failures.add("E_EDIT_COMMAND", "edit", name, "replace_keypoints target does not exist")
        return
    source = _resolve_source(command, ("source", "keypoint_source", "version_source"), base_dir)
    if source is None or not source.is_file():
        failures.add("E_EDIT_COMMAND", "edit", f"{name}:{version}", "keypoint version source is missing")
        return
    relative = f"keypoints/{version}/{Path(name).stem}.json"
    copy_atomic(source, root / relative)
    versions = entry.get("keypoint_versions")
    if not isinstance(versions, dict):
        versions = {}
        entry["keypoint_versions"] = versions
    versions[version] = relative


def _set_window(config: dict[str, Any], command: dict[str, Any], failures: FailureLog) -> None:
    window = config.setdefault("window", {})
    provided = False
    for key in ("width", "height", "step_x", "step_y"):
        if key in command:
            value = command[key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                failures.add("E_WINDOW_ALIGN", "edit", f"window.{key}", "window values must be positive integers")
            else:
                window[key] = value
                provided = True
    if not provided:
        failures.add("E_EDIT_COMMAND", "edit", "set_window", "no window field was supplied")
    _apply_to_active_strategy(config, "window", dict(window))
    if config.get("tier") not in ("small", "medium", "large"):
        config["tier"] = "custom"


def _set_step(config: dict[str, Any], command: dict[str, Any], failures: FailureLog) -> None:
    window = config.setdefault("window", {})
    for key in ("step_x", "step_y"):
        if key in command:
            value = command[key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                failures.add("E_WINDOW_ALIGN", "edit", f"window.{key}", "window steps must be positive integers")
            else:
                window[key] = value
    _apply_to_active_strategy(config, "window", dict(window))
    if config.get("tier") not in ("small", "medium", "large"):
        config["tier"] = "custom"


def _set_tier(config: dict[str, Any], command: dict[str, Any], failures: FailureLog) -> None:
    tier = command.get("tier", command.get("value"))
    if tier not in {"auto", "custom", "small", "medium", "large"}:
        failures.add("E_TIER", "edit", "tier", "unknown tier in edit command", actual=tier)
        return
    config["tier"] = tier


def _set_merge_threshold(config: dict[str, Any], command: dict[str, Any], failures: FailureLog) -> None:
    value = command.get("value", command.get("merge_gap", command.get("threshold")))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        failures.add("E_EDIT_COMMAND", "edit", "cc.merge_gap", "merge threshold must be a non-negative integer")
        return
    cc = config.setdefault("cc", {})
    cc["merge_gap"] = value
    tier = config.get("tier")
    if tier in ("small", "medium", "large"):
        strategies = config.get("strategies", {})
        if isinstance(strategies, dict) and isinstance(strategies.get(tier), dict):
            strategy_cc = strategies[tier].setdefault("cc", {})
            if isinstance(strategy_cc, dict):
                strategy_cc["merge_gap"] = value
    else:
        config["tier"] = "custom"


def _set_quantization(config: dict[str, Any], command: dict[str, Any], failures: FailureLog) -> None:
    precision = config.setdefault("precision", {})
    changed = False
    for key in ("fixed_bits", "quant_step"):
        if key in command:
            value = command[key]
            if not isinstance(value, int) or isinstance(value, bool):
                failures.add("E_QUANT_ENUM", "edit", f"precision.{key}", "quantization values must be integers")
            else:
                precision[key] = value
                changed = True
    if not changed:
        failures.add("E_EDIT_COMMAND", "edit", "set_quantization", "no quantization field supplied")
    tier = config.get("tier")
    if tier in ("small", "medium", "large"):
        strategies = config.get("strategies", {})
        if isinstance(strategies, dict) and isinstance(strategies.get(tier), dict):
            strategy_precision = strategies[tier].setdefault("precision", {})
            if isinstance(strategy_precision, dict):
                strategy_precision.update(precision)
    else:
        config["tier"] = "custom"


def _set_cc(config: dict[str, Any], command: dict[str, Any], failures: FailureLog) -> None:
    cc = config.setdefault("cc", {})
    for key in ("min_area", "max_aspect", "split_ratio", "ink_threshold"):
        if key in command:
            cc[key] = command[key]


def _set_nonnegative_int(
    config: dict[str, Any], command: dict[str, Any], key: str, failures: FailureLog, *, allow_zero: bool
) -> None:
    value = command.get("value", command.get(key))
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        failures.add("E_EDIT_COMMAND", "edit", key, f"value must be an integer >= {minimum}")
        return
    config[key] = value


def _apply_to_active_strategy(config: dict[str, Any], section: str, value: dict[str, Any]) -> None:
    tier = config.get("tier")
    if tier not in ("small", "medium", "large"):
        return
    strategies = config.get("strategies", {})
    if isinstance(strategies, dict) and isinstance(strategies.get(tier), dict):
        target = strategies[tier].get(section)
        if isinstance(target, dict):
            target.update(value)
