from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

import yaml

from .config import InputLoadError, resolve_inputs, validate_inputs
from .pipeline import PipelineError, execute, failed_report
from .snapshot import SnapshotError, SnapshotStore
from .util import SCHEME_ORDER, atomic_write_text, dump_yaml, failure, is_int, load_yaml


def run_service(
    config_path: Path,
    state_dir: Path,
    expect_version: int | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    try:
        bundle = resolve_inputs(config_path)
    except InputLoadError as exc:
        return failed_report("E_CONFIG_INVALID", exc.failures, "input loading failed")
    store = SnapshotStore(state_dir)
    try:
        manifest = store.load_manifest()
    except SnapshotError as exc:
        return failed_report(
            "E_STATE_INVALID",
            [failure("E_STATE_INVALID", "snapshot", "state", str(exc))],
            "snapshot state is invalid",
        )
    if manifest is not None and bundle["config_hash"] != manifest.get("config_hash"):
        return failed_report(
            "E_CONFIG_MISMATCH",
            [
                failure(
                    "E_CONFIG_MISMATCH",
                    "snapshot",
                    "config_hash",
                    "configuration was modified outside the controlled edit command",
                )
            ],
            "configuration hash mismatch",
        )
    if expect_version is not None:
        found_version = manifest.get("global_version") if manifest is not None else None
        if found_version is None or int(found_version) != int(expect_version):
            return failed_report(
                "E_CONFIG_MISMATCH",
                [
                    failure(
                        "E_CONFIG_MISMATCH",
                        "snapshot",
                        "global_version",
                        f"expected {expect_version}, found {found_version}",
                    )
                ],
                "snapshot version mismatch",
            )
    failures = validate_inputs(bundle)
    if failures:
        return failed_report("E_CONFIG_INVALID", failures, "configuration validation failed")
    try:
        return execute(
            bundle,
            state_dir,
            stop_event=stop_event,
            expect_version=expect_version,
        )
    except PipelineError as exc:
        return failed_report(exc.code, exc.failures, str(exc))
    except SnapshotError as exc:
        return failed_report(
            "E_STATE_INVALID",
            [failure("E_STATE_INVALID", "snapshot", "state", str(exc))],
            "snapshot state is invalid",
        )
    except Exception as exc:
        return failed_report(
            "E_RUNTIME",
            [failure("E_RUNTIME", "runtime", "pipeline", f"{type(exc).__name__}: {exc}")],
            "pipeline execution failed",
        )


def edit_service(
    config_path: Path,
    state_dir: Path,
    edit_path: Path,
    expect_version: int | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    store = SnapshotStore(state_dir)
    try:
        manifest = store.load_manifest()
    except SnapshotError as exc:
        return failed_report(
            "E_STATE_INVALID",
            [failure("E_STATE_INVALID", "snapshot", "state", str(exc))],
            "snapshot state is invalid",
        )
    if manifest is None:
        return failed_report(
            "E_CONFIG_MISMATCH",
            [
                failure(
                    "E_CONFIG_MISMATCH",
                    "snapshot",
                    "manifest",
                    "edit requires an existing successful snapshot",
                )
            ],
            "no snapshot exists",
        )
    if expect_version is not None and int(manifest.get("global_version", -1)) != int(expect_version):
        return failed_report(
            "E_CONFIG_MISMATCH",
            [
                failure(
                    "E_CONFIG_MISMATCH",
                    "snapshot",
                    "global_version",
                    f"expected {expect_version}, found {manifest.get('global_version')}",
                )
            ],
            "snapshot version mismatch",
        )
    try:
        current_bundle = resolve_inputs(config_path)
    except InputLoadError as exc:
        return failed_report("E_CONFIG_INVALID", exc.failures, "input loading failed")
    if current_bundle["config_hash"] != manifest.get("config_hash"):
        return failed_report(
            "E_CONFIG_MISMATCH",
            [
                failure(
                    "E_CONFIG_MISMATCH",
                    "snapshot",
                    "config_hash",
                    "configuration was modified outside the controlled edit command",
                )
            ],
            "configuration hash mismatch",
        )
    failures = validate_inputs(current_bundle)
    if failures:
        return failed_report("E_CONFIG_INVALID", failures, "configuration validation failed")
    try:
        edit_document = load_yaml(edit_path.resolve())
    except (OSError, yaml.YAMLError) as exc:
        return failed_report(
            "E_COMMAND_INVALID",
            [failure("E_COMMAND_INVALID", "command", "edit", f"cannot read edit YAML: {exc}")],
            "edit document cannot be loaded",
        )
    if not isinstance(edit_document, dict):
        return failed_report(
            "E_COMMAND_INVALID",
            [failure("E_COMMAND_INVALID", "command", "edit", "root must be a mapping")],
            "invalid edit document",
        )
    expected_hash = edit_document.get("base_config_hash")
    if expected_hash is not None and expected_hash != current_bundle["config_hash"]:
        return failed_report(
            "E_CONFIG_MISMATCH",
            [
                failure(
                    "E_CONFIG_MISMATCH",
                    "command",
                    "base_config_hash",
                    "edit document was prepared for a different configuration version",
                )
            ],
            "edit base configuration hash mismatch",
        )
    new_config = copy.deepcopy(current_bundle["config"])
    new_overrides = copy.deepcopy(current_bundle["overrides"])
    operations = edit_document.get("operations")
    if not isinstance(operations, list) or not operations:
        return failed_report(
            "E_COMMAND_INVALID",
            [failure("E_COMMAND_INVALID", "command", "operations", "must be a non-empty list")],
            "invalid edit operations",
        )
    command_failures: list[dict[str, str]] = []
    for index, operation in enumerate(operations):
        _apply_operation(operation, index, new_config, new_overrides, command_failures)
    if command_failures:
        return failed_report("E_COMMAND_INVALID", command_failures, "invalid edit command")

    prospective_bundle = dict(current_bundle)
    prospective_bundle["config"] = new_config
    prospective_bundle["overrides"] = new_overrides
    validation_errors = validate_inputs(prospective_bundle)
    if validation_errors:
        return failed_report(
            "E_CONFIG_INVALID", validation_errors, "edited configuration validation failed"
        )

    original_config_bytes = config_path.read_bytes()
    original_overrides_bytes = current_bundle["overrides_path"].read_bytes()
    config_changed = new_config != current_bundle["config"]
    overrides_changed = new_overrides != current_bundle["overrides"]
    if config_changed:
        atomic_write_text(config_path, dump_yaml(new_config))
    if overrides_changed:
        atomic_write_text(current_bundle["overrides_path"], dump_yaml(new_overrides))
    try:
        bundle = resolve_inputs(config_path)
        post_failures = validate_inputs(bundle)
        if post_failures:
            raise PipelineError(
                "E_CONFIG_INVALID", "edited configuration became invalid after serialization", post_failures
            )
        return execute(
            bundle,
            state_dir,
            stop_event=stop_event,
            expect_version=expect_version,
            skip_config_hash_check=True,
        )
    except PipelineError as exc:
        _restore_bytes(config_path, original_config_bytes)
        _restore_bytes(current_bundle["overrides_path"], original_overrides_bytes)
        return failed_report(exc.code, exc.failures, str(exc))
    except Exception as exc:
        _restore_bytes(config_path, original_config_bytes)
        _restore_bytes(current_bundle["overrides_path"], original_overrides_bytes)
        return failed_report(
            "E_RUNTIME",
            [failure("E_RUNTIME", "command", "edit", f"{type(exc).__name__}: {exc}")],
            "edit command failed",
        )


def _restore_bytes(path: Path, data: bytes) -> None:
    atomic_write_text(path, data.decode("utf-8"))


def _apply_operation(
    operation: Any,
    index: int,
    config: dict[str, Any],
    overrides: dict[str, Any],
    failures: list[dict[str, str]],
) -> None:
    prefix = f"operations[{index}]"

    def add(field: str, message: str) -> None:
        failures.append(failure("E_COMMAND_INVALID", "command", f"{prefix}.{field}", message))

    if not isinstance(operation, dict):
        add("operation", "must be a mapping")
        return
    op = operation.get("op")
    if op == "switch_calibration":
        scheme = operation.get("scheme")
        enabled = operation.get("enabled")
        if scheme not in SCHEME_ORDER:
            add("scheme", f"must be one of {', '.join(SCHEME_ORDER)}")
            return
        if not isinstance(enabled, bool):
            add("enabled", "must be boolean")
            return
        config["calibrators"][scheme]["enabled"] = enabled
        return
    if op == "set_bins":
        bins = operation.get("bins")
        if not is_int(bins):
            add("bins", "must be an integer")
            return
        target = operation.get("target")
        scheme = operation.get("scheme")
        if target is None:
            target = "calibration" if scheme is not None else "evaluation"
        if target == "evaluation":
            config["evaluation"]["calibration_bins"] = bins
            return
        if target != "calibration":
            add("target", "must be evaluation or calibration")
            return
        if scheme not in ("isotonic", "histogram"):
            add("scheme", "binned calibration scheme required")
            return
        config["calibrators"][scheme]["bins"] = bins
        return
    if op == "set_regularization":
        scheme = operation.get("scheme")
        if scheme not in SCHEME_ORDER:
            add("scheme", f"must be one of {', '.join(SCHEME_ORDER)}")
            return
        if "strength" not in operation:
            add("strength", "is required")
            return
        config["calibrators"][scheme]["regularization"] = operation["strength"]
        return
    if op == "set_threshold_step":
        if "step" not in operation:
            add("step", "is required")
            return
        config["evaluation"]["threshold_step"] = operation["step"]
        return
    if op == "replace_sample":
        sample_id = operation.get("sample_id")
        if not is_int(sample_id) or sample_id <= 0:
            add("sample_id", "must be a positive integer")
            return
        values = operation.get("values", {})
        if not isinstance(values, dict):
            add("values", "must be a mapping")
            return
        target = operation.get("target")
        if target is not None and target not in (0, 1):
            add("target", "must be 0 or 1")
            return
        samples = overrides.setdefault("samples", {})
        if not isinstance(samples, dict):
            add("samples", "overrides samples must be a mapping")
            return
        entry = samples.setdefault(str(sample_id), {})
        if not isinstance(entry, dict):
            add("samples", "sample override must be a mapping")
            return
        merged_values = dict(entry.get("values", {}))
        merged_values.update(values)
        if merged_values:
            entry["values"] = merged_values
        if target is not None:
            entry["target"] = target
        return
    add("op", f"unsupported operation: {op}")


def verify_service(state_dir: Path) -> dict[str, Any]:
    store = SnapshotStore(state_dir)
    try:
        manifest = store.load_manifest()
    except SnapshotError as exc:
        return failed_report(
            "E_STATE_INVALID",
            [failure("E_STATE_INVALID", "snapshot", "state", str(exc))],
            "snapshot state is invalid",
        )
    if manifest is None:
        return failed_report(
            "E_STATE_INVALID",
            [failure("E_STATE_INVALID", "snapshot", "manifest", "no manifest exists")],
            "no snapshot exists",
        )
    return {
        "status": "ok",
        "business_code": None,
        "manifest": manifest,
        "layout": store.verify_layout(),
    }
