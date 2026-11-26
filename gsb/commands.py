"""Validated edit-command stream application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from gsb.config import RunConfig
from gsb.constants import E_COMMAND_INVALID, E_CONFIG_INVALID
from gsb.errors import GsbError, ValidationIssue, sort_issues
from gsb.fonts import TemplateLibrary
from gsb.models import DatasetState, EditCommand, SampleRecord
from gsb.pipeline import PipelineCache, PipelineResult, run_pipeline
from gsb.samples import load_annotation_version, load_sample_record
from gsb.util import safe_relative_path

SUPPORTED_OPS = {
    "set_quantization",
    "set_window",
    "set_threshold",
    "set_workers",
    "set_budget",
    "replace_annotation_version",
    "replace_image",
    "add_sample",
    "delete_sample",
    "noop",
}


@dataclass(frozen=True)
class CommandApplication:
    dataset: DatasetState
    config: RunConfig
    command_states: tuple[PipelineResult, ...]


def load_command_stream(path: str | Path) -> tuple[list[EditCommand], list[ValidationIssue]]:
    command_path = Path(path)
    issues: list[ValidationIssue] = []
    commands: list[EditCommand] = []
    try:
        lines = command_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], [
            ValidationIssue(
                code=E_COMMAND_INVALID,
                message="command stream not found",
                source=str(command_path),
            )
        ]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        source = f"{command_path}:{line_number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    code=E_COMMAND_INVALID,
                    message=f"invalid command JSON: {exc}",
                    source=source,
                )
            )
            continue
        if not isinstance(raw, dict):
            issues.append(
                ValidationIssue(
                    code=E_COMMAND_INVALID,
                    message="command must be a JSON object",
                    source=source,
                )
            )
            continue
        seq = raw.get("seq")
        op = raw.get("op")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
            issues.append(
                ValidationIssue(
                    code=E_COMMAND_INVALID,
                    message="command seq must be a positive integer",
                    source=source,
                )
            )
            continue
        if not isinstance(op, str) or op not in SUPPORTED_OPS:
            issues.append(
                ValidationIssue(
                    code=E_COMMAND_INVALID,
                    message=f"unsupported command op: {op}",
                    source=source,
                )
            )
            continue
        payload = {key: value for key, value in raw.items() if key not in {"seq", "op"}}
        commands.append(EditCommand(seq=seq, op=op, payload=payload))
    sequences = [command.seq for command in commands]
    if sequences != list(range(1, len(commands) + 1)):
        issues.append(
            ValidationIssue(
                code=E_COMMAND_INVALID,
                message="command seq values must be continuous from one",
                source=str(command_path),
            )
        )
    return commands, sort_issues(issues)


def _required_string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise GsbError(E_COMMAND_INVALID, f"command field {name} must be a non-empty string")
    return value


def _required_int(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise GsbError(E_COMMAND_INVALID, f"command field {name} must be an integer")
    return value


def _record_from_manifest(
    dataset: DatasetState,
    *,
    manifest_name: str,
    annotation_name: str | None,
    config: RunConfig,
    expected_sample_id: str,
) -> SampleRecord:
    manifest_path = safe_relative_path(dataset.dataset_path.parent, manifest_name)
    annotation_path = (
        safe_relative_path(dataset.dataset_path.parent, annotation_name)
        if annotation_name is not None
        else None
    )
    record, issues = load_sample_record(
        manifest_path,
        config=config,
        annotation_path=annotation_path,
    )
    if record is None or issues:
        first = issues[0] if issues else ValidationIssue(
            code=E_COMMAND_INVALID,
            message="sample could not be loaded",
            source=manifest_name,
        )
        raise GsbError(first.code, first.message, source=first.source, sample_id=expected_sample_id)
    if record.sample_id != expected_sample_id:
        raise GsbError(
            E_COMMAND_INVALID,
            "manifest sample_id does not match command sample_id",
            source=manifest_name,
            sample_id=expected_sample_id,
        )
    return record


def apply_commands(
    dataset: DatasetState,
    config: RunConfig,
    commands: Iterable[EditCommand],
) -> tuple[DatasetState, RunConfig, list[dict[str, Any]]]:
    state = dataset.clone()
    current_config = config
    log: list[dict[str, Any]] = []
    for command in commands:
        try:
            if command.op == "noop":
                pass
            elif command.op == "set_quantization":
                level = _required_string(command.payload, "level")
                current_config = current_config.with_quantization(level)
            elif command.op == "set_window":
                updates = {
                    name: _required_int(command.payload, name)
                    for name in ("width", "height", "stride_x", "stride_y")
                    if name in command.payload
                }
                if not updates:
                    raise GsbError(E_COMMAND_INVALID, "set_window requires at least one dimension")
                current_config = current_config.with_window(**updates)
            elif command.op == "set_threshold":
                name = _required_string(command.payload, "name")
                value = _required_int(command.payload, "value")
                current_config = current_config.with_threshold(name, value)
            elif command.op == "set_workers":
                current_config = current_config.with_workers(
                    _required_int(command.payload, "max_workers")
                )
            elif command.op == "set_budget":
                current_config = current_config.with_budget(
                    _required_int(command.payload, "max_template_points")
                )
            elif command.op == "replace_annotation_version":
                sample_id = _required_string(command.payload, "sample_id")
                if sample_id not in state.samples:
                    raise GsbError(E_COMMAND_INVALID, f"unknown sample: {sample_id}")
                annotation_name = _required_string(command.payload, "annotation")
                annotation_path = safe_relative_path(
                    state.dataset_path.parent,
                    annotation_name,
                )
                record = state.samples[sample_id]
                annotation, issues = load_annotation_version(
                    annotation_path,
                    manifest=record.manifest,
                    config=current_config,
                )
                if annotation is None or issues:
                    first = issues[0]
                    raise GsbError(first.code, first.message, source=first.source, sample_id=sample_id)
                state.samples[sample_id] = SampleRecord(
                    manifest=record.manifest,
                    image_bytes=record.image_bytes,
                    image_fingerprint=record.image_fingerprint,
                    annotation=annotation,
                )
                state.annotation_versions[sample_id] = annotation.version
            elif command.op == "replace_image":
                sample_id = _required_string(command.payload, "sample_id")
                if sample_id not in state.samples:
                    raise GsbError(E_COMMAND_INVALID, f"unknown sample: {sample_id}")
                manifest_name = _required_string(command.payload, "manifest")
                annotation_name = command.payload.get("annotation")
                if annotation_name is not None and not isinstance(annotation_name, str):
                    raise GsbError(E_COMMAND_INVALID, "annotation must be a string")
                record = _record_from_manifest(
                    state,
                    manifest_name=manifest_name,
                    annotation_name=annotation_name,
                    config=current_config,
                    expected_sample_id=sample_id,
                )
                state.samples[sample_id] = record
                if record.annotation is not None:
                    state.annotation_versions[sample_id] = record.annotation.version
            elif command.op == "add_sample":
                sample_id = _required_string(command.payload, "sample_id")
                if sample_id in state.samples:
                    raise GsbError(E_COMMAND_INVALID, f"sample already exists: {sample_id}")
                record = _record_from_manifest(
                    state,
                    manifest_name=_required_string(command.payload, "manifest"),
                    annotation_name=command.payload.get("annotation"),
                    config=current_config,
                    expected_sample_id=sample_id,
                )
                state.samples[sample_id] = record
                state.sample_order.append(sample_id)
                if record.annotation is not None:
                    state.annotation_versions[sample_id] = record.annotation.version
            elif command.op == "delete_sample":
                sample_id = _required_string(command.payload, "sample_id")
                if sample_id not in state.samples:
                    raise GsbError(E_COMMAND_INVALID, f"unknown sample: {sample_id}")
                del state.samples[sample_id]
                state.sample_order = [item for item in state.sample_order if item != sample_id]
                state.annotation_versions.pop(sample_id, None)
            else:
                raise GsbError(E_COMMAND_INVALID, f"unsupported command op: {command.op}")
            state.command_cursor = command.seq
            log.append({"seq": command.seq, "op": command.op, "status": "applied"})
        except (GsbError, ValueError) as exc:
            if isinstance(exc, GsbError):
                issue = exc.issue()
            else:
                issue = ValidationIssue(
                    code=E_COMMAND_INVALID,
                    message=str(exc),
                    source=str(command.seq),
                )
            log.append(
                {
                    "seq": command.seq,
                    "op": command.op,
                    "status": "failed",
                    "error": issue.to_dict(),
                }
            )
            state.command_cursor = command.seq
    return state, current_config, log


def run_command_stream(
    *,
    dataset: DatasetState,
    config: RunConfig,
    templates: TemplateLibrary,
    commands: Iterable[EditCommand],
    cache: PipelineCache | None = None,
    prior_result: PipelineResult | None = None,
) -> tuple[PipelineResult, list[dict[str, Any]], PipelineCache]:
    current_dataset = dataset
    current_config = config
    current_cache = cache or (prior_result.cache if prior_result is not None else PipelineCache())
    prior_recognition = prior_result.recognition_results if prior_result is not None else {}
    pending_ids = prior_result.pending_ids if prior_result is not None else ()
    last_result = prior_result
    logs: list[dict[str, Any]] = []
    for command in commands:
        updated_dataset, updated_config, command_log = apply_commands(
            current_dataset,
            current_config,
            [command],
        )
        current_dataset = updated_dataset
        current_config = updated_config
        entry = command_log[0]
        if entry["status"] == "applied":
            result = run_pipeline(
                current_dataset,
                config=current_config,
                templates=templates,
                cache=current_cache,
                prior_recognition_results=prior_recognition,
                pending_ids=pending_ids,
            )
            current_cache = result.cache
            prior_recognition = result.recognition_results
            pending_ids = result.pending_ids
            last_result = result
            entry["cache_hits"] = result.report["incremental"]["cache_hits"]
            entry["recognition_reuses"] = result.report["incremental"]["recognition_reuses"]
            entry["command_cursor"] = current_dataset.command_cursor
        logs.append(entry)
    assert last_result is not None
    return last_result, logs, current_cache
