"""Strict, non-short-circuiting loaders for configs, samples, and commands."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .config import (
    SCHEME_ALIASES,
    SCHEME_DISPLAY,
    WorkbenchConfig,
    apply_config_patch,
)
from .hashing import sha256_json
from .models import Issue, LoadedCommands, LoadedSamples, Sample


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sample_issue(
    code: str,
    sample_id: str | None,
    field: str,
    message: str,
    line: int,
) -> Issue:
    return Issue(code=code, sample_id=sample_id, field=field, message=message, line=line)


def _vector(value: Any) -> tuple[list[float] | None, str | None]:
    if not isinstance(value, list):
        return None, "must be an array"
    result: list[float] = []
    for index, item in enumerate(value):
        if not _is_finite_number(item):
            return None, f"element {index} must be a finite number"
        result.append(float(item))
    return result, None


def parse_sample(
    raw: Any,
    config: WorkbenchConfig,
    line: int,
) -> tuple[Sample | None, list[Issue]]:
    issues: list[Issue] = []
    if not isinstance(raw, dict):
        return None, [_sample_issue("E_SAMPLE_OBJECT", None, "$", "sample must be an object", line)]

    sample_id = raw.get("sample_id", raw.get("id"))
    if not isinstance(sample_id, str) or not sample_id.strip():
        issues.append(
            _sample_issue("E_SAMPLE_ID", None, "sample_id", "must be a non-empty string", line)
        )
        sample_id_for_issue = None
    else:
        sample_id = sample_id.strip()
        sample_id_for_issue = sample_id

    query_position = raw.get("query_position")
    if not _is_int(query_position) or query_position < 0:
        issues.append(
            _sample_issue(
                "E_SAMPLE_QUERY_POSITION",
                sample_id_for_issue,
                "query_position",
                "must be a non-negative integer",
                line,
            )
        )

    raw_key_positions = raw.get("key_positions")
    key_positions: list[int] = []
    if not isinstance(raw_key_positions, list):
        issues.append(
            _sample_issue(
                "E_SAMPLE_KEY_POSITIONS",
                sample_id_for_issue,
                "key_positions",
                "must be an array",
                line,
            )
        )
    else:
        for index, position in enumerate(raw_key_positions):
            if not _is_int(position) or position < 0:
                issues.append(
                    _sample_issue(
                        "E_SAMPLE_KEY_POSITION",
                        sample_id_for_issue,
                        f"key_positions[{index}]",
                        "must be a non-negative integer",
                        line,
                    )
                )
            else:
                key_positions.append(int(position))

    query_vector, query_error = _vector(raw.get("query_vector"))
    if query_error is not None:
        issues.append(
            _sample_issue(
                "E_SAMPLE_QUERY_VECTOR",
                sample_id_for_issue,
                "query_vector",
                query_error,
                line,
            )
        )
    elif len(query_vector or []) != config.dimension:
        issues.append(
            _sample_issue(
                "E_SAMPLE_FEATURE_LENGTH",
                sample_id_for_issue,
                "query_vector",
                f"length {len(query_vector or [])} does not equal dimension {config.dimension}",
                line,
            )
        )
        query_vector = None

    raw_key_vectors = raw.get("key_vectors")
    key_vectors: list[list[float]] = []
    if not isinstance(raw_key_vectors, list):
        issues.append(
            _sample_issue(
                "E_SAMPLE_KEY_VECTORS",
                sample_id_for_issue,
                "key_vectors",
                "must be an array",
                line,
            )
        )
    else:
        for index, vector_value in enumerate(raw_key_vectors):
            vector, error = _vector(vector_value)
            if error is not None:
                issues.append(
                    _sample_issue(
                        "E_SAMPLE_KEY_VECTOR",
                        sample_id_for_issue,
                        f"key_vectors[{index}]",
                        error,
                        line,
                    )
                )
                continue
            assert vector is not None
            if len(vector) != config.dimension:
                issues.append(
                    _sample_issue(
                        "E_SAMPLE_FEATURE_LENGTH",
                        sample_id_for_issue,
                        f"key_vectors[{index}]",
                        f"length {len(vector)} does not equal dimension {config.dimension}",
                        line,
                    )
                )
                continue
            key_vectors.append(vector)

    if isinstance(raw_key_positions, list) and isinstance(raw_key_vectors, list):
        if len(raw_key_positions) != len(raw_key_vectors):
            issues.append(
                _sample_issue(
                    "E_SAMPLE_KEY_COUNT",
                    sample_id_for_issue,
                    "key_positions/key_vectors",
                    "key position and key vector counts must match",
                    line,
                )
            )

    if issues:
        return None, issues

    assert isinstance(sample_id, str)
    assert isinstance(query_position, int)
    assert query_vector is not None
    canonical = {
        "sample_id": sample_id,
        "query_position": query_position,
        "key_positions": key_positions,
        "query_vector": query_vector,
        "key_vectors": key_vectors,
    }
    sample = Sample(
        sample_id=sample_id,
        query_position=query_position,
        key_positions=tuple(key_positions),
        query_vector=tuple(query_vector),
        key_vectors=tuple(tuple(vector) for vector in key_vectors),
        content_fingerprint=sha256_json(canonical),
        line=line,
    )
    return sample, []


def parse_samples_jsonl(path: str | Path, config: WorkbenchConfig) -> LoadedSamples:
    sample_path = Path(path)
    issues: list[Issue] = []
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    duplicate_count = 0
    try:
        lines = sample_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return LoadedSamples(
            samples=[],
            issues=[
                Issue(
                    code="E_SAMPLE_FILE",
                    sample_id=None,
                    field=str(sample_path),
                    message="sample file does not exist",
                )
            ],
        )
    except (OSError, UnicodeError) as exc:
        return LoadedSamples(
            samples=[],
            issues=[
                Issue(
                    code="E_SAMPLE_FILE",
                    sample_id=None,
                    field=str(sample_path),
                    message=f"cannot read sample file: {exc}",
                )
            ],
        )

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                _sample_issue(
                    "E_SAMPLE_JSON",
                    None,
                    "$",
                    f"invalid JSON: {exc.msg}",
                    line_no,
                )
            )
            continue
        sample, sample_issues = parse_sample(raw, config, line_no)
        raw_id = raw.get("sample_id", raw.get("id")) if isinstance(raw, dict) else None
        normalized_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else None
        if normalized_id is not None and normalized_id in seen_ids:
            duplicate_count += 1
            issues.append(
                _sample_issue(
                    "E_SAMPLE_ID_DUPLICATE",
                    normalized_id,
                    "sample_id",
                    "sample_id must be unique",
                    line_no,
                )
            )
            continue
        if normalized_id is not None:
            seen_ids.add(normalized_id)
        issues.extend(sample_issues)
        if sample is not None:
            samples.append(sample)

    samples.sort(key=lambda sample: sample.sample_id)
    issues.sort(key=lambda issue: issue.sort_key)
    return LoadedSamples(
        samples=samples,
        issues=issues,
        line_count=len(lines),
        duplicate_count=duplicate_count,
    )


COMMAND_NAMES = {"run", "set", "sensitivity", "report", "status"}


def _command_issue(
    code: str,
    line: int,
    field: str,
    message: str,
) -> Issue:
    return Issue(
        code=code,
        sample_id=None,
        field=field,
        message=message,
        line=line,
    )


def _validate_positive_int_param(value: Any, field: str, line: int) -> list[Issue]:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return [_command_issue("E_COMMAND_PARAM", line, field, "must be a positive integer")]
    return []


def parse_commands_jsonl(
    path: str | Path,
    config: WorkbenchConfig | None = None,
) -> LoadedCommands:
    command_path = Path(path)
    try:
        lines = command_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return LoadedCommands(
            commands=[],
            issues=[
                Issue(
                    code="E_COMMAND_FILE",
                    sample_id=None,
                    field=str(command_path),
                    message="command file does not exist",
                )
            ],
        )
    except (OSError, UnicodeError) as exc:
        return LoadedCommands(
            commands=[],
            issues=[
                Issue(
                    code="E_COMMAND_FILE",
                    sample_id=None,
                    field=str(command_path),
                    message=f"cannot read command file: {exc}",
                )
            ],
        )

    commands: list[dict[str, Any]] = []
    issues: list[Issue] = []
    shadow_config = config
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                _command_issue("E_COMMAND_JSON", line_no, "$", f"invalid JSON: {exc.msg}")
            )
            continue
        if not isinstance(raw, dict):
            issues.append(_command_issue("E_COMMAND_OBJECT", line_no, "$", "must be an object"))
            continue
        name = raw.get("command")
        if not isinstance(name, str) or name not in COMMAND_NAMES:
            issues.append(
                _command_issue(
                    "E_COMMAND_NAME",
                    line_no,
                    "command",
                    f"command must be one of {sorted(COMMAND_NAMES)}",
                )
            )
            continue
        params = raw.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            issues.append(_command_issue("E_COMMAND_PARAMS", line_no, "params", "must be an object"))
            continue
        issue_count_before_command = len(issues)
        command = {"command": name, "params": params, "line": line_no}

        if name == "run":
            for key in ("point_budget", "sample_quota", "snapshot_interval"):
                if key in params:
                    issues.extend(
                        _validate_positive_int_param(params[key], f"params.{key}", line_no)
                    )
        elif name == "set":
            patch = raw.get("patch", params.get("patch"))
            if not isinstance(patch, dict):
                issues.append(
                    _command_issue("E_COMMAND_PATCH", line_no, "patch", "must be an object")
                )
                continue
            if shadow_config is not None:
                patched_config, patch_issues = apply_config_patch(shadow_config, patch)
                for issue in patch_issues:
                    issues.append(
                        _command_issue(issue.code, line_no, f"patch.{issue.field}", issue.message)
                    )
                if patched_config is not None:
                    shadow_config = patched_config
            command["patch"] = patch
        elif name == "sensitivity":
            for key in ("precision_levels", "quantization_levels", "fixed_decimal_widths"):
                if key in params and (not isinstance(params[key], list) or not params[key]):
                    issues.append(
                        _command_issue(
                            "E_COMMAND_PARAM",
                            line_no,
                            f"params.{key}",
                            "must be a non-empty array",
                        )
                    )
            if "scheme_candidates" in params:
                values = params["scheme_candidates"]
                if not isinstance(values, list) or not values:
                    issues.append(
                        _command_issue(
                            "E_COMMAND_PARAM",
                            line_no,
                            "params.scheme_candidates",
                            "must be a non-empty array",
                        )
                    )
                elif any(
                    not isinstance(value, str)
                    or SCHEME_ALIASES.get(value, SCHEME_ALIASES.get(value.strip(), value))
                    not in SCHEME_DISPLAY
                    for value in values
                ):
                    issues.append(
                        _command_issue(
                            "E_COMMAND_PARAM",
                            line_no,
                            "params.scheme_candidates",
                            "contains an unsupported scheme",
                        )
                    )
            if "data_scales" in params:
                scales = params["data_scales"]
                if (
                    not isinstance(scales, list)
                    or not scales
                    or any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        or not 0 < float(value) <= 1
                        for value in scales
                    )
                ):
                    issues.append(
                        _command_issue(
                            "E_COMMAND_PARAM",
                            line_no,
                            "params.data_scales",
                            "must be a non-empty array of values in (0, 1]",
                        )
                    )
        if len(issues) == issue_count_before_command:
            commands.append(command)

    issues.sort(key=lambda issue: issue.sort_key)
    return LoadedCommands(commands=commands, issues=issues, line_count=len(lines))


def merge_issues(*groups: Iterable[Issue]) -> list[Issue]:
    merged: list[Issue] = []
    for group in groups:
        merged.extend(group)
    merged.sort(key=lambda issue: issue.sort_key)
    return merged
