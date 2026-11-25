"""Validation for JSONL corpus, model package, lexicon and command input."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .codes import Code
from .labels import LABELS, TYPE_CODES, parse_label


class NonFiniteConstant(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise NonFiniteConstant(f"non-finite JSON constant: {value}")


def load_json_lenient(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text, parse_constant=_reject_constant)


def _failure(line: int | None, sample_id: str | None, code: str, message: str, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "message": message}
    if line is not None:
        item["line"] = line
    if sample_id is not None:
        item["sample_id"] = sample_id
    item.update(extra)
    return item


def _sequence(row: dict[str, Any], sequence_name: str) -> list[Any] | None:
    spaced = row.get(f"{sequence_name}_sequence")
    if isinstance(spaced, str):
        return spaced.split()
    value = row.get(sequence_name)
    if isinstance(value, list):
        return value
    if sequence_name == "char" and isinstance(value := row.get("text"), str):
        return value.split() if " " in value else list(value)
    return None


def validate_bieso_labels(labels: list[str]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    open_start: int | None = None
    open_type: str | None = None

    def close_invalid(position: int, detail: str) -> None:
        errors.append({"position": position, "code": Code.INVALID_BIESO_SPAN, "detail": detail})

    for index, label in enumerate(labels):
        parsed = parse_label(label)
        if parsed is None:
            continue
        prefix, entity_type = parsed
        if prefix == "O":
            if open_start is not None:
                close_invalid(index, f"位置 {open_start} 的 {open_type} 片段被 O 未闭合打断")
                open_start = None
                open_type = None
            continue
        if prefix == "S":
            if open_start is not None:
                close_invalid(index, f"S-{entity_type} 前存在未闭合 {open_type} 片段")
                open_start = None
                open_type = None
            continue
        if prefix == "B":
            if open_start is not None:
                close_invalid(index, f"B-{entity_type} 打断位置 {open_start} 的未闭合 {open_type} 片段")
            open_start = index
            open_type = entity_type
            continue
        if open_start is None or open_type is None:
            close_invalid(index, f"{label} 前缺少同类型 B")
            continue
        if entity_type != open_type:
            close_invalid(index, f"{label} 与 B-{open_type} 类型不一致")
        elif prefix == "E":
            open_start = None
            open_type = None

    if open_start is not None:
        close_invalid(len(labels) - 1, f"从位置 {open_start} 开始的 {open_type} 片段到达行尾仍未闭合")
    return errors


def parse_corpus(lines: Iterable[str] | str | Path) -> dict[str, Any]:
    if isinstance(lines, (str, Path)):
        lines = Path(lines).read_text(encoding="utf-8").splitlines()
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        sample_id: str | None = None
        try:
            row = json.loads(raw_line, parse_constant=_reject_constant)
            if not isinstance(row, dict):
                raise ValueError("JSONL row is not an object")
            raw_id = row.get("id")
            if not isinstance(raw_id, str) or not raw_id:
                raise ValueError("missing non-empty string id")
            sample_id = raw_id
            if sample_id in seen:
                failures.append(_failure(line_number, sample_id, Code.DUPLICATE_SAMPLE_ID, "样本标识重复", evidence=sample_id))
                continue

            chars = _sequence(row, "char")
            labels = _sequence(row, "label")
            if chars is None or labels is None:
                failures.append(_failure(line_number, sample_id, Code.INVALID_SAMPLE, "缺少 char_sequence/label_sequence 或其列表形式", field="sequence"))
                continue
            if not all(isinstance(char, str) and char for char in chars):
                failures.append(_failure(line_number, sample_id, Code.INVALID_SAMPLE, "字序列包含空值或非字符串", field="char"))
                continue
            if not all(isinstance(label, str) and label for label in labels):
                failures.append(_failure(line_number, sample_id, Code.INVALID_SAMPLE, "标签序列包含空值或非字符串", field="label"))
                continue
            if len(chars) != len(labels):
                failures.append(
                    _failure(
                        line_number,
                        sample_id,
                        Code.TAG_LENGTH_MISMATCH,
                        "字序列与标签序列长度不一致",
                        char_count=len(chars),
                        label_count=len(labels),
                    )
                )
                continue
            unknown = [label for label in labels if label not in LABELS]
            if unknown:
                failures.append(_failure(line_number, sample_id, Code.UNKNOWN_TAG, "存在枚举外标签", unknown_labels=sorted(set(unknown))))
                continue
            span_errors = validate_bieso_labels(labels)
            if span_errors:
                failures.append(_failure(line_number, sample_id, Code.INVALID_BIESO_SPAN, "BIESO 片段非法", span_errors=span_errors[:10]))
                continue

            seen.add(sample_id)
            samples.append({"id": sample_id, "chars": list(chars), "labels": list(labels)})
        except NonFiniteConstant as exc:
            failures.append(_failure(line_number, sample_id, Code.NON_FINITE_WEIGHT, str(exc)))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            failures.append(_failure(line_number, sample_id, Code.INVALID_JSONL, f"JSON/结构无法解析：{exc}"))

    counts: dict[str, int] = {}
    for item in failures:
        counts[item["code"]] = counts.get(item["code"], 0) + 1
    return {"samples": samples, "failures": failures, "business_counts": counts}


def parse_lexicon(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        try:
            data = load_json_lenient(source)
        except (OSError, json.JSONDecodeError, NonFiniteConstant) as exc:
            return {"entries": [], "failures": [_failure(None, None, Code.PACKAGE_TRUNCATED, f"词表无法读取：{exc}")]}
    else:
        data = source
    raw_entries = data.get("entries", data) if isinstance(data, dict) else data
    entries: list[dict[str, str]] = []
    failures: list[dict[str, Any]] = []
    if not isinstance(raw_entries, list):
        return {"entries": [], "failures": [_failure(None, None, Code.LEXICON_INVALID, "词表必须是 entries 列表或数组")]}
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_entries):
        locator = f"entries[{index}]"
        if not isinstance(raw, dict):
            failures.append(_failure(None, None, Code.LEXICON_INVALID, "条目不是对象", locator=locator))
            continue
        entry_id = raw.get("id")
        text = raw.get("text")
        entity_type = raw.get("type")
        if not isinstance(entry_id, str) or not entry_id or entry_id in seen_ids:
            failures.append(_failure(None, entry_id if isinstance(entry_id, str) else None, Code.LEXICON_INVALID, "条目 id 缺失或重复", locator=locator))
            continue
        if not isinstance(text, str) or not text:
            failures.append(_failure(None, entry_id, Code.LEXICON_INVALID, "条目 text 必须是非空字符串", locator=locator))
            continue
        if entity_type not in TYPE_CODES:
            failures.append(_failure(None, entry_id, Code.LEXICON_INVALID, f"未知实体类型：{entity_type}", locator=locator))
            continue
        seen_ids.add(entry_id)
        entries.append({"id": entry_id, "text": text, "type": entity_type})
    return {"entries": entries, "failures": failures}


def _walk_finite(value: Any, path: str = "$") -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return path
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            found = _walk_finite(item, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, dict):
        for key, item in value.items():
            found = _walk_finite(item, f"{path}.{key}")
            if found:
                return found
    return None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_package(data: Any) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return [_failure(None, None, Code.INVALID_PACKAGE, "参数包顶层必须是对象")]

    version = data.get("model_version")
    if not isinstance(version, str) or not version:
        failures.append(_failure(None, None, Code.INVALID_MODEL_VERSION, "model_version 必须是非空字符串", locator="model_version"))

    feature_dim = data.get("feature_dim")
    bucket_count = data.get("bucket_count")
    if not _positive_int(feature_dim):
        failures.append(_failure(None, None, Code.INVALID_FEATURE_DIM, "feature_dim 必须是正整数", feature_dim=feature_dim))
    if not _positive_int(bucket_count):
        failures.append(_failure(None, None, Code.INVALID_BUCKET_COUNT, "bucket_count 必须是正整数", bucket_count=bucket_count))

    nonfinite_path = _walk_finite(data)
    if nonfinite_path:
        failures.append(_failure(None, None, Code.NON_FINITE_WEIGHT, "参数包含非有限数", locator=nonfinite_path))

    if failures:
        return failures

    assert isinstance(feature_dim, int) and isinstance(bucket_count, int)
    labels = data.get("label_encoding", list(LABELS))
    if labels != list(LABELS):
        failures.append(_failure(None, None, Code.FEATURE_DIM_OUT_OF_RANGE, "label_encoding 与固定标签编码不一致", locator="label_encoding"))
        return failures

    hash_table = data.get("hash_table")
    emission = data.get("emission_weights")
    transitions = data.get("transition_weights")
    if not isinstance(hash_table, list) or len(hash_table) != bucket_count:
        failures.append(_failure(None, None, Code.FEATURE_DIM_OUT_OF_RANGE, "hash_table 行数必须等于 bucket_count", expected=bucket_count, actual=len(hash_table) if isinstance(hash_table, list) else None, locator="hash_table"))
    else:
        for bucket, row in enumerate(hash_table):
            if not isinstance(row, list) or len(row) != feature_dim:
                failures.append(_failure(None, None, Code.FEATURE_DIM_OUT_OF_RANGE, "hash_table 列数必须等于 feature_dim", bucket=bucket, expected=feature_dim, locator=f"hash_table[{bucket}]"))
                break

    label_count = len(LABELS)
    if not isinstance(emission, list) or len(emission) != label_count:
        failures.append(_failure(None, None, Code.FEATURE_DIM_OUT_OF_RANGE, "emission_weights 第一维必须等于标签数", expected=label_count, actual=len(emission) if isinstance(emission, list) else None, locator="emission_weights"))
    else:
        for label_index, rows in enumerate(emission):
            if not isinstance(rows, list) or len(rows) != feature_dim:
                failures.append(_failure(None, None, Code.FEATURE_DIM_OUT_OF_RANGE, "emission_weights 第二维必须等于 feature_dim", label_index=label_index, locator=f"emission_weights[{label_index}]"))
                break
            for feature_index, buckets in enumerate(rows):
                if not isinstance(buckets, list) or len(buckets) != bucket_count:
                    failures.append(_failure(None, None, Code.FEATURE_DIM_OUT_OF_RANGE, "emission_weights 第三维必须等于 bucket_count", label_index=label_index, feature_index=feature_index, locator=f"emission_weights[{label_index}][{feature_index}]"))
                    break
            else:
                continue
            break

    if not isinstance(transitions, dict):
        failures.append(_failure(None, None, Code.INVALID_PACKAGE, "transition_weights 必须是对象", locator="transition_weights"))
    else:
        start = transitions.get("start")
        end = transitions.get("end")
        matrix = transitions.get("matrix")
        if not isinstance(start, list) or len(start) != label_count:
            failures.append(_failure(None, None, Code.FEATURE_DIM_OUT_OF_RANGE, "transition_weights.start 长度必须等于标签数", locator="transition_weights.start"))
        if not isinstance(end, list) or len(end) != label_count:
            failures.append(_failure(None, None, Code.FEATURE_DIM_OUT_OF_RANGE, "transition_weights.end 长度必须等于标签数", locator="transition_weights.end"))
        if not isinstance(matrix, list) or len(matrix) != label_count or any(not isinstance(row, list) or len(row) != label_count for row in matrix):
            failures.append(_failure(None, None, Code.FEATURE_DIM_OUT_OF_RANGE, "transition_weights.matrix 必须是标签数见方矩阵", locator="transition_weights.matrix"))

    return failures


def load_package(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    locator = str(source) if isinstance(source, (str, Path)) else "<dict>"
    if isinstance(source, dict):
        data = source
    else:
        try:
            data = load_json_lenient(source)
        except NonFiniteConstant as exc:
            return {"ok": False, "package": None, "failures": [_failure(None, None, Code.NON_FINITE_WEIGHT, str(exc), locator=locator)]}
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"ok": False, "package": None, "failures": [_failure(None, None, Code.PACKAGE_TRUNCATED, f"参数包缺失、截断或 JSON 非法：{exc}", locator=locator)]}
    failures = validate_package(data)
    return {"ok": not failures, "package": data if not failures else None, "failures": failures}


def parse_command_stream(source: str | Path | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        data = load_json_lenient(source)
    else:
        data = source
    if isinstance(data, dict) and "commands" in data:
        commands = data["commands"]
    else:
        commands = data
    if not isinstance(commands, list):
        raise ValueError("command stream must be a list or {'commands': [...]}")
    return commands


KNOWN_INJECTION_KINDS = {
    "truncate_package",
    "bad_tag_length",
    "unknown_tag",
    "duplicate_id",
    "invalid_bieso",
    "feature_dim_invalid",
    "feature_dim_out_of_range",
    "non_finite_weight",
}


def load_command_stream(source: str | Path | dict[str, Any] | list[Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    try:
        commands = parse_command_stream(source)
    except (OSError, json.JSONDecodeError, ValueError, NonFiniteConstant) as exc:
        return {
            "commands": [],
            "failures": [_failure(None, None, Code.UNKNOWN_COMMAND, f"命令流无法加载：{exc}", locator=str(source))],
        }

    normalized: list[Any] = []
    for index, command in enumerate(commands):
        normalized.append(command)
        if not isinstance(command, dict) or not isinstance(command.get("op"), str) or not command["op"]:
            failures.append(
                _failure(
                    None,
                    None,
                    Code.UNKNOWN_COMMAND,
                    "命令必须是包含非空字符串 op 的对象",
                    locator=f"commands[{index}]",
                )
            )
    return {"commands": normalized, "failures": failures}


def load_injection_manifest(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    try:
        data = load_json_lenient(source) if isinstance(source, (str, Path)) else source
    except (OSError, json.JSONDecodeError, NonFiniteConstant) as exc:
        return {
            "injections": [],
            "failures": [_failure(None, None, Code.COMMAND_VALIDATION, f"注入清单无法加载：{exc}", locator=str(source))],
        }
    injections = data.get("injections", data) if isinstance(data, dict) else data
    if not isinstance(injections, list):
        return {
            "injections": [],
            "failures": [_failure(None, None, Code.COMMAND_VALIDATION, "注入清单必须是列表或包含 injections 列表")],
        }
    failures: list[dict[str, Any]] = []
    for index, injection in enumerate(injections):
        locator = f"injections[{index}]"
        if not isinstance(injection, dict):
            failures.append(_failure(None, None, Code.COMMAND_VALIDATION, "注入项必须是对象", locator=locator))
            continue
        if not isinstance(injection.get("id"), str) or not injection["id"]:
            failures.append(_failure(None, None, Code.COMMAND_VALIDATION, "注入项缺少字符串 id", locator=locator))
        if injection.get("kind") not in KNOWN_INJECTION_KINDS:
            failures.append(
                _failure(
                    None,
                    injection.get("id") if isinstance(injection.get("id"), str) else None,
                    Code.COMMAND_VALIDATION,
                    "注入 kind 未登记",
                    locator=locator,
                    kind=injection.get("kind"),
                )
            )
    return {"injections": injections, "failures": failures}
