from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .biz import (
    COMMAND_SCHEMA_INVALID,
    DUPLICATE_TRACK_VERSION,
    INVALID_BUCKET_CAPACITY,
    INVALID_PEAK_THRESHOLD,
    INVALID_TIME_GEAR,
    MANIFEST_SCHEMA_INVALID,
    SAMPLE_RATE_NOT_DECLARED,
    BIT_DEPTH_NOT_DECLARED,
    issue_dict,
)
from .canonical import fingerprint
from .config import TIME_GEAR_STEPS, is_positive_int, parse_config
from .wavio import SUPPORTED_BIT_DEPTHS, parse_wav_container


def load_json_file(path: str | Path) -> tuple[Any, list[dict[str, Any]]]:
    path = Path(path)
    if not path.exists():
        return None, [issue_dict(MANIFEST_SCHEMA_INVALID, str(path), "JSON file does not exist")]
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), []
    except json.JSONDecodeError as exc:
        return None, [
            issue_dict(
                MANIFEST_SCHEMA_INVALID,
                str(path),
                "JSON parse failed",
                line=exc.lineno,
                column=exc.colno,
                message=exc.msg,
            )
        ]


def format_fingerprint(wav_info: dict[str, Any]) -> str:
    return fingerprint(
        {
            "container": wav_info.get("format", "pcm_wav"),
            "audio_format": 1,
            "channels": wav_info.get("channels"),
            "sample_rate": wav_info.get("sample_rate"),
            "bit_depth": wav_info.get("bit_depth"),
            "bytes_per_sample": wav_info.get("bytes_per_sample"),
            "block_align": wav_info.get("block_align"),
        },
        "fmt_",
    )


def _required_string(spec: dict[str, Any], key: str, index: int, issues: list[dict[str, Any]]) -> bool:
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        issues.append(
            issue_dict(
                MANIFEST_SCHEMA_INVALID,
                f"clips[{index}].{key}",
                f"{key} must be a non-empty string",
                value=value,
            )
        )
        return False
    return True


def validate_manifest(
    manifest: dict[str, Any] | None,
    data_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    valid_clips: list[dict[str, Any]] = []
    if not isinstance(manifest, dict):
        return [], [issue_dict(MANIFEST_SCHEMA_INVALID, "manifest", "manifest must be a JSON object")]
    clips = manifest.get("clips")
    if not isinstance(clips, list):
        return [], [issue_dict(MANIFEST_SCHEMA_INVALID, "manifest.clips", "clips must be a list")]

    seen_pairs: set[tuple[str, int]] = set()
    seen_clip_ids: set[str] = set()
    data_dir = Path(data_dir)

    for index, spec in enumerate(clips):
        target = f"clips[{index}]"
        clip_issues_before = len(issues)
        if not isinstance(spec, dict):
            issues.append(issue_dict(MANIFEST_SCHEMA_INVALID, target, "clip entry must be an object"))
            continue

        clip_id = spec.get("clip_id")
        if _required_string(spec, "clip_id", index, issues):
            target = f"clip:{clip_id}"
            if clip_id in seen_clip_ids:
                issues.append(issue_dict(MANIFEST_SCHEMA_INVALID, target, "duplicate clip_id", clip_id=clip_id))
            seen_clip_ids.add(clip_id)

        track_id = spec.get("track_id")
        _required_string(spec, "track_id", index, issues)
        version = spec.get("version")
        if not is_positive_int(version):
            issues.append(
                issue_dict(MANIFEST_SCHEMA_INVALID, f"{target}.version", "version must be a positive integer", value=version)
            )

        path_value = spec.get("path")
        if not isinstance(path_value, str) or not path_value:
            issues.append(issue_dict(MANIFEST_SCHEMA_INVALID, f"{target}.path", "path must be a non-empty string"))
            path_value = f"{clip_id}.wav" if isinstance(clip_id, str) else None

        declared_rates = spec.get("declared_sample_rates")
        if not isinstance(declared_rates, list) or not all(is_positive_int(v) for v in declared_rates):
            issues.append(
                issue_dict(
                    MANIFEST_SCHEMA_INVALID,
                    f"{target}.declared_sample_rates",
                    "declared_sample_rates must be a list of positive integers",
                    value=declared_rates,
                )
            )
            declared_rates = []

        declared_depths = spec.get("declared_bit_depths")
        if not isinstance(declared_depths, list) or not all(v in SUPPORTED_BIT_DEPTHS for v in declared_depths):
            issues.append(
                issue_dict(
                    MANIFEST_SCHEMA_INVALID,
                    f"{target}.declared_bit_depths",
                    "declared_bit_depths must be 8/16/24/32",
                    value=declared_depths,
                )
            )
            declared_depths = []

        source_rate = spec.get("source_sample_rate")
        if not is_positive_int(source_rate):
            issues.append(
                issue_dict(
                    MANIFEST_SCHEMA_INVALID,
                    f"{target}.source_sample_rate",
                    "source_sample_rate must be a positive integer",
                    value=source_rate,
                )
            )
        bit_depth = spec.get("bit_depth")
        if bit_depth not in SUPPORTED_BIT_DEPTHS:
            issues.append(
                issue_dict(
                    MANIFEST_SCHEMA_INVALID,
                    f"{target}.bit_depth",
                    "bit_depth must be 8/16/24/32",
                    value=bit_depth,
                )
            )

        if isinstance(track_id, str) and is_positive_int(version):
            pair = (track_id, int(version))
            if pair in seen_pairs:
                issues.append(
                    issue_dict(
                        DUPLICATE_TRACK_VERSION,
                        target,
                        "track_id and version must be unique",
                        track_id=track_id,
                        version=version,
                    )
                )
            seen_pairs.add(pair)

        if issues[clip_issues_before:]:
            continue

        assert path_value is not None
        wav_path = data_dir / path_value
        wav_info, wav_issues = parse_wav_container(wav_path)
        for item in wav_issues:
            item["target"] = target
            item.setdefault("detail", {})["relative_path"] = path_value
        issues.extend(wav_issues)
        if wav_issues or wav_info is None:
            continue

        if wav_info["sample_rate"] not in declared_rates:
            issues.append(
                issue_dict(
                    SAMPLE_RATE_NOT_DECLARED,
                    target,
                    "WAV sample rate is not in declared_sample_rates",
                    actual=wav_info["sample_rate"],
                    declared=declared_rates,
                )
            )
        if wav_info["bit_depth"] not in declared_depths:
            issues.append(
                issue_dict(
                    BIT_DEPTH_NOT_DECLARED,
                    target,
                    "WAV bit depth is not in declared_bit_depths",
                    actual=wav_info["bit_depth"],
                    declared=declared_depths,
                )
            )

        expected_samples = spec.get("declared_source_samples")
        if expected_samples is not None and expected_samples != wav_info["sample_count"]:
            issues.append(
                issue_dict(
                    "E1005",
                    target,
                    "declared source sample count differs from data chunk",
                    declared=expected_samples,
                    actual=wav_info["sample_count"],
                )
            )

        clip_has_conflict = any(item["target"] == target for item in issues)
        if not clip_has_conflict:
            record = dict(spec)
            record["path"] = path_value
            record["wav"] = wav_info
            record["format_fingerprint"] = format_fingerprint(wav_info)
            record["source_sample_rate"] = wav_info["sample_rate"]
            record["bit_depth"] = wav_info["bit_depth"]
            record["source_sample_count"] = wav_info["sample_count"]
            record.setdefault("role", "library")
            valid_clips.append(record)

    valid_clips.sort(key=lambda c: (str(c.get("track_id")), int(c.get("version")), str(c.get("clip_id"))))
    return valid_clips, issues


def _command_issue(index: int, code: str, message: str, **detail: Any) -> dict[str, Any]:
    return issue_dict(code, f"commands[{index}]", message, command_index=index, **detail)


def validate_commands(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(payload, dict):
        commands = payload.get("commands")
    else:
        commands = payload
    if not isinstance(commands, list):
        return [], [issue_dict(COMMAND_SCHEMA_INVALID, "commands", "command stream must be a list or {commands: []}")]

    valid: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    known_ops = {
        "replace_clip",
        "delete_clip",
        "add_clip",
        "set_peak_threshold",
        "set_bucket_capacity",
        "set_time_quant_gear",
        "set_spectrum_quant_gear",
        "set_frame_threshold",
        "set_clip_format",
        "set_config",
    }

    for index, raw_command in enumerate(commands):
        if not isinstance(raw_command, dict):
            issues.append(_command_issue(index, COMMAND_SCHEMA_INVALID, "command must be an object"))
            continue
        op = raw_command.get("op")
        command = dict(raw_command)
        command["command_id"] = f"cmd-{index:04d}"
        ok = True

        if op not in known_ops:
            issues.append(_command_issue(index, COMMAND_SCHEMA_INVALID, "unknown command op", op=op))
            continue

        if op in {"replace_clip", "delete_clip", "set_frame_threshold", "set_clip_format"}:
            if not isinstance(raw_command.get("clip_id"), str) or not raw_command["clip_id"]:
                issues.append(_command_issue(index, COMMAND_SCHEMA_INVALID, "clip_id is required", op=op))
                ok = False

        if op == "replace_clip":
            if not is_positive_int(raw_command.get("seed")):
                issues.append(_command_issue(index, COMMAND_SCHEMA_INVALID, "replace seed must be positive integer"))
                ok = False

        elif op == "add_clip":
            clip = raw_command.get("clip")
            if not isinstance(clip, dict) or not isinstance(clip.get("clip_id"), str):
                issues.append(_command_issue(index, COMMAND_SCHEMA_INVALID, "add_clip requires a clip object"))
                ok = False

        elif op == "set_peak_threshold":
            if not is_positive_int(raw_command.get("value")):
                issues.append(
                    _command_issue(index, INVALID_PEAK_THRESHOLD, "peak threshold must be a positive integer", value=raw_command.get("value"))
                )
                ok = False

        elif op == "set_bucket_capacity":
            if not is_positive_int(raw_command.get("value")):
                issues.append(
                    _command_issue(index, INVALID_BUCKET_CAPACITY, "bucket capacity must be a positive integer", value=raw_command.get("value"))
                )
                ok = False

        elif op == "set_time_quant_gear":
            if raw_command.get("value") not in TIME_GEAR_STEPS:
                issues.append(
                    _command_issue(
                        index,
                        INVALID_TIME_GEAR,
                        "time quantization gear is not allowed",
                        value=raw_command.get("value"),
                        allowed=sorted(TIME_GEAR_STEPS),
                    )
                )
                ok = False

        elif op == "set_spectrum_quant_gear":
            from .config import SPECTRUM_GEAR_SHIFTS

            if raw_command.get("value") not in SPECTRUM_GEAR_SHIFTS:
                issues.append(_command_issue(index, "E3005", "spectrum quantization gear is not allowed"))
                ok = False

        elif op == "set_frame_threshold":
            frame_index = raw_command.get("frame_index")
            value = raw_command.get("value")
            if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
                issues.append(_command_issue(index, COMMAND_SCHEMA_INVALID, "frame_index must be a non-negative integer"))
                ok = False
            if value is not None and not is_positive_int(value):
                issues.append(_command_issue(index, INVALID_PEAK_THRESHOLD, "frame threshold must be positive integer or null"))
                ok = False

        elif op == "set_clip_format":
            if not is_positive_int(raw_command.get("sample_rate")) or raw_command.get("bit_depth") not in SUPPORTED_BIT_DEPTHS:
                issues.append(_command_issue(index, COMMAND_SCHEMA_INVALID, "invalid target sample_rate or bit_depth"))
                ok = False

        elif op == "set_config":
            _, config_issues = parse_config(raw_command.get("config"))
            if config_issues:
                issues.extend(
                    issue_dict(
                        item["code"],
                        f"commands[{index}].{item['target']}",
                        item["message"],
                        **{k: v for k, v in item.items() if k not in {"code", "target", "message", "detail"}},
                    )
                    for item in config_issues
                )
                ok = False

        if ok:
            valid.append(command)

    return valid, issues
