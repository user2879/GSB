from __future__ import annotations

from pathlib import Path
from typing import Any

from .biz import COMMAND_SCHEMA_INVALID, issue_dict
from .synth import generate_clip_file


def find_clip_index(manifest: dict[str, Any], clip_id: str) -> int | None:
    for index, clip in enumerate(manifest.get("clips", [])):
        if isinstance(clip, dict) and clip.get("clip_id") == clip_id:
            return index
    return None


def apply_command_to_inputs(
    manifest: dict[str, Any],
    config: dict[str, Any],
    command: dict[str, Any],
    data_dir: str | Path,
) -> list[dict[str, Any]]:
    op = command["op"]
    issues: list[dict[str, Any]] = []
    clips = manifest.setdefault("clips", [])

    if op == "add_clip":
        clip = dict(command["clip"])
        clip.setdefault("declared_sample_rates", [11025, 22050, 44100])
        clip.setdefault("declared_bit_depths", [8, 16, 24, 32])
        clip.setdefault("internal_sample_rate", int(config["internal_sample_rate"]))
        clip.setdefault("internal_sample_count", 12288)
        if command.get("generate", True):
            clip = generate_clip_file(clip, data_dir)
        clips.append(clip)
        return issues

    clip_id = command.get("clip_id")
    index = find_clip_index(manifest, clip_id) if clip_id else None
    if op in {"replace_clip", "set_frame_threshold", "set_clip_format", "delete_clip"} and index is None:
        issues.append(issue_dict(COMMAND_SCHEMA_INVALID, f"command:{command.get('command_id')}", "clip_id not found", clip_id=clip_id))
        return issues

    if op == "delete_clip":
        assert index is not None
        removed = clips.pop(index)
        path = Path(data_dir) / removed.get("path", "")
        if str(path).startswith(str(Path(data_dir))):
            path.unlink(missing_ok=True)
        return issues

    if op == "replace_clip":
        assert index is not None
        clip = dict(clips[index])
        clip["seed"] = int(command["seed"])
        clip.pop("content_fingerprint", None)
        clip = generate_clip_file(clip, data_dir)
        clips[index] = clip
        return issues

    if op == "set_clip_format":
        assert index is not None
        clip = dict(clips[index])
        clip["source_sample_rate"] = int(command["sample_rate"])
        clip["bit_depth"] = int(command["bit_depth"])
        clip["internal_sample_rate"] = int(config["internal_sample_rate"])
        clip = generate_clip_file(clip, data_dir)
        clips[index] = clip
        return issues

    if op == "set_peak_threshold":
        config["peak_threshold"] = int(command["value"])
    elif op == "set_bucket_capacity":
        config["bucket_capacity"] = int(command["value"])
    elif op == "set_time_quant_gear":
        config["time_quant_gear"] = command["value"]
        from .config import TIME_GEAR_STEPS

        config["time_quant_step"] = TIME_GEAR_STEPS[command["value"]]
    elif op == "set_spectrum_quant_gear":
        config["spectrum_quant_gear"] = command["value"]
        from .config import SPECTRUM_GEAR_SHIFTS

        config["spectrum_quant_shift"] = SPECTRUM_GEAR_SHIFTS[command["value"]]
    elif op == "set_frame_threshold":
        overrides = config.setdefault("frame_threshold_overrides", {})
        clip_overrides = overrides.setdefault(clip_id, {})
        key = str(int(command["frame_index"]))
        if command.get("value") is None:
            clip_overrides.pop(key, None)
        else:
            clip_overrides[key] = int(command["value"])
    elif op == "set_config":
        from .config import parse_config

        new_config, config_issues = parse_config(command.get("config"))
        if config_issues:
            issues.extend(config_issues)
        else:
            config.clear()
            config.update(new_config)
    return issues
