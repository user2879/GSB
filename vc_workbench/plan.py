"""Conversion-session command stream and mutable effective parameters."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import WorkbenchConfig
from .errors import E_COMMAND_INVALID
from .util import canonical_json, q_from_float, sha256_value


class CommandError(ValueError):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.code = E_COMMAND_INVALID
        self.detail = detail


def _parse_int_like(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise CommandError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    raise CommandError(f"{name} must be an integer")


def command_sequence(command: dict[str, Any]) -> Any:
    return command.get("seq", command.get("command_seq", command.get("sequence")))


def _normalise_relative_source(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CommandError("segment path must be a non-empty string")
    value = value.strip().replace("\\", "/")
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise CommandError("segment path must be a relative path without '..'")
    if not value.startswith("sources/"):
        value = "sources/" + value.lstrip("./")
    return value


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    segment_id: str
    path: str
    start_sample: int = 0
    end_sample: int | None = None

    @classmethod
    def from_raw(cls, raw: Any) -> "SegmentSpec":
        if not isinstance(raw, dict):
            raise CommandError("segment definition must be an object")
        segment_id = raw.get("segment_id", raw.get("id"))
        if not isinstance(segment_id, str) or not segment_id:
            raise CommandError("segment id must be a non-empty string")
        path = _normalise_relative_source(raw.get("path"))
        start = _parse_int_like(raw.get("start_sample", 0), "start_sample")
        end_raw = raw.get("end_sample")
        end = None if end_raw is None else _parse_int_like(end_raw, "end_sample")
        if start < 0 or (end is not None and end <= start):
            raise CommandError(f"invalid sample range for segment {segment_id}")
        return cls(segment_id=segment_id, path=path, start_sample=start, end_sample=end)

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "path": self.path,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
        }


@dataclass(slots=True)
class SessionPlan:
    base_dir: Path
    defaults: dict[str, Any]
    initial_segments: list[SegmentSpec]
    commands: list[dict[str, Any]]
    fault_injection: list[dict[str, Any]]
    format_version: int
    source_path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> "SessionPlan":
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CommandError(f"cannot read command stream {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise CommandError("command stream root must be an object")
        commands_raw = raw.get("commands", [])
        if not isinstance(commands_raw, list):
            raise CommandError("commands must be an array")
        initial_raw = raw.get("initial_segments", raw.get("segments", []))
        if not isinstance(initial_raw, list) or not initial_raw:
            raise CommandError("initial_segments must be a non-empty array")
        segments = [SegmentSpec.from_raw(item) for item in initial_raw]
        ids = [segment.segment_id for segment in segments]
        if len(ids) != len(set(ids)):
            raise CommandError("initial segment ids must be unique")
        defaults_raw = raw.get("defaults", {})
        if not isinstance(defaults_raw, dict):
            raise CommandError("defaults must be an object")
        defaults = {
            "target_speaker": defaults_raw.get("target_speaker", raw.get("target_speaker", "target_a")),
            "pitch_offset_semitones": defaults_raw.get("pitch_offset_semitones", 0.0),
            "timbre_strength": defaults_raw.get("timbre_strength", 0.75),
            "formant_warp": defaults_raw.get("formant_warp", 1.0),
            "smoothing": defaults_raw.get("smoothing", 0.2),
        }
        if not isinstance(defaults["target_speaker"], str) or not defaults["target_speaker"]:
            raise CommandError("target_speaker must be a non-empty string")
        parsed_commands: list[dict[str, Any]] = []
        expected_seq = 1
        for item in commands_raw:
            if not isinstance(item, dict):
                raise CommandError("each command must be an object")
            parsed_commands.append(copy.deepcopy(item))
            sequence = command_sequence(item)
            if isinstance(sequence, int) and not isinstance(sequence, bool):
                expected_seq = sequence + 1
        fault_injection = raw.get("fault_injection", [])
        if not isinstance(fault_injection, list):
            raise CommandError("fault_injection must be an array")
        return cls(
            base_dir=path.parent.resolve(),
            defaults=defaults,
            initial_segments=segments,
            commands=parsed_commands,
            fault_injection=copy.deepcopy(fault_injection),
            format_version=int(raw.get("format_version", 1)),
            source_path=path.resolve(),
        )

    @property
    def target_speaker(self) -> str:
        return str(self.defaults["target_speaker"])

    def command_stream_hash(self) -> str:
        return sha256_value(
            {
                "format_version": self.format_version,
                "defaults": self.defaults,
                "initial_segments": [segment.as_dict() for segment in self.initial_segments],
                "commands": self.commands,
                "fault_injection": self.fault_injection,
            }
        )

    def command_prefix_hash(self, count: int) -> str:
        return sha256_value(
            {
                "format_version": self.format_version,
                "defaults": self.defaults,
                "initial_segments": [segment.as_dict() for segment in self.initial_segments],
                "commands": self.commands[:count],
            }
        )

    def fault_for(self, segment_id: str, stage: str) -> dict[str, Any] | None:
        for item in self.fault_injection:
            if item.get("segment_id") == segment_id and item.get("stage") == stage:
                return item
        return None


def initial_params(plan: SessionPlan, cfg: WorkbenchConfig) -> dict[str, Any]:
    scale = cfg.scale
    return {
        "target_speaker": plan.target_speaker,
        "pitch_offset_semitones_q": q_from_float(float(plan.defaults["pitch_offset_semitones"]), scale),
        "timbre_strength_q": q_from_float(float(plan.defaults["timbre_strength"]), scale),
        "formant_warp_q": q_from_float(float(plan.defaults["formant_warp"]), scale),
        "smoothing_q": q_from_float(float(plan.defaults["smoothing"]), scale),
        "overrides": {},
        "scale": scale,
    }


def _float_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandError(f"{name} must be numeric")
    return float(value)


def _validate_param_value(name: str, value: float) -> None:
    if name == "pitch_offset_semitones" and not -36.0 <= value <= 36.0:
        raise CommandError("pitch_offset_semitones must be in [-36, 36]")
    if name == "timbre_strength" and not 0.0 <= value <= 1.0:
        raise CommandError("timbre_strength must be in [0, 1]")
    if name == "formant_warp" and not 0.5 <= value <= 2.0:
        raise CommandError("formant_warp must be in [0.5, 2.0]")
    if name == "smoothing" and not 0.0 <= value <= 0.49:
        raise CommandError("smoothing must be in [0, 0.49]")


def _param_patch(raw: dict[str, Any], cfg: WorkbenchConfig) -> dict[str, int]:
    mapping = {
        "pitch_offset_semitones": ("pitch_offset_semitones_q", None),
        "timbre_strength": ("timbre_strength_q", None),
        "formant_warp": ("formant_warp_q", None),
        "smoothing": ("smoothing_q", None),
    }
    patch: dict[str, int] = {}
    aliases = {
        "pitch_offset": "pitch_offset_semitones",
        "pitch_shift_semitones": "pitch_offset_semitones",
        "pitch_offset_hz": "pitch_offset_semitones",
        "timbre": "timbre_strength",
        "formant_curvature": "formant_warp",
        "formant_curve": "formant_warp",
        "spectral_smoothing": "smoothing",
        "smooth": "smoothing",
    }
    raw = dict(raw)
    for alias, canonical in aliases.items():
        if alias in raw and canonical not in raw:
            raw[canonical] = raw[alias]
    for name, (q_name, _) in mapping.items():
        if name in raw:
            value = _float_value(raw[name], name)
            _validate_param_value(name, value)
            patch[q_name] = q_from_float(value, cfg.scale)
        q_value_name = name + "_q"
        if q_value_name in raw:
            q_value = _parse_int_like(raw[q_value_name], q_value_name)
            _validate_param_value(name, q_value / cfg.scale)
            patch[q_name] = q_value
    return patch


def apply_command(
    params: dict[str, Any],
    segments: list[SegmentSpec],
    command: dict[str, Any],
    cfg: WorkbenchConfig,
) -> tuple[dict[str, Any], list[SegmentSpec]]:
    """Apply one already sequence-checked command to a copy of the session."""

    new_params = copy.deepcopy(params)
    new_segments = list(segments)
    op = command.get("op", command.get("type"))
    if not isinstance(op, str) or not op:
        raise CommandError("command op must be a non-empty string")
    op = {
        "append": "append_segment",
        "replace": "replace_segment",
        "set_target_speaker": "set_target",
        "update_params": "set_params",
        "set_pitch": "set_pitch_offset",
        "set_formant_curvature": "set_formant_warp",
        "set_spectral_smoothing": "set_smoothing",
    }.get(op, op)
    known = {
        "set_target",
        "set_params",
        "set_pitch_offset",
        "set_timbre_strength",
        "set_formant_warp",
        "set_smoothing",
        "override_segment",
        "set_segment_params",
        "clear_segment_override",
        "append_segment",
        "replace_segment",
        "noop",
    }
    if op not in known:
        raise CommandError(f"unknown command op: {op}")

    if op == "set_target":
        target = command.get("target_speaker", command.get("target"))
        if not isinstance(target, str) or not target:
            raise CommandError("set_target requires target_speaker")
        new_params["target_speaker"] = target
        return new_params, new_segments

    if op in {"set_params", "set_pitch_offset", "set_timbre_strength", "set_formant_warp", "set_smoothing"}:
        if op == "set_params":
            patch = _param_patch(command, cfg)
            target = command.get("target_speaker", command.get("target"))
            if target is not None:
                if not isinstance(target, str) or not target:
                    raise CommandError("set_params target_speaker must be a non-empty string")
                new_params["target_speaker"] = target
        else:
            name = {
                "set_pitch_offset": "pitch_offset_semitones",
                "set_timbre_strength": "timbre_strength",
                "set_formant_warp": "formant_warp",
                "set_smoothing": "smoothing",
            }[op]
            value_raw = command.get("value", command.get(name))
            patch = _param_patch({name: value_raw}, cfg)
        new_params.update(patch)
        return new_params, new_segments

    if op in {"override_segment", "set_segment_params"}:
        segment_id = command.get("segment_id", command.get("id"))
        if not isinstance(segment_id, str) or not any(s.segment_id == segment_id for s in new_segments):
            raise CommandError(f"unknown segment_id: {segment_id!r}")
        patch = _param_patch(command, cfg)
        if not patch:
            raise CommandError("segment parameter command has no parameters")
        override = dict(new_params.get("overrides", {}).get(segment_id, {}))
        override.update(patch)
        new_params.setdefault("overrides", {})[segment_id] = dict(sorted(override.items()))
        return new_params, new_segments

    if op == "clear_segment_override":
        segment_id = command.get("segment_id", command.get("id"))
        if not isinstance(segment_id, str):
            raise CommandError("clear_segment_override requires segment_id")
        new_params.setdefault("overrides", {}).pop(segment_id, None)
        return new_params, new_segments

    if op == "append_segment":
        raw = command.get("segment", command)
        if "path" not in raw:
            raw = dict(raw)
            raw["path"] = raw.get("audio_path", raw.get("source_path"))
        segment = SegmentSpec.from_raw(raw)
        if any(existing.segment_id == segment.segment_id for existing in new_segments):
            raise CommandError(f"duplicate segment id: {segment.segment_id}")
        new_segments.append(segment)
        return new_params, new_segments

    if op == "replace_segment":
        segment_id = command.get("segment_id", command.get("id"))
        index = next((i for i, segment in enumerate(new_segments) if segment.segment_id == segment_id), None)
        if index is None:
            raise CommandError(f"unknown segment_id: {segment_id!r}")
        raw = command.get("segment", command)
        if "path" not in raw:
            raw = dict(raw)
            raw["path"] = raw.get("audio_path", raw.get("source_path"))
        replacement = SegmentSpec.from_raw(
            {
                "segment_id": segment_id,
                "path": raw.get("path"),
                "start_sample": raw.get("start_sample", 0),
                "end_sample": raw.get("end_sample"),
            }
        )
        new_segments[index] = replacement
        return new_params, new_segments

    return new_params, new_segments


def effective_params(params: dict[str, Any], segment_id: str) -> dict[str, int | str]:
    result: dict[str, int | str] = {
        "target_speaker": str(params["target_speaker"]),
        "pitch_offset_semitones_q": int(params["pitch_offset_semitones_q"]),
        "timbre_strength_q": int(params["timbre_strength_q"]),
        "formant_warp_q": int(params["formant_warp_q"]),
        "smoothing_q": int(params["smoothing_q"]),
    }
    override = params.get("overrides", {}).get(segment_id, {})
    for key, value in sorted(override.items()):
        result[key] = int(value)
    return result


def params_public(params: dict[str, Any], cfg: WorkbenchConfig) -> dict[str, Any]:
    return {
        "target_speaker": params["target_speaker"],
        "pitch_offset_semitones_q": int(params["pitch_offset_semitones_q"]),
        "timbre_strength_q": int(params["timbre_strength_q"]),
        "formant_warp_q": int(params["formant_warp_q"]),
        "smoothing_q": int(params["smoothing_q"]),
        "overrides": {
            segment_id: dict(sorted(override.items()))
            for segment_id, override in sorted(params.get("overrides", {}).items())
        },
        "scale": cfg.scale,
    }


def command_public(command: dict[str, Any]) -> str:
    return canonical_json(command)
