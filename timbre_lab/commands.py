from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio import AudioFile, read_wav_strict, validate_samples
from .config import AnalysisConfig
from .errors import E_COMMAND_INVALID, BusinessError
from .model import AudioDocument, Segment, find_segment, make_segment_id


@dataclass
class CommandImpact:
    command_id: str
    op: str
    affected_segments: set[str] = field(default_factory=set)
    affected_stages: set[str] = field(default_factory=set)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "op": self.op,
            "affected_segments": sorted(self.affected_segments),
            "affected_stages": sorted(self.affected_stages),
            "details": self.details,
        }


def _refresh(document: AudioDocument, config: AnalysisConfig) -> None:
    document.segments.sort(key=lambda segment: segment.stable_key())
    document.refresh_frame_indices(config)


def _require_selector(command: dict[str, Any]) -> dict[str, Any]:
    selector = command.get("target", command.get("selector"))
    if not isinstance(selector, dict):
        raise BusinessError(E_COMMAND_INVALID, "command requires a target selector object", command=command.get("id"))
    return selector


def _command_number(value: Any, field_name: str, command: dict[str, Any]) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BusinessError(E_COMMAND_INVALID, f"{field_name} must be numeric", command=command.get("id")) from exc
    if not math.isfinite(number):
        raise BusinessError(E_COMMAND_INVALID, f"{field_name} must be finite", command=command.get("id"))
    return number


def _split_segment(
    document: AudioDocument,
    config: AnalysisConfig,
    command: dict[str, Any],
) -> CommandImpact:
    selector = _require_selector(command)
    try:
        target = find_segment(document, selector)
    except KeyError as exc:
        raise BusinessError(E_COMMAND_INVALID, str(exc), command=command.get("id")) from exc
    at_ms = command.get("at_ms", command.get("split_ms"))
    if at_ms is None:
        raise BusinessError(E_COMMAND_INVALID, "split requires at_ms", command=command.get("id"))
    split_value = _command_number(at_ms, "at_ms", command)
    split_sample = int(round(split_value * config.sample_rate / 1000.0))
    hop = config.frame_shift_samples
    split_sample = int(round(split_sample / hop)) * hop
    minimum_samples = max(config.frame_length_samples, int(math.ceil(config.min_segment_ms * config.sample_rate / 1000.0)))
    if split_sample - target.start_sample < minimum_samples or target.end_sample - split_sample < minimum_samples:
        raise BusinessError(
            E_COMMAND_INVALID,
            "split point leaves a segment shorter than the configured minimum",
            command=command.get("id"),
            segment_id=target.segment_id,
            at_ms=split_value,
            minimum_ms=config.min_segment_ms,
        )
    left = Segment(
        segment_id=make_segment_id(target.file, target.start_sample, split_sample),
        file=target.file,
        start_sample=target.start_sample,
        end_sample=split_sample,
        start_frame=0,
        end_frame=0,
        speaker=target.speaker,
        style_hint=target.style_hint,
        origin=f"split:{command.get('id', '')}",
    )
    right = Segment(
        segment_id=make_segment_id(target.file, split_sample, target.end_sample),
        file=target.file,
        start_sample=split_sample,
        end_sample=target.end_sample,
        start_frame=0,
        end_frame=0,
        speaker=target.speaker,
        style_hint=target.style_hint,
        origin=f"split:{command.get('id', '')}",
    )
    document.segments = [segment for segment in document.segments if segment.segment_id != target.segment_id]
    document.segments.extend([left, right])
    _refresh(document, config)
    return CommandImpact(
        command_id=str(command.get("id", "")),
        op="split",
        affected_segments={target.segment_id, left.segment_id, right.segment_id},
        affected_stages={"feature", "description", "assignment", "speaker"},
        details={"source_segment": target.segment_id, "left": left.segment_id, "right": right.segment_id, "at_ms": split_value},
    )


def _merge_segments(
    document: AudioDocument,
    config: AnalysisConfig,
    command: dict[str, Any],
) -> CommandImpact:
    selector = _require_selector(command)
    try:
        target = find_segment(document, selector)
    except KeyError as exc:
        raise BusinessError(E_COMMAND_INVALID, str(exc), command=command.get("id")) from exc
    direction = str(command.get("direction", "next"))
    same_file = document.segments_for_file(target.file)
    index = next(index for index, segment in enumerate(same_file) if segment.segment_id == target.segment_id)
    if direction == "next":
        if index + 1 >= len(same_file):
            raise BusinessError(E_COMMAND_INVALID, "no adjacent segment after target", command=command.get("id"))
        left, right = target, same_file[index + 1]
    elif direction == "previous":
        if index == 0:
            raise BusinessError(E_COMMAND_INVALID, "no adjacent segment before target", command=command.get("id"))
        left, right = same_file[index - 1], target
    else:
        raise BusinessError(E_COMMAND_INVALID, "merge direction must be next or previous", command=command.get("id"))
    if left.end_sample > right.start_sample:
        raise BusinessError(E_COMMAND_INVALID, "segments overlap and cannot be merged", command=command.get("id"))
    explicit_speaker = command.get("speaker")
    label_conflict = left.speaker != right.speaker
    left_duration = left.end_sample - left.start_sample
    right_duration = right.end_sample - right.start_sample
    if explicit_speaker is not None:
        speaker = str(explicit_speaker)
        label_resolution = "command"
    elif left.speaker == right.speaker:
        speaker = left.speaker
        label_resolution = "same_label"
    else:
        speaker = left.speaker if left_duration >= right_duration else right.speaker
        label_resolution = "longer_duration_then_earlier"
    style_hint = left.style_hint if left_duration >= right_duration else right.style_hint
    merged = Segment(
        segment_id=make_segment_id(left.file, left.start_sample, right.end_sample),
        file=left.file,
        start_sample=left.start_sample,
        end_sample=right.end_sample,
        start_frame=0,
        end_frame=0,
        speaker=speaker,
        style_hint=style_hint,
        origin=f"merge:{command.get('id', '')}",
    )
    old_ids = {left.segment_id, right.segment_id}
    document.segments = [segment for segment in document.segments if segment.segment_id not in old_ids]
    document.segments.append(merged)
    _refresh(document, config)
    return CommandImpact(
        command_id=str(command.get("id", "")),
        op="merge",
        affected_segments=old_ids | {merged.segment_id},
        affected_stages={"feature", "description", "assignment", "speaker"},
        details={
            "merged_segment": merged.segment_id,
            "inputs": sorted(old_ids),
            "label_conflict": label_conflict,
            "label_resolution": label_resolution,
        },
    )


def _set_speaker(
    document: AudioDocument,
    config: AnalysisConfig,
    command: dict[str, Any],
) -> CommandImpact:
    selector = _require_selector(command)
    speaker = command.get("speaker")
    if speaker is None or not str(speaker).strip():
        raise BusinessError(E_COMMAND_INVALID, "set_speaker requires a non-empty speaker", command=command.get("id"))
    all_with_speaker = bool(command.get("all_with_speaker", False))
    if all_with_speaker:
        old_speaker = command.get("old_speaker", speaker)
        targets = [segment for segment in document.segments if segment.speaker == old_speaker]
    else:
        try:
            targets = [find_segment(document, selector)]
        except KeyError as exc:
            raise BusinessError(E_COMMAND_INVALID, str(exc), command=command.get("id")) from exc
    if not targets:
        raise BusinessError(E_COMMAND_INVALID, "set_speaker matched no segments", command=command.get("id"))
    affected = set()
    old_values = {}
    for segment in targets:
        old_values[segment.segment_id] = segment.speaker
        segment.speaker = str(speaker)
        affected.add(segment.segment_id)
    return CommandImpact(
        command_id=str(command.get("id", "")),
        op="set_speaker",
        affected_segments=affected,
        affected_stages={"assignment", "speaker"},
        details={"speaker": str(speaker), "old_values": old_values},
    )


def _replace_interval(
    document: AudioDocument,
    config: AnalysisConfig,
    command: dict[str, Any],
    base_dir: Path,
) -> CommandImpact:
    selector = _require_selector(command)
    try:
        target = find_segment(document, selector)
    except KeyError as exc:
        raise BusinessError(E_COMMAND_INVALID, str(exc), command=command.get("id")) from exc
    interval = command.get("interval", {})
    if not isinstance(interval, dict):
        raise BusinessError(E_COMMAND_INVALID, "replace interval must be an object", command=command.get("id"))
    start_ms = interval.get("start_ms", 0.0)
    end_ms = interval.get("end_ms")
    start_value = _command_number(start_ms, "interval.start_ms", command)
    start_sample = target.start_sample + int(round(start_value * config.sample_rate / 1000.0))
    if end_ms is None:
        end_sample = target.end_sample
    else:
        end_value = _command_number(end_ms, "interval.end_ms", command)
        end_sample = target.start_sample + int(round(end_value * config.sample_rate / 1000.0))
    if not (target.start_sample <= start_sample < end_sample <= target.end_sample):
        raise BusinessError(
            E_COMMAND_INVALID,
            "replacement interval must be non-empty and contained in the target segment",
            command=command.get("id"),
            segment_id=target.segment_id,
        )
    source = command.get("source")
    if source is not None:
        source_path = Path(str(source))
        if not source_path.is_absolute():
            source_path = base_dir / source_path
        try:
            replacement_audio = read_wav_strict(source_path)
            replacement = replacement_audio.samples
        except BusinessError as exc:
            raise BusinessError(E_COMMAND_INVALID, f"replacement WAV is invalid: {exc.message}", command=command.get("id"), source=str(source)) from exc
    elif "samples" in command:
        replacement = validate_samples(command["samples"], name=f"replacement:{command.get('id')}")
    else:
        raise BusinessError(E_COMMAND_INVALID, "replace requires source WAV or inline samples", command=command.get("id"))
    expected = end_sample - start_sample
    if int(replacement.size) != expected:
        raise BusinessError(
            E_COMMAND_INVALID,
            "replacement sample count must exactly match the target interval",
            command=command.get("id"),
            expected_samples=expected,
            replacement_samples=int(replacement.size),
        )
    audio = document.files[target.file]
    samples = audio.samples.copy()
    samples[start_sample:end_sample] = replacement
    document.files[target.file] = AudioFile(audio.name, samples, audio.source_path)
    return CommandImpact(
        command_id=str(command.get("id", "")),
        op="replace",
        affected_segments={target.segment_id},
        affected_stages={"feature", "description", "assignment", "speaker"},
        details={
            "segment_id": target.segment_id,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "replacement_samples": int(replacement.size),
        },
    )


def _set_quantization(
    document: AudioDocument,
    config: AnalysisConfig,
    overrides: dict[str, int],
    command: dict[str, Any],
) -> CommandImpact:
    dimension = str(command.get("dimension", "")).strip()
    bits = command.get("bit_width", command.get("bits"))
    if not dimension or bits is None:
        raise BusinessError(E_COMMAND_INVALID, "set_quantization requires dimension and bit_width", command=command.get("id"))
    try:
        bits_int = int(bits)
    except (TypeError, ValueError) as exc:
        raise BusinessError(E_COMMAND_INVALID, "bit_width must be an integer", command=command.get("id")) from exc
    if dimension == "*":
        for item in config.dimensions:
            config.validate_bit_width(item.name, bits_int)
        for item in config.dimensions:
            overrides[item.name] = bits_int
        names = [item.name for item in config.dimensions]
    else:
        config.validate_bit_width(dimension, bits_int)
        overrides[dimension] = bits_int
        names = [dimension]
    return CommandImpact(
        command_id=str(command.get("id", "")),
        op="set_quantization",
        affected_segments={segment.segment_id for segment in document.segments},
        affected_stages={"description", "assignment", "speaker"},
        details={"dimensions": names, "bit_width": bits_int},
    )


def apply_command(
    document: AudioDocument,
    config: AnalysisConfig,
    overrides: dict[str, int],
    command: dict[str, Any],
    base_dir: str | Path = ".",
) -> CommandImpact:
    if not isinstance(command, dict):
        raise BusinessError(E_COMMAND_INVALID, "each command must be an object")
    op = str(command.get("op", "")).strip()
    backup_document = document.clone()
    backup_overrides = copy.deepcopy(overrides)
    try:
        if op == "split":
            impact = _split_segment(document, config, command)
        elif op == "merge":
            impact = _merge_segments(document, config, command)
        elif op in {"set_speaker", "update_speaker"}:
            impact = _set_speaker(document, config, command)
        elif op in {"replace", "replace_interval"}:
            impact = _replace_interval(document, config, command, Path(base_dir))
        elif op in {"set_quantization", "set_quantization_level"}:
            impact = _set_quantization(document, config, overrides, command)
        else:
            raise BusinessError(E_COMMAND_INVALID, f"unsupported command op: {op!r}", command=command.get("id"))
        validation_failures = document.validate(config)
        if validation_failures:
            raise BusinessError(E_COMMAND_INVALID, "command produced an invalid segment document", command=command.get("id"), failures=validation_failures)
        return impact
    except Exception:
        document.files = backup_document.files
        document.segments = backup_document.segments
        overrides.clear()
        overrides.update(backup_overrides)
        raise
