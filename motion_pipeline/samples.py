"""Deterministic sequence-description JSONL loading and validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .stable import public_failure, sha256_json, sort_failures


ALLOWED_SHAPES = ("circle", "rect")


@dataclass(frozen=True)
class SequenceSpec:
    sequence_id: str
    frames: tuple[dict[str, Any], ...]
    line: int

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def history_frames(self, limit: int, trajectory_limit: int) -> list[dict[str, Any]]:
        return [self._slice_frame(frame, trajectory_limit) for frame in self.frames[:limit]]

    def future_frames(
        self, start: int, count: int, trajectory_limit: int
    ) -> list[dict[str, Any]]:
        end = start + count
        return [self._slice_frame(frame, trajectory_limit) for frame in self.frames[start:end]]

    @staticmethod
    def _slice_frame(frame: dict[str, Any], trajectory_limit: int) -> dict[str, Any]:
        return {
            "trajectories": frame["trajectories"][:trajectory_limit],
            "color_blocks": frame["color_blocks"],
            "noise_seed": frame["noise_seed"],
        }

    def history_content_hash(
        self, *, history_limit: int, trajectory_limit: int
    ) -> str:
        selected = self.history_frames(history_limit, trajectory_limit)
        return sha256_json({"sequence_id": self.sequence_id, "frames": selected})


@dataclass
class SequenceLoadResult:
    sequences: list[SequenceSpec]
    failures: list[dict[str, Any]]
    total_lines: int


def _seq_failure(
    sequence_id: str,
    message: str,
    *,
    line: int | None = None,
    field: str | None = None,
) -> dict[str, Any]:
    return public_failure(
        "E_SEQ_INVALID", sequence_id, message, line=line, field=field
    )


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_color(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 255 for item in value)
    )


def _normalize_trajectory(
    raw: Any, sequence_id: str, line: int, index: int, errors: list[dict[str, Any]]
) -> dict[str, Any] | None:
    field = f"frames[].trajectories[{index}]"
    if not isinstance(raw, dict):
        errors.append(_seq_failure(sequence_id, f"{field} must be a mapping", line=line, field=field))
        return None
    shape = raw.get("shape")
    if shape not in ALLOWED_SHAPES:
        errors.append(
            _seq_failure(
                sequence_id,
                f"{field}.shape must be one of {list(ALLOWED_SHAPES)}",
                line=line,
                field=f"{field}.shape",
            )
        )
        return None
    required = ["x", "y", "color"]
    if shape == "circle":
        required.append("radius")
    elif shape == "rect":
        required.extend(["width", "height"])
    normalized: dict[str, Any] = {"shape": shape}
    for key in required:
        if key == "color":
            if not _valid_color(raw.get(key)):
                errors.append(
                    _seq_failure(
                        sequence_id,
                        f"{field}.color must be [r,g,b] with channels 0..255",
                        line=line,
                        field=f"{field}.color",
                    )
                )
                return None
            normalized[key] = [int(channel) for channel in raw[key]]
        else:
            value = raw.get(key)
            if not _finite(value):
                errors.append(
                    _seq_failure(
                        sequence_id,
                        f"{field}.{key} must be a finite number",
                        line=line,
                        field=f"{field}.{key}",
                    )
                )
                return None
            normalized[key] = float(value)
    for optional in ("vx", "vy", "opacity"):
        if optional in raw:
            if not _finite(raw[optional]):
                errors.append(
                    _seq_failure(
                        sequence_id,
                        f"{field}.{optional} must be a finite number",
                        line=line,
                        field=f"{field}.{optional}",
                    )
                )
                return None
            normalized[optional] = float(raw[optional])
    if "trajectory_id" in raw:
        if not isinstance(raw["trajectory_id"], str) or not raw["trajectory_id"]:
            errors.append(
                _seq_failure(
                    sequence_id,
                    f"{field}.trajectory_id must be a non-empty string",
                    line=line,
                    field=f"{field}.trajectory_id",
                )
            )
            return None
        normalized["trajectory_id"] = raw["trajectory_id"]
    return normalized


def _normalize_block(
    raw: Any, sequence_id: str, line: int, index: int, errors: list[dict[str, Any]]
) -> dict[str, Any] | None:
    field = f"frames[].color_blocks[{index}]"
    if not isinstance(raw, dict):
        errors.append(_seq_failure(sequence_id, f"{field} must be a mapping", line=line, field=field))
        return None
    normalized: dict[str, Any] = {}
    for key in ("x", "y", "width", "height"):
        value = raw.get(key)
        if not _finite(value):
            errors.append(
                _seq_failure(
                    sequence_id,
                    f"{field}.{key} must be a finite number",
                    line=line,
                    field=f"{field}.{key}",
                )
            )
            return None
        normalized[key] = float(value)
    if not _valid_color(raw.get("color")):
        errors.append(
            _seq_failure(
                sequence_id,
                f"{field}.color must be [r,g,b] with channels 0..255",
                line=line,
                field=f"{field}.color",
            )
        )
        return None
    normalized["color"] = [int(channel) for channel in raw["color"]]
    return normalized


def _normalize_sequence(
    raw: Any, sequence_id: str, line: int
) -> tuple[SequenceSpec | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return None, [_seq_failure(sequence_id, "sequence line must be a JSON object", line=line)]
    frames_raw = raw.get("frames")
    if not isinstance(frames_raw, list) or not frames_raw:
        return None, [
            _seq_failure(sequence_id, "frames must be a non-empty list", line=line, field="frames")
        ]
    fallback_seed = raw.get("noise_seed", 0)
    if not isinstance(fallback_seed, int) or isinstance(fallback_seed, bool):
        errors.append(
            _seq_failure(
                sequence_id,
                "noise_seed must be an integer",
                line=line,
                field="noise_seed",
            )
        )
        fallback_seed = 0
    normalized_frames: list[dict[str, Any]] = []
    for frame_index, frame_raw in enumerate(frames_raw):
        if not isinstance(frame_raw, dict):
            errors.append(
                _seq_failure(
                    sequence_id,
                    f"frames[{frame_index}] must be a mapping",
                    line=line,
                    field=f"frames[{frame_index}]",
                )
            )
            continue
        trajectories_raw = frame_raw.get("trajectories")
        if not isinstance(trajectories_raw, list) or not trajectories_raw:
            errors.append(
                _seq_failure(
                    sequence_id,
                    f"frames[{frame_index}].trajectories must be a non-empty list",
                    line=line,
                    field=f"frames[{frame_index}].trajectories",
                )
            )
            continue
        trajectories: list[dict[str, Any]] = []
        for trajectory_index, trajectory_raw in enumerate(trajectories_raw):
            normalized = _normalize_trajectory(
                trajectory_raw, sequence_id, line, trajectory_index, errors
            )
            if normalized is not None:
                trajectories.append(normalized)
        blocks_raw = frame_raw.get("color_blocks", [])
        if not isinstance(blocks_raw, list):
            errors.append(
                _seq_failure(
                    sequence_id,
                    f"frames[{frame_index}].color_blocks must be a list",
                    line=line,
                    field=f"frames[{frame_index}].color_blocks",
                )
            )
            continue
        blocks: list[dict[str, Any]] = []
        for block_index, block_raw in enumerate(blocks_raw):
            normalized_block = _normalize_block(
                block_raw, sequence_id, line, block_index, errors
            )
            if normalized_block is not None:
                blocks.append(normalized_block)
        noise_seed = frame_raw.get("noise_seed", fallback_seed)
        if not isinstance(noise_seed, int) or isinstance(noise_seed, bool):
            errors.append(
                _seq_failure(
                    sequence_id,
                    f"frames[{frame_index}].noise_seed must be an integer",
                    line=line,
                    field=f"frames[{frame_index}].noise_seed",
                )
            )
            continue
        if len(trajectories) != len(trajectories_raw) or len(blocks) != len(blocks_raw):
            continue
        normalized_frames.append(
            {
                "trajectories": trajectories,
                "color_blocks": blocks,
                "noise_seed": int(noise_seed),
            }
        )
    if errors:
        return None, errors
    if len(normalized_frames) != len(frames_raw):
        return None, [
            _seq_failure(sequence_id, "one or more frames could not be normalized", line=line)
        ]
    return SequenceSpec(sequence_id=sequence_id, frames=tuple(normalized_frames), line=line), []


def parse_sequence_lines(lines: list[tuple[int, str]]) -> SequenceLoadResult:
    failures: list[dict[str, Any]] = []
    sequences: list[SequenceSpec] = []
    seen_ids: set[str] = set()
    for line_number, text in lines:
        if not text.strip():
            continue
        fallback_id = f"@line:{line_number:08d}"
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            failures.append(
                _seq_failure(fallback_id, f"invalid JSON: {exc.msg}", line=line_number)
            )
            continue
        sequence_id_raw = raw.get("sequence_id") if isinstance(raw, dict) else None
        sequence_id = sequence_id_raw if isinstance(sequence_id_raw, str) and sequence_id_raw else fallback_id
        if not isinstance(sequence_id_raw, str) or not sequence_id_raw:
            failures.append(
                _seq_failure(
                    sequence_id,
                    "sequence_id must be a non-empty string",
                    line=line_number,
                    field="sequence_id",
                )
            )
            continue
        if sequence_id in seen_ids:
            failures.append(
                _seq_failure(
                    sequence_id,
                    "duplicate sequence_id; first valid occurrence is retained",
                    line=line_number,
                    field="sequence_id",
                )
            )
            continue
        sequence, sequence_errors = _normalize_sequence(raw, sequence_id, line_number)
        if sequence_errors:
            failures.extend(sequence_errors)
            continue
        assert sequence is not None
        seen_ids.add(sequence_id)
        sequences.append(sequence)
    sequences.sort(key=lambda item: item.sequence_id)
    return SequenceLoadResult(
        sequences=sequences,
        failures=sort_failures(failures),
        total_lines=len(lines),
    )


def load_sequences(path: str | Path) -> SequenceLoadResult:
    try:
        raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return SequenceLoadResult(
            sequences=[],
            failures=[_seq_failure("@file", f"cannot read sequence file: {exc}")],
            total_lines=0,
        )
    lines = list(enumerate(raw_lines, start=1))
    return parse_sequence_lines(lines)

