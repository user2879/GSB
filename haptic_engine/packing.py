"""Frame binning, run-length merging and deterministic overflow splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Device


@dataclass(frozen=True)
class Command:
    """One driver command covering a contiguous half-open sample interval."""

    start_sample: int
    end_sample: int
    duration_samples: int
    level: int

    def to_dict(self) -> dict:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "duration_samples": self.duration_samples,
            "level": self.level,
        }


@dataclass(frozen=True)
class Block:
    """A frame block: the commands physically emitted inside one frame slot."""

    block_index: int
    nominal_frame: int
    is_overflow: bool
    frame_start_sample: int
    frame_end_sample: int
    commands: tuple[Command, ...]

    @property
    def command_count(self) -> int:
        return len(self.commands)

    def covered_range(self) -> tuple[int, int]:
        return self.commands[0].start_sample, self.commands[-1].end_sample

    def to_dict(self) -> dict:
        return {
            "block_index": self.block_index,
            "nominal_frame": self.nominal_frame,
            "is_overflow": self.is_overflow,
            "frame_start_sample": self.frame_start_sample,
            "frame_end_sample": self.frame_end_sample,
            "command_count": self.command_count,
            "commands": [command.to_dict() for command in self.commands],
        }


def block_from_dict(payload: dict) -> Block:
    return Block(
        block_index=payload["block_index"],
        nominal_frame=payload["nominal_frame"],
        is_overflow=payload["is_overflow"],
        frame_start_sample=payload["frame_start_sample"],
        frame_end_sample=payload["frame_end_sample"],
        commands=tuple(
            Command(
                start_sample=item["start_sample"],
                end_sample=item["end_sample"],
                duration_samples=item["duration_samples"],
                level=item["level"],
            )
            for item in payload["commands"]
        ),
    )


def _run_length_segments(levels: np.ndarray) -> list[tuple[int, int, int]]:
    """Return ``(start, end, level)`` runs over contiguous equal samples."""
    if levels.size == 0:
        return []
    change_points = np.flatnonzero(np.diff(levels)) + 1
    boundaries = np.concatenate(([0], change_points, [levels.size]))
    segments: list[tuple[int, int, int]] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        segments.append((int(left), int(right), int(levels[int(left)])))
    return segments


def build_blocks(levels: np.ndarray, device: Device) -> tuple[list[Block], int]:
    """Pack a per-sample level array into frame blocks.

    Samples are binned by frame duration. Equal contiguous quantized levels are
    merged into a single duration-carrying command. Runs crossing a frame
    boundary are split, and a frame exceeding ``max_commands_per_frame`` pushes
    its surplus commands into deterministic follow-up overflow blocks. Sample
    intervals are never shortened or dropped, so the expanded sequence is
    identical to issuing every sample individually.
    """
    frame_size = device.frame_samples
    total_samples = int(levels.size)
    nominal_frame_count = max(1, (total_samples + frame_size - 1) // frame_size)

    pending: list[tuple[int, int, int]] = []
    blocks: list[Block] = []
    overflow_count = 0

    for nominal_frame in range(nominal_frame_count):
        frame_start = nominal_frame * frame_size
        frame_end = min((nominal_frame + 1) * frame_size, total_samples)
        frame_levels = levels[frame_start:frame_end]
        for start, end, level in _run_length_segments(frame_levels):
            pending.append((frame_start + start, frame_start + end, level))

        # Repeatedly fill frame slots; surplus commands become overflow blocks.
        slot_frame = nominal_frame
        while pending:
            capacity = device.max_commands_per_frame
            chosen = pending[:capacity]
            pending = pending[capacity:]
            is_overflow = slot_frame != nominal_frame
            if is_overflow:
                overflow_count += len(chosen)
            commands = tuple(
                Command(start_sample=start, end_sample=end, duration_samples=end - start, level=level)
                for start, end, level in chosen
            )
            blocks.append(
                Block(
                    block_index=len(blocks),
                    nominal_frame=nominal_frame,
                    is_overflow=is_overflow,
                    frame_start_sample=frame_start,
                    frame_end_sample=frame_end,
                    commands=commands,
                )
            )
            slot_frame += 1

    return blocks, overflow_count


def expand_blocks(blocks: list[Block], total_samples: int) -> np.ndarray:
    """Reconstruct the per-sample level array from packed blocks."""
    expanded = np.empty(total_samples, dtype=np.int64)
    for block in blocks:
        for command in block.commands:
            expanded[command.start_sample:command.end_sample] = command.level
    return expanded


def validate_packing(blocks: list[Block], levels: np.ndarray) -> None:
    """Assert byte-for-byte equivalence between packed and raw sample streams."""
    covered = np.zeros(levels.size, dtype=bool)
    for block in blocks:
        for command in block.commands:
            if command.end_sample - command.start_sample != command.duration_samples:
                raise ValueError("command duration does not match sample interval")
            covered[command.start_sample:command.end_sample] = True
    if not np.all(covered):
        raise ValueError("packed blocks leave samples uncovered")
    if not np.array_equal(expand_blocks(blocks, levels.size), levels):
        raise ValueError("packed stream differs from per-sample stream")


def validate_device_packing(blocks: list[Block], levels: np.ndarray, device: Device) -> None:
    for block in blocks:
        if block.command_count > device.max_commands_per_frame:
            raise ValueError("frame block exceeds per-frame command budget")
        if block.frame_end_sample - block.frame_start_sample > device.frame_samples:
            raise ValueError("frame block spans more samples than one frame")
    validate_packing(blocks, levels)
