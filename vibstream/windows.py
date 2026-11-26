"""Rolling window construction, overlap references, and real eviction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .blocks import SignalBlock
from .canonical import sha256_value
from .config import AppConfig


@dataclass(frozen=True)
class SampleCell:
    value: float
    block_id: str
    arrival_seq: int
    resampled: bool


@dataclass(frozen=True)
class Window:
    window_id: str
    index: int
    start_sample_by_channel: dict[str, int]
    end_sample_by_channel: dict[str, int]
    block_refs: tuple[str, ...]
    first_arrival_seq: int
    anchor_arrival_seq: int

    def geometry(self, config: AppConfig) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "index": self.index,
            "length": config.window.length,
            "step": config.window.step,
            "ranges": {
                channel: [self.start_sample_by_channel[channel], self.end_sample_by_channel[channel]]
                for channel in sorted(self.start_sample_by_channel)
            },
            "block_refs": sorted(self.block_refs),
        }

    def geometry_fingerprint(self, config: AppConfig) -> str:
        return sha256_value({"kind": "window-geometry", "value": self.geometry(config)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "index": self.index,
            "start_sample_by_channel": dict(sorted(self.start_sample_by_channel.items())),
            "end_sample_by_channel": dict(sorted(self.end_sample_by_channel.items())),
            "block_refs": sorted(self.block_refs),
            "first_arrival_seq": self.first_arrival_seq,
            "anchor_arrival_seq": self.anchor_arrival_seq,
        }


@dataclass(frozen=True)
class WindowBuild:
    retained_windows: tuple[Window, ...]
    candidate_windows: tuple[Window, ...]
    evictions: tuple[dict[str, Any], ...]
    introductions: tuple[dict[str, Any], ...]
    sample_map: dict[str, dict[int, SampleCell]]
    block_arrival_by_id: dict[str, int]


def _start_index(acquisition_ms: int, sample_rate_hz: int) -> int:
    return (acquisition_ms * sample_rate_hz) // 1000


def _candidate_indexes(block: SignalBlock, config: AppConfig) -> set[int]:
    candidates: set[int] = set()
    length = config.window.length
    step = config.window.step
    for channel in config.channels:
        start = _start_index(block.acquisition_ms, channel.sample_rate_hz)
        end = start + block.length_for(channel.channel_id) - 1
        low_numerator = start - length + 1
        low = max(0, (low_numerator + step - 1) // step) if low_numerator > 0 else 0
        high = end // step
        if high >= low:
            candidates.update(range(low, high + 1))
    return candidates


def _build_sample_map(blocks: Iterable[SignalBlock], config: AppConfig) -> dict[str, dict[int, SampleCell]]:
    sample_map: dict[str, dict[int, SampleCell]] = {channel.channel_id: {} for channel in config.channels}
    for block in sorted(blocks, key=lambda item: (item.arrival_seq, item.block_id)):
        missing_by_channel: dict[str, list[tuple[int, int]]] = {}
        for segment in block.missing_segments:
            missing_by_channel.setdefault(segment.channel, []).append((segment.start, segment.end))
        resampling_by_channel: dict[str, list[tuple[int, int]]] = {}
        for marker in block.resampling_markers:
            resampling_by_channel.setdefault(marker.channel, []).append((marker.start, marker.end))
        for channel in config.channels:
            channel_id = channel.channel_id
            start = _start_index(block.acquisition_ms, channel.sample_rate_hz)
            missing = missing_by_channel.get(channel_id, [])
            markers = resampling_by_channel.get(channel_id, [])
            for local_index, value in enumerate(block.signals[channel_id]):
                if any(local_start <= local_index < local_end for local_start, local_end in missing):
                    continue
                resampled = any(
                    local_start <= local_index < local_end for local_start, local_end in markers
                )
                sample_map[channel_id][start + local_index] = SampleCell(
                    value=value,
                    block_id=block.block_id,
                    arrival_seq=block.arrival_seq,
                    resampled=resampled,
                )
    return sample_map


def _make_window(index: int, blocks: list[SignalBlock], config: AppConfig) -> Window:
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    refs: set[str] = set()
    first_arrival: int | None = None
    anchor_arrival = 0
    for channel in config.channels:
        channel_id = channel.channel_id
        start = index * config.window.step
        end = start + config.window.length
        starts[channel_id] = start
        ends[channel_id] = end
        for block in blocks:
            block_start = _start_index(block.acquisition_ms, channel.sample_rate_hz)
            block_end = block_start + block.length_for(channel_id)
            if max(start, block_start) < min(end, block_end):
                refs.add(block.block_id)
                if first_arrival is None or block.arrival_seq < first_arrival:
                    first_arrival = block.arrival_seq
                anchor_arrival = max(anchor_arrival, block.arrival_seq)
    return Window(
        window_id=f"w{index:06d}",
        index=index,
        start_sample_by_channel=starts,
        end_sample_by_channel=ends,
        block_refs=tuple(sorted(refs)),
        first_arrival_seq=first_arrival or 0,
        anchor_arrival_seq=anchor_arrival,
    )


def build_windows(blocks: Iterable[SignalBlock], config: AppConfig) -> WindowBuild:
    ordered_blocks = sorted(blocks, key=lambda item: (item.arrival_seq, item.block_id))
    sample_map = _build_sample_map(ordered_blocks, config)
    known_indexes: set[int] = set()
    introductions: list[dict[str, Any]] = []
    evictions: list[dict[str, Any]] = []
    active_indexes: set[int] = set()
    newly_added_by_block: dict[str, list[int]] = {}

    for block in ordered_blocks:
        candidates = sorted(_candidate_indexes(block, config) - known_indexes)
        newly_added_by_block[block.block_id] = candidates
        for index in candidates:
            known_indexes.add(index)
            active_indexes.add(index)
            introductions.append(
                {
                    "window_id": f"w{index:06d}",
                    "window_index": index,
                    "trigger_block": block.block_id,
                    "trigger_arrival_seq": block.arrival_seq,
                }
            )
        while len(active_indexes) > config.window.retention_depth:
            evicted = min(active_indexes)
            active_indexes.remove(evicted)
            evictions.append(
                {
                    "window_id": f"w{evicted:06d}",
                    "window_index": evicted,
                    "trigger_block": block.block_id,
                    "trigger_arrival_seq": block.arrival_seq,
                    "reason": "retention_depth_exceeded",
                    "retained_depth": config.window.retention_depth,
                }
            )

    candidate_windows = tuple(_make_window(index, ordered_blocks, config) for index in sorted(known_indexes))
    retained_indexes = sorted(set(active_indexes))
    retained_windows = tuple(_make_window(index, ordered_blocks, config) for index in retained_indexes)
    return WindowBuild(
        retained_windows=retained_windows,
        candidate_windows=candidate_windows,
        evictions=tuple(evictions),
        introductions=tuple(introductions),
        sample_map=sample_map,
        block_arrival_by_id={block.block_id: block.arrival_seq for block in ordered_blocks},
    )
